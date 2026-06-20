import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.constants import constants
import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''inputs'''
    g = {
        'W_um' : 0.85,
        'H_um' : 0.4,
        'Lambda_um' : 0.3,
        'L_cav_um' : 6.0,
        'L_ref_um' : 3.0,
        'l_min_nm' : 150.0,
        'l_max_nm' : 170.0,
        't_inse_nm' : 12.5,
        't_sio2_um' : 2.0,
        'lambda0_nm' : 927.0,
        }
    wavelengths_nm = np.linspace(920, 960, 40)
    n_positions = 17
    n_grid = 10000
    
    '''geometrical parameters
        Cavity:
            ***######|######***
    '''
    Lc = float(g['L_cav_um'])
    Lr = float(g['L_ref_um'])
    Lam = float(g['Lambda_um'])
    lmin = float(g['l_min_nm'])
    lmax = float(g['l_max_nm'])
    H = float(g['H_um'])
    W = float(g['W_um'])
    t_inse = float(g['t_inse_nm'])
    lambda0 = float(g['lambda0_nm'])

    '''create the grid and binary mask (True/False) for the cavity and reflector'''
    x_max = 0.5 * Lc + Lr
    s = np.linspace(-1., 1., n_grid)
    beta = 2.6
    x = x_max * np.tanh(beta * s) / np.tanh(beta)
    x[0] = -x_max
    x[-1] = x_max
    cavity_mask = np.abs(x) <= 0.5 * Lc
    reflector_mask = ~cavity_mask
        
    '''quantum dot profiles inside cavity (cavity_mask = True)'''
    l_profile = np.full_like(x, lmin)
    xc = np.zeros_like(x)
    xc[cavity_mask] = x[cavity_mask] / (0.5 * Lc)
    l_profile[cavity_mask] = lmin + (lmax - lmin) * (1. - xc[cavity_mask] ** 2)
        
    '''reflector profiles oustide the cavity'''
    xr = np.abs(x[reflector_mask]) - 0.5 * Lc
    phase = 2. * np.pi * xr / Lam
    refl_mod = 0.5 * ( 1. + np.cos(phase) )
    l_profile[reflector_mask] = lmin + (lmax - lmin) * refl_mod

    '''make sure the profile is symetric'''
    l_profile = 0.5 * (l_profile + l_profile[::-1])

    '''define model objects'''
    n_bg = 1.86 + 0.015 * ( (wavelengths_nm - lambda0) / lambda0 )
    n_inse = 2.65 + 0.04 * np.exp(-((wavelengths_nm -lambda0) / 30. )**2)
    confinement = 0.07 * (W / 0.85) + 0.05 * (H / 0.4) + 0.03 * (t_inse / 12.5)

    '''potential parametrization'''
    thickness_norm = (l_profile - lmin) / max(lmax - lmin,1e-12)
    potential_shape = 0.65 * thickness_norm + 0.35 * thickness_norm ** 2
    interface_boost = np.exp(-((np.abs(x) - 0.5 * Lc) / 0.18) ** 2) 
    center_boost = np.exp(-(x / (0.19 * Lc + 1e-12)) ** 2)
    v_geo = potential_shape + 0.18 * interface_boost + 0.42 * center_boost
    v_geo = 0.5 * (v_geo + v_geo[::-1])

    '''define derivatives for numerical calculations'''
    dx = np.diff(x)
    dxm = np.empty_like(x)
    dxp = np.empty_like(x)
    dxm[1:] = dx
    dxm[0] = dx[0]
    dxp[:-1] = dx
    dxp[-1] = dx[-1]

    '''parametrization for sparse matrix for partial differential equation'''
    a = 1. + 0.12 * thickness_norm + 0.08 * center_boost
    a_half_plus = 0.5 * ( a[:-1] + a[1:] )
    a_half_minus = a_half_plus.copy() 
    
    main_kin = np.zeros(n_grid)
    upper_kin = np.zeros(n_grid - 1)
    lower_kin = np.zeros(n_grid - 1)
    for i in range(1, n_grid -1):
        hm = x[i] - x[i-1] 
        hp = x[i+1] - x[i] 
        cim = - 2.0 * a_half_minus[i-1] / (hm * (hm + hp))
        cip = - 2.0 * a_half_plus[i] / (hp * (hm + hp))
        cii = - cim - cip
        lower_kin[i-1] = cim
        main_kin[i] = cii
        upper_kin[i] = cip
    main_kin[0] = 1.
    main_kin[-1] = 1.
    
    '''define matrix'''
    K = sparse.diags([lower_kin, main_kin, upper_kin], offsets=[-1, 0, 1], format = 'csc')
    
    '''define n_position gaussians for the quantum dot positions'''
    pos = np.linspace(-0.44 * Lc, 0.44 * Lc, n_positions)
    sigma_src = max(0.06 * Lc, 0.12)
    '''n_grid x n_position matrix'''
    source_matrix = np.exp(-0.5 * ( (x[:, None] - pos[None, :]) / sigma_src ) ** 2)
    source_matrix /= np.maximum( np.trapezoid(source_matrix, x, axis = 0 ), 1e-15)

    '''define more profiles for detected and guided waveprofiles'''
    det_profile = 0.6 * center_boost + 0.25 * potential_shape + 0.15 * interface_boost
    det_profile = np.maximum(det_profile, 0.)
    guide_profile = 0.55 * center_boost + 0.25 * np.exp(-(( x - 0.18 * Lc ) / (0.16 * Lc + 1e-12)) ** 2) + 0.2 * np.exp(-((x + 0.18 * Lc) / (0.16 * Lc + 1e-12)) ** 2)
    guide_profile = np.maximum(guide_profile, 0.0)

    '''define data structures for total and guided waves'''
    M = wavelengths_nm.size
    local_total = np.zeros((n_positions, M), dtype = float)
    local_guided = np.zeros((n_positions, M), dtype = float)
    ref_total = np.zeros(M, dtype = float)
    rel_trans = np.zeros(M, dtype = float)

    '''define coupling parameter model proportional to confinement'''
    I = sparse.identity(n_grid, format = 'csc')
    eps_reg = 4.e-3
    q_couple = 0.12 + 0.03 * confinement
    spectral_width_nm = 1.35 + 0.06 * abs(lmax - lmin)
    
    for j, lam in enumerate(wavelengths_nm):
        k0 = 2.0 * np.pi / max(lam * 1e-3, 1e-12)
        detune = (lam - lambda0) / spectral_width_nm
        mode_amp = 1.0 / np.sqrt(detune ** 2 + eps_reg ** 2)        
        v_lam = (k0 ** 2) * (0.08 + confinement + 0.22 * ( n_inse[j] - n_bg[j] ) + q_couple * v_geo)
        Vdiag = sparse.diags( v_lam, format = 'csc' )
        '''define matrix to be solved. Add epsilon*identity matrix to improve numerical stability'''
        A = (K + Vdiag + eps_reg * I).tocsc()
        '''A matrix inversion = invA'''
        lu = splu(A)
        '''operator applied on the source to solve the field invA(source) = field'''
        field = lu.solve(source_matrix)
        field_sq = field ** 2
        
        tot_j = np.trapezoid( field_sq * det_profile[:, None], x, axis = 0)
        guid_j = np.trapezoid( field_sq * guide_profile[:, None] , x, axis = 0)
        resonant_factor = 1. + 0.14 * mode_amp * (1. + 0.25 * np.cos(np.pi * pos / (0.5 * Lc + 1e-12)) ** 2)
        tot_j = np.maximum(tot_j * resonant_factor, 0.)
        guid_j = np.maximum(guid_j * (1. + 0.22 * mode_amp), 0.)
        
        ref_strength = 0.35 + 0.1 * confinement + 0.03 * (n_inse[j] - n_bg[j])
        Aref = (K + sparse.diags((k0 ** 2) * (0.22 + ref_strength) * np.ones(n_grid), format = 'csc') + 1.8e-2 * I).tocsc()
        luref = splu(Aref)
        field_ref = luref.solve(source_matrix)
        field_ref_sq = field_ref ** 2
        
        ref_j_pos = np.trapezoid((field_ref ** 2) * (0.35 + 0.25 * center_boost)[:, None], x, axis = 0)
        ref_total[j] = np.mean(np.maximum(ref_j_pos, 1e-14))
        
        local_total[:, j] = np.maximum(tot_j, 1e-14)
        local_guided[:, j] = np.minimum(np.maximum(guid_j, 0.), local_total[:, j])
        broad_bg = 0.75 + 0.08 * np.cos(2. * np.pi * (lam - wavelengths_nm[0]) / max(wavelengths_nm[-1] - wavelengths_nm[0], 1e-12))
        narrow_peak = 0.75 * (mode_amp / np.max([mode_amp, 1e-12]))
        rel_trans[j] = broad_bg + narrow_peak
    
    def apply_total(weights):
        w = np.asarray(weights, dtype = float)
        if w.shape != (n_positions, ):
            raise ValueError("weights must have shape (n_position,)")
        w = np.maximum(w, 0.)
        if np.sum(w) <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / np.sum(w)
        return np.sum(local_guided * w[:, None], axis = 0)

    def apply_guided(weights):
        w = np.asarray(weights, dtype = float)
        if w.shape != (n_positions, ):
            raise ValueError("weights must have shape (n_positions,)")
        w = np.maximum(w, 0.)
        if np.sum(w) <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / np.sum(w)
        return np.sum(local_guided * w[:, None], axis = 0)

    operators = {
        'x_um': x,
        'l_profile_um': l_profile,
        'positions_um': pos,
        'local_total': local_total,
        'local_guided': local_guided,
        'reference_power': ref_total,
        'relative_transmission': rel_trans,
        'geometry': g,
        'apply_total': apply_total,
        'apply_guided': apply_guided,
    }
         
