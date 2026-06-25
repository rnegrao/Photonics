import numpy as np
import matplotlib.pylab as plt
import scipy.constants as constant

def epsilon_r(z_val):
    epsilon_vacuum = np.empty_like(z_val)
    epsilon_r = np.empty_like(z_val)
    epsilon_vacuum_mask = np.abs(z_val) >= 2.0
    epsilon_vacuum[epsilon_vacuum_mask] = 1.0 
    epsilon_r_mask = np.abs(z_val) < 2.0
    epsilon_r[epsilon_r_mask] = 6.0
    epsilon_i = epsilon_r + epsilon_vacuum

    return epsilon_i

def mu_r(z_val):
    mu_vacuum = np.empty_like(z_val)
    mu_r = np.empty_like(z_val)
    mu_vacuum_mask = np.abs(z_val) >= 2.0
    mu_vacuum[mu_vacuum_mask] = 1.0 
    mu_r_mask = np.abs(z_val) < 2.0
    mu_r[mu_r_mask] = 2.0
    mu_i = mu_r + mu_vacuum
    return mu_i

if __name__ == "__main__":
    '''compute grid resolution'''
    h = 10.e-9 
    dz = h
    '''compute timestep - Courant stability condition'''
    dt = h/( 2. * constant.speed_of_light )
    t_val = np.arange(0. , 15.e-14, dt)
    '''1D medium geometry definition'''
    L = 5e-6
    z_val = np.arange(-L, L, dz)
    epsilon_r = epsilon_r(z_val)
    mu_r = mu_r(z_val)
    '''compute coefficients'''
    me = constant.speed_of_light * dt / ( epsilon_r * dz )
    mh = constant.speed_of_light * dt / ( mu_r * dz )
    '''source settings (gaussian pulse) 750 THz (UV)  '''
    source_pos = z_val.size // 8
    f = 750e12
    t0 = 6 * 0.5 / f       
    spread = 0.5 / f
    A = - np.sqrt( epsilon_r[source_pos] / mu_r[source_pos] )
    '''initialize field components'''
    Ez = np.zeros(z_val.size)
    Hy = np.zeros(z_val.size)
    Ez_source = np.zeros(z_val.size)
    
    plt.figure(figsize=(10,4))        
    for n in range(t_val.size):     
        '''update Magnetic Field Hy using spatial derivatives of Ez'''
        for i in range(z_val.size-1):
            Hy[i] = Hy[i] - mh[i] * ( Ez[i+1] - Ez[i] )
        Hy[1000] = Hy[1000] - mh[1000] * ( 0.0 - Ez[1000] )  
        
        '''update Electric Field Ez using spatial derivatives of Hy'''
        Ez[0] = Ez[0] - me[0] * ( Hy[0] - 0.0 )
        for i in range(1, z_val.size):
            Ez[i] = Ez[i] - me[i] * ( Hy[i] - Hy[i-1] )
            
        '''inject Source Wave (source at the center)''' 
        pulse = A * np.exp( - (( t_val[n] - t0)/spread ) ** 2 )
        Ez_source[source_pos] = pulse
        Ez = Ez + Ez_source

        '''plot final timestep'''
        plt.clf()
        plt.title(f"1D FDTD Ex/Hy Wave Propagation (Gaussian Pulse). Time: {t_val[n]} (s)")
        plt.xlabel("Position z [\mu m]")
        plt.ylabel("Field Amplitude")
        plt.plot(z_val*1e6, Ez, label="Electric field [Ex(z)]", color = "black")
        plt.plot(z_val*1e6, Hy, label="Magnetic field [Hy(z)]", color = "red", linestyle = "--")
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.pause(1e-7)
