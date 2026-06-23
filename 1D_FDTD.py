import numpy as np
import matplotlib.pylab as plt
import scipy.constants as constant

if __name__ == "__main__":
    '''compute grid resolution'''
    h = 10.e-9 
    dz = h
    '''compute timestep'''
    dt = h/( 2. * constant.speed_of_light )
    t_val = np.arange(0. , 5.e-14, dt)
    '''1D medium geometry definition'''
    L = 2.5e-6
    z_val = np.arange(-10*L, 10*L, dz)
    '''source settings (gaussian pulse) 750 THz (UV)  '''
    source_pos = z_val.size // 2
    f = 750e12
    t0 = 6 * 0.5 / f       
    spread = 0.5 / f
    '''compute coefficients'''
    epsilon_r = 1.
    mu_r = 1.
    me = constant.speed_of_light * dt / ( epsilon_r * dz )
    mh = constant.speed_of_light * dt / ( mu_r * dz )
    '''initialize field components'''
    Ez = np.zeros(z_val.size)
    Hy = np.zeros(z_val.size)

    plt.ion()
    plt.figure(figsize=(10,4))

        
    for n in range(t_val.size):     
        '''update Electric Field Ez using spatial derivatives of Hy'''
        for i in range(1, z_val.size-1):
            Ez[i] = Ez[i] - me * ( Hy[i] - Hy[i-1] )
            
        '''update Magnetic Field Hy using spatial derivatives of Ez'''
        for i in range(z_val.size-1):
            Hy[i] = Hy[i] - mh * ( Ez[i+1] - Ez[i] )
            
        '''inject Source Wave (source at the center)'''
        pulse = np.exp( - (( t_val[n] - t0)/spread ) ** 2 )
        Ez[source_pos] = pulse

        '''plot final timestep'''
        plt.clf()
        plt.title(f"1D FDTD Ex/Hy Wave Propagation (Gaussian Pulse). Time: {t_val[n]}")
        plt.xlabel("Position z [\mu m]")
        plt.ylabel("Field Amplitude")
        plt.plot(z_val*1e6, Ez, label="Electric field [Ex(z)]", color = "black")
        plt.plot(z_val*1e6, Hy, label="Magnetic field [Hy(z)]", color = "red", linestyle = "--")
        plt.grid(True)
        plt.legend()
        plt.draw()
        plt.pause(0.01)
