# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:58:19 2022

@author: chris
"""
import numpy as np
import start_velocity as sv
from matplotlib import pyplot as plt


def boundary(numbc, u):
    u = np.append(u, u[:,-numbc:], axis=1)
    u = np.append(u, u[:,:numbc], axis=1)
    return u


def initial(x, n):
    u0 = sv.start_rieman_easy(len(x), 0.6/n, 0.95/n, dim=n)
    u0 = sv.start_barrier_easy(len(x), 0.3/n, 0.6/n, RHS=0.95/n, dim=n)
    return u0


def feq(V, r, n, nx):
    func = np.zeros((n,nx))
    func[n-1,:] = r * (1-r) #** 2
    func[0,:] = r-func[n-1,:]
    return func


def Lambda(u, V, n, nx):
    u = np.append(u, np.ones((1,nx)), axis=0)
    lam = np.empty((0, nx))
    for idx in range(n - 1):
        sum_ = 0 
        for jdx in range(idx+1, n):
            sum_ += (V[jdx] - V[idx]) * (1/u[jdx,:] - 1/u[jdx+1,:])
        lam = np.append(lam, (V[idx] - sum_).reshape((1, nx)), axis=0)

    lam = np.append(lam, np.ones((1, nx)), axis=0)
    return lam
    

def lambda_max(f, V):
    rho = sum(f)
    q = np.dot(f.T, V)
    spd = -1/(1-rho) * q
    
    lammax = max(abs(spd))
    return max(lammax, 1)


def Fsys(u, V, n, nx):
    func = Lambda(u, V, n, nx) * u
    return func
    
    
def fluxJX(u, v, dt, dx, V, n, nx):
    
    ustar = (u + v)/2 - dt/dx/2 * ( Fsys(v,V,n,nx) - Fsys(u,V,n,nx) );  
    flx = 1/2 * (Fsys(ustar,V,n,nx) + 1/2 * (Fsys(u,V,n,nx) + Fsys(v,V,n,nx))) \
            - dx/dt/4 * (v - u)
    
    return flx


def main(N, nx, nbc, x_range=[0,1], t_end=0.1, T=0.001, EASYMODE=True):
    
    n = N+1
    nx_inner = nx - 2 * nbc
    x_low = x_range[0]
    x_up = x_range[1]
    dx = (x_up - x_low) / nx_inner
    
    #inter = range(0, nx_inner+1)
    
    x = np.linspace(x_low + dx/2, x_up - dx/2, nx_inner)
    x_long = np.linspace(x_low - dx/2, x_up + dx/2, nx_inner + 2)

    f = initial(x, n)
    f = boundary(nbc, f)
    
    u = (1 - sum(f)).reshape(1,nx)
    V = np.linspace(0, 1, num=n)
    
    q = np.dot(f.T, V)
       
    psum = np.zeros((1, nx))
    
    for k in range(1, n):
        psum += f[k-1, :]
        u = np.append(u, u[0,:] / (1-psum), axis=0)
    
    t_begin = 0
    time = t_begin
    timeline = np.array([time])
    it = 1
    
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    
    fig.suptitle(f'time = {time}')
    
    line0, = ax[0].plot(x, (1-u[0,:])[:-2*nbc])
    ax[0].set_ylabel('rho')
    ax[0].set_xlabel('x')
    ax[0].grid('true')
    ax[0].set_ylim((0,1))
    
    line1, = ax[1].plot(x, q[:-2*nbc])
    ax[1].set_ylabel('q')
    ax[1].set_xlabel('x')
    ax[1].grid('true')
    ax[1].set_ylim((0,1))
    
    plt.show()
    
    while time < t_end:
        
        dt = 0.5 * 0.9 * dx / lambda_max(f,V)
        dt = min( dt , t_end-time )
        
        helper = np.append(u[:,-1].reshape(n, 1), u[:,:nx_inner], axis=1)
        #helper = np.append( u[:,:nx_inner], u[:,-1].reshape(n, 1), axis=1)
        
        flux = fluxJX(helper, u[:, :nx_inner+1], dt, dx, V, n, nx_inner+1);
        u[:,:nx_inner] = u[:,:nx_inner] - dt/dx * (flux[:,1:]-flux[:,:-1]);
        
        f[n-1,:] = u[0,:] * (1/u[n-1,:] - 1)
        for k in range(n-1):
            f[k,:] = u[0,:] * (1/u[k,:] - 1/u[k+1,:])
        
        f = boundary(nbc, f[:,:-2*nbc])
        
        r= 1-u[0,:]
        q = np.dot(f.T, V)
        rel = feq(V, r, n, nx)
        
        f = ( T*f + dt*rel ) / (T + dt);
        
        #u[0,:] = 1-sum(f)
        u[0,:] = 1 - f.sum(axis=0)
        
        psum = np.zeros((1,nx))
        
        for k in range(1, n):
            psum += f[k-1, :]
            
            if not EASYMODE:
                u[k,:] = u[0,:] / (1-psum)
        
        time += dt
        it += 1
        #timeline = np.append(timeline, [time])
        
        pics = 100
        
        if it % pics == 0 or time >= t_end:
            line0.set_ydata( (1-u[0,:])[:-2*nbc])
            line1.set_ydata( q[:-2*nbc] )
            fig.suptitle(f'time = {time}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
           # print(max((1-u[0,:])[:-2*nbc]), min((1-u[0,:])[:-2*nbc]))
           # break
       
    plt.savefig('Test2.png')

    #ax[0].plot(x, (1-u[0,:])[:-2*nbc])
    #ax[1].plot(x, q[:-2*nbc])
    #plt.show()

    


if __name__ == '__main__':
    start = sv.start_rieman_easy(1000, 0.3, 0.475, dim=2)
    
    #print(boundary(2, start))
    
    main(2, 10000, 1, T=.1, EASYMODE=False)
