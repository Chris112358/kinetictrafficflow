# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:19:33 2022

@author: chris
"""

import numpy as np
from start_velocity import myexception

def F_LLF(u, v, f, Df):
    s = max(abs(Df(u)), abs(Df(v)))
    F = ( f(u) + f(v) - s * (v - u)) / 2
    return F


#new schemes
def Lambda(u, V):
    n, nx = u.shape
    
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


def feq(V, r):
    n = len(V)
    nx = len(r)
    func = np.zeros((n,nx))
    #func[-1,:] = r * (1-r) **2
    func[-1,:] = (-16*(r**4) + 32*(r**3) - 21*(r**2) + 5*r) 
    
    
    func[0,:] = r-func[-1,:]
    return func


def Fsys(u, V):
    func = Lambda(u, V) * u
    return func
    
    
def fluxJX(u, v, dt, dx, V):
    
    ustar = (u + v)/2 - dt/dx/2 * ( Fsys(v,V) - Fsys(u,V) );  
    flx = 1/2 * (Fsys(ustar,V) + 1/2 * (Fsys(u,V) + Fsys(v,V))) \
            - dx/dt/4 * (v - u)
    
    return flx


def step_easy(u, dt, dx, V, nx_inner, method='JX'):
    ''' same as step, but only one call is necessary / all calculation is done here''' 
    if method == 'JX':
        
        '''n_s has to be the index of first inner point, n_e is n_s + number of inner points''' 
        n_s, n_e = nx_inner
        
        flux = fluxJX(u[:, n_s-1:n_e], u[:, n_s:n_e+1], dt, dx, V);
        u[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux[:,1:]-flux[:,:-1]);
        
    return u
    
    
def step(u_l, u_m, u_r, f, Df, lam, method='LLF'):
    ''' calculate the step with a given method
    of the form:  d_t u + d_x (f(u)) = 0
    
       u_l is u_-1, 
       u_m is u_0,
       u_r is u_+1,
       
       lam is DT/DX
       
       f is the function
       Df is its derivative
       method you can choose:
            - LLF: Local Lax Friedrich
            
       returns the step for u_m / u_0 '''
    
    if method == 'LLF':
        u = u_m - lam * ( F_LLF(u_m, u_r, f, Df) - F_LLF(u_l, u_m, f, Df))
        
        #print(Df(0)*lam)
        
    return u