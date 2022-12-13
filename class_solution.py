# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:00:31 2022

@author: Christian
"""

import numpy as np
import start_velocity as sv

class ValuesError(Exception):
    pass

class Values():
    def __init__(self, x=None, nx=1000, x_range=[0,1], dim_n=2, **qwargs):
        
        try:
            assert isinstance(x, np.ndarray) or x==None, f'x must be of type ndarray but {type(x)} was given'
            assert isinstance(x_range[0], int), f'x_range must contain 2 int values but {type(x_range[0])} was given'
            assert isinstance(x_range[1], int), f'x_range must contain 2 int values but {type(x_range[1])} was given'
            assert isinstance(dim_n, int) and dim_n>0 , f'dim_n must be a positive int value but type {type(dim_n)} and value {dim_n} was given'
            assert isinstance(nx, int) and nx>1, f'nx must be a positive int value but type {type(nx)} and value {nx} was given'
        except AssertionError as E:
            raise ValuesError(E)
        
        if not x is None:
            self.x = x
            self.nx = x.shape[0]
            self.x_start = x[0]
            self.x_end = x[-1]
            
        else:
            self.nx = nx
            self.x_start, self.x_end = x_range[:2]
            self.x = np.linspace(self.x_start, self.x_end, num = self.nx)
            
        self.ndim = dim_n
        self.v = np.zeros((self.ndim, self.nx))
        self.dx = (self.x_end - self.x_start) / (self.nx - 1) 
        
        ### TODO: set function to change values
        
    def init(self, method='riemann', **kwargs):
            if method.lower() == 'riemann':
                
                try: 
                    LHS = kwargs['LHS']
                    RHS = kwargs['RHS']
                except:
                    raise ValuesError('For using Riemann problem you need to give both RHS and LHS')
                
                try:
                    if (( not isinstance(LHS, float) and len(LHS) != self.ndim) or
                       ( not isinstance(RHS, float) and len(RHS) != self.ndim)):
                        raise ValuesError(f'LHS and RHS must be either floats or list of floats with length {self.ndim} but LHS: {LHS} and RHS: {RHS} was given')      
                except:
                    raise ValuesError(f'LHS and RHS must be either floats or list of floats with length {self.ndim} but LHS: {LHS} and RHS: {RHS} was given')
                       
                half = self.nx // 2
                
                div1 = div2 = self.ndim
                
                for idx in range(self.ndim):
                    if isinstance(LHS, float):
                        l = LHS
                    else:
                        l = LHS[idx]
                        div1 = 1
                    if isinstance(RHS, float):
                        r = RHS
                    else:
                        r = RHS[idx]  
                        div2 = 1
                    
                    self.v[idx,:] = [l/div1] * half + [r/div2] * (self.nx - half)
                    
            if method.lower() == 'sinus':
                try:
                    mid = kwargs['mid']
                    amp = kwargs['amp']
                    per = kwargs['periods']
                except:
                    raise ValuesError('To use "sinus" you need to give mid, amp and periods as arguments')
                    
                if not (isinstance(mid, float) or isinstance(amp, float) or 
                        isinstance(per, float)):
                    raise ValuesError(f'mid, amp and periods must be floats but {type(mid), type(amp), type(per)} is given')

                x_dif = self.x_end - self.x_start + self.dx
                
                sinus = np.sin( 2 * np.pi * per * (self.x - self.x_start)/x_dif ) * amp + mid
                
                self.v = np.array( [sinus/self.ndim] * self.ndim )

    
#helper functions:
eps = 10**-9

def boundary_u(u, nb, bound='periodic'):
    
    if bound == 'free':
        for idx in range(nb):
            u[:,idx] = u[:,nb]
            u[:,-1-idx] = u[:,-1-nb]
            
    if bound == 'periodic':
        u[:,:nb] = u[:,-nb*2:-nb]
        u[:,-nb:] = u[:,nb:nb*2]
        
    return u


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


def feq(V, r, eq=1):
    n = len(V)
    nx = len(r)
    func = np.zeros((n,nx))
    
    if eq == 1:
        func[-1,:] = r * (1-r)
    elif eq == 2:
        func[-1,:] = r * (1-r) **2
    elif eq == 3:
        func[-1,:] = (-16*(r**4) + 32*(r**3) - 21*(r**2) + 5*r) 
    else:
        func[-1,:] = r * (1-r)  # Fallback to 1
        print('False eq was given try to use either 1, 2 or 3. 1 Was taken now.')
    
    
    func[0,:] = r-func[-1,:]
    return func


def Fsys(u, V):
    func = Lambda(u, V) * u
    #func = 0.1 * u
    return func
    
    
def fluxJX(u, v, dt, dx, V):
    
    ustar = (u + v)/2 - dt/dx/2 * ( Fsys(v,V) - Fsys(u,V) );  
    flx = 1/2 * (Fsys(ustar,V) + 1/2 * (Fsys(u,V) + Fsys(v,V))) \
            - dx/dt/4 * (v - u)
            
    return flx


def fluxRelax(u, v, dt, dx, V, A):
    ### v = uj+1, u = uj
    
    #A = 1.02 * max(1, abs(lambda_max(u, V))) ** (1/2)
    #A = dx/dt

    flx = 1/2 * (Fsys(u,V) + Fsys(v,V)) - 1/2 * A * (v - u)
    
    return flx


def slope(t, fun='minmod'):
    if fun == 'minmod':
        lim = np.maximum(0, np.minimum(1,t))
    elif fun == 'leer':
        lim = (np.abs(t) + t) / (1 + np.abs(t))
    return lim


def theta(u, v, w, V, A, sign):
    ''' u = uj, v = uj-1, w = uj+1'''
    denom = Fsys(u, V) + sign * A * u - Fsys(v, V) - sign * A * v
    nom =  Fsys(w, V) + sign * A * w - Fsys(u, V) - sign * A * u
    
    #while np.any(denom == 0):
     #   denom += eps
    
    while np.any(nom==0):
        nom += eps
        denom += eps
    
    return slope(denom/nom)


def sigma(u, v, w, dx, A, V, sign):
    s = 1/dx * (Fsys(w, V) + sign * A * w - Fsys(u, V) - sign * A * u)  
    t = theta(u,v,w,V,A,sign)

    return s*t
    

def fluxRelax2(U, nx_inner, dt, dx, V, A):
    
    n_s, n_e = nx_inner
    u = U[:, n_s-1:n_e]
    v = U[:, n_s-2:n_e-1]
    w = U[:, n_s:n_e+1]
    z = U[:, n_s+1:n_e+2]
    
    part1 = fluxRelax(u, w, dt, dx, V, A)
    #A = 1.02 * max(1, abs(lambda_max(u, V))) ** (1/2)
    #A = dx/dt
    sigp = sigma(u, v, w, dx, A, V, 1)
    sigm = sigma(w, u, z, dx, A, V, -1)
    part2 = dx/4 * (sigp-sigm)
    
    return part1 + part2



def step_easy(u, dt, dx, V, nx_inner, method='JX', A=None, bound = 'periodic'):
    ''' same as step, but only one call is necessary / all calculation is done here''' 
    
    '''n_s has to be the index of first inner point, n_e is n_s + number of inner points'''
    n_s, n_e = nx_inner
    if A is None:
        A = 1.01 * max(1, abs(lambda_max(u, V))) #** (1/2)
        
    #A = dx/dt
    
    if method == 'JX':
        
        flux = fluxJX(u[:, n_s-1:n_e], u[:, n_s:n_e+1], dt, dx, V)
        u[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux[:,1:]-flux[:,:-1])
        
    elif method == 'Relax2':
        
        u1 = np.copy(u)
        u2 = np.copy(u)
        
        flux1 = fluxRelax2(u, nx_inner, dt, dx, V, A)
        u1[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux1[:,1:]-flux1[:,:-1])
        
        u1 = boundary_u(u1, 2, bound=bound)
        
        flux2 = fluxRelax2(u1, nx_inner, dt, dx, V, A)
        u2[:,n_s:n_e] = u1[:,n_s:n_e] - dt/dx * (flux2[:,1:]-flux2[:,:-1])
        
        u = (u + u2)/2
        
    elif method == 'Relax':
        
        flux = fluxRelax(u[:, n_s-1:n_e], u[:, n_s:n_e+1], dt, dx, V, A)
        u[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux[:,1:]-flux[:,:-1])
        
    else:
        print(f'Error while calculating no mehod named {method} found. Doing JX instead')
        flux = fluxJX(u[:, n_s-1:n_e], u[:, n_s:n_e+1], dt, dx, V)
        u[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux[:,1:]-flux[:,:-1])
        
    return u    

        
        
if __name__ == '__main__':
    v = Values(nx=10)
    
    v.init(method='sinus', mid=0.39, amp=0.25, periods=6)
    
    print(v.v)
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
        