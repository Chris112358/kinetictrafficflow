# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:39:55 2021

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def A(f, H=1, roh=.5, v=None):
    '''
    use to calculate the matrix A
    
    Parameters
    ----------
    f : np Array
        main input
    H : float, optional
        main velocity. The default is 1.
    roh : float, optional
        element of [0,1]. The default is .5.
    v : np.array, optional
        if none equidistan v are assumed Need to be the same size as f.
        The default is None.

    Returns
    -------
    matrix of A

    '''
    
    dim = f.shape[0] - 1

    roh = sum(f)
        
    if v is None:
        try:
            v = np.array([i/dim for i in range(dim+1) ])
        except ZeroDivisionError:
            v = [1]
    
    nums = np.array(range(1,dim+1))
    
    def nums(i, j):
        return [v[j] - v[i] for j in range(i+1, j+1) ]
    
    sums = [sum( nums(i, dim) * f[1+i:] ) for i in range(dim+1)]
    
    a_diag = np.array([ v[i] - H/(1-roh) * sums[i] for i in range(dim+1)])
    
    A = np.diag(a_diag)
    
    for idx in range(1,dim+1):
        lis = [H/(1-roh) * (v[i] - v[idx+i]) * f[idx+i] for i in range(dim-idx+1)]
        A += np.diag(lis, -idx)
    
    #print(A)
    return A


def timestep(f, delta_t, delta_x, A, RHS=None, bounds='riemann'):
    '''
    

    Parameters
    ----------
    U0 : np.array may be 2 dimensional (nxm)
        Starting value of the timestep the colums (first argument) 
        represents the values for specific x values
    delta_t : float
        describes the time step.
    delta_x : float
        describes the space step.
    A : function pointer to function A
        describes the matrix of change
    RHS : np.array, optional
        describes the right hand side. none for zeros. The default is None.

    Returns
    -------
    returns the time step of U0
    
    '''
    
    dim, space_dim = f.shape
    
    if RHS is None:
        RHS = np.zeros((dim, 1))
       
    alpha = delta_t / (2 * delta_x)
   
    Out = np.reshape([list(RHS)]*space_dim, (space_dim, dim)).transpose() 
    
    if bounds == 'riemann':
        f = np.append(f, f[:,(-1,0)], axis=1)
    
   
    for idx in range(space_dim):
            
        try:
            Out[:,idx] += 1/2 * (f[:, idx+1] + f[:, idx-1]) \
                        - alpha * (A(f[:, idx]).dot((f[:, idx+1] - f[:, idx-1])))
        except IndexError:
            print('Hi')
            if bounds == 'cycle':
                idx = idx-space_dim
                Out[:,idx] += 1/2 * (f[:, idx+1] + f[:, idx-1]) \
                        - alpha * (A(f[:, idx]).dot((f[:, idx+1] - f[:, idx-1])))
            elif bounds == 'riemann':
                idx = idx-1
    
    
    #print(Out, '\n')
    return Out


def test_riemann(length, dimensions, switch=False):
    mid = length // 2
    
    lis = []
    summe = 0
    
    for _ in range(dimensions):
        right = random.uniform(0.1, 0.9-summe)
        left = random.uniform(0.01, right)
        l = [left] * mid + [right] * (length-mid)
        if switch:
            l = list(reversed(l))
        lis.append(l)
        summe += right
    
    return np.array(lis)


def try_riemann(length, dim, times, switch=False, plot=0):
    
    x = test_riemann(length, dim, switch=switch)
    DT = 0.01
    DX = 0.01
    fig, axs = plt.subplots(2)
    
    axs[0].plot(x[plot])
    for _ in range(times):
        x = timestep(x, DT, DX, A,)
        
    
    axs[0].plot(x[plot])
    axs[1].plot(sum(x))
    
    
    
def conservative(roh, z, DT, DX,):
    
    r_len = roh.shape[0]
    z_len = z.shape[0]
    
    roh_new = np.zeros(r_len)
    z_new = np.zeros(z_len)
    
    if z_len != r_len:
        raise Exception('z and roh dont have equal length')
        
    roh = np.append(roh, roh[(-1,0),], )
    z = np.append(z, z[(-1,0),], )
        
    alpha = DT/DX
    #beta = DT/epsilon
    q = z*(1-roh)
        
    for i in range( r_len):
        roh_new[i] = roh[i] - alpha * (q[i]*(1-roh[i+1]+q[i+1])/(1-roh[i]+q[i]) \
                                       - q[i-1]*(1-roh[i]+q[i])/(1-roh[i-1]+q[i-1]))
            
        z_new[i] = z[i]-alpha*(z[i]-z[i-1])
        #z_new[i] = (z_mid - beta * F(roh_new[i])/(1-roh_new[i])) / (1+beta)
    
    return roh_new, z_new


if __name__ == '__main__':
    
    f2, f1 = test_riemann(100, 2, switch=False)
    r = f1 + f2
    q = f2
    z = q/(1-r)
    DT = 0.01
    DX = 0.01
    r2, z2 = r, z
    for _ in range(10):
        r2, z2 = conservative(r2, z2, DT, DX,)
    q2 = z2*(1-r2)    
    
    fig, axs = plt.subplots(3)
    axs[0].plot(r)
    axs[1].plot(q)
    axs[0].plot(r2)
    axs[1].plot(q2)
    axs[2].plot(z)
    axs[2].plot(z2)
    
    plt.savefig('Example.svg')
    
    #try_riemann(200, 4, 20, switch=False, plot=-1)
    
    #plt.plot(vec)
    #x = timestep(test3, DT,DX,A, )#RHS=np.array([2.0,3.0]))
    #plt.plot(sum(x))
    #x = timestep(x, DT,DX,A, )
    #plt.plot(sum(x))
    #x = timestep(x, DT,DX,A, )
    #plt.plot(sum(x))
    
    #for _ in range(1):
    #    x = timestep(x, DT,DX,A, )
    #plt.plot(sum(x))
    
    
    
    