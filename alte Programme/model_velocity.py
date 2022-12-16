# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:53:57 2022

@author: chris
"""

import numpy as np
from matplotlib import pyplot as plt

import start_velocity as sv
import schemes


def step_pz(p_old, z_old, DT, DX):
    ''' calculate the step for pq model'''
    
    lenp = len(p_old)
    lenz = len(z_old)
    
    if not lenp == lenz:
        raise sv.myexception('p and q have to be same length')
    
    q_old = z_old*(1-p_old)
    
    p = np.append(p_old, p_old[(-1, 0),], )
    q = np.append(q_old, q_old[(-1, 0),], )
    z = np.append(z_old, z_old[(-1, 0),], )
    
    p_new = np.zeros(lenp)
    z_new = np.zeros(lenz)
    
    TX = DT/DX
    
    for idx in range(lenp):
        minu = (1 - p[idx+1] + q[idx+1]) / (1 - p[idx] + q[idx])
        subt = (1 - p[idx] + q[idx]) / (1 - p[idx-1] + q[idx-1])
        
        p_new[idx] = p[idx] - TX * (q[idx] * minu - q[idx-1] * subt)
        z_new[idx] = z[idx] - TX * (z[idx] - z[idx-1])
        
    q_new = z_new*(1-p_new)
    
    return p_new, q_new, z_new


def model_pz(start, x_range=[0,1], t_range=None, time_steps=1, ):
    x_dim = start.shape[1]
    
    try:
        DX = (x_range[1] - x_range[0]) / (x_dim - 1)
    except ZeroDivisionError:
        raise sv.myexception('Length of start should be at least 2 to get some sort of model')
    
    if t_range is None:
        DT = DX
    else:
        DT = t_range / time_steps
    
    p = start[(0,)]
    z = start[(1,)]
    q = z*(1-p)
    
    p_save = p
    z_save = z
    
    for _ in range(time_steps):
        p_step, q_step, z_step = step_pz(p, z, DT, DX)
        p = p_step
        z = z_step
    
    fig, axs = plt.subplots(4)
    axs[0].plot(p_save, label='p')
    axs[1].plot(z_save, label='z')
    axs[2].plot(q, label='q')
    axs[3].plot(p_save-q , label='f_1')
    axs[0].plot(p_step)
    axs[1].plot(z_step)
    axs[2].plot(q_step)
    axs[3].plot(p_step-q_step)
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")
    axs[3].legend(loc="upper right")
    
    
def step_fN(f_start, DT, DX, v):
    x_dim = f_start.shape[1]
    N_dim = f_start.shape[0]
    
    
    p = np.zeros(x_dim)
    for idx in range(N_dim):
        p += f_start[idx,:]
        
    N = np.empty((0, x_dim))
    for idx in range(N_dim):
        num = 1 - p
        summ = np.zeros(x_dim)
        for jdx in range(idx):
            summ += f_start[jdx,:]
            
        den = 1 - summ
        N = np.append(N, (num/den).reshape((1, x_dim)) , axis=0)
    New = np.empty_like(N)
    
    N = np.append(N, np.ones((1, x_dim)), axis=0)
        
    lam = np.empty((0, x_dim))
    for idx in range(N_dim - 1):
        sum_ = 0 
        for jdx in range(idx+1, N_dim):
            sum_ += (v[jdx] - v[idx]) * (1/N[jdx,:] - 1/N[jdx+1,:])
        lam = np.append(lam, (v[idx] - sum_).reshape((1, x_dim)), axis=0)

    lam = np.append(lam, np.ones((1, x_dim)), axis=0)
    #print(lam)
    
    
    #append N to achieve boundaries
    N = np.append(N, N[:,(-1, 0)], axis=1)
    
    for idx in range(N_dim):
        #in each row go through all elements
        for jdx in range(x_dim):
            New[idx, jdx] = schemes.step(N[idx, jdx-1], N[idx, jdx], N[idx, jdx+1], 
                               lambda x: lam[idx, jdx] * x, #f
                               lambda x: lam[idx, jdx], #DF
                               DT/DX)
                               #1/lam[idx, jdx]) #lambda
        
    ### solve d_t N_i + d_x (lambda_i(N) N_i) = 0
    ### lambda_i(N) = v_i - sum_j=i+1^N [ (v_j -v_i)*(1/N_j - 1/N_j+1) ]
    ### N_j+1 = 1

    New = np.append(New, np.ones((1, x_dim)), axis=0)
    f = np.empty((0, x_dim))
    
    for idx in range(N_dim):
        f_new = New[0,:] * (1/New[idx,:] - 1/New[idx+1,:])
        f = np.append(f, f_new.reshape(1, x_dim), axis=0)
        
    ### solve d_t f_i = - 1/T * ( f_i - f_i^e(p) )

    return f
    

def model_fN(start, x_range=[0,1], t_range=None, time_steps=1, ):
    x_dim = start.shape[1]
    N_dim = start.shape[0]
    
    v = np.array([i/(N_dim-1) for i in range(N_dim)])
    
    try:
        DX = (x_range[1] - x_range[0]) / (x_dim - 1)
    except ZeroDivisionError:
        raise sv.myexception('Length of start should be at least 2 to get some sort of model')
    
    if t_range is None:
        DT = DX
    else:
        DT = t_range / time_steps
        
    f_step = start
    
    for _ in range(time_steps):
        f_step = step_fN(f_step, DT, DX, v)
    
    fig, axs = plt.subplots(N_dim)
    for idx in range(N_dim):
        axs[idx].plot(start[idx,:], label='f{}'.format(idx))
        axs[idx].plot(f_step[idx,:])
        axs[idx].legend(loc="upper right")

    #print(start, f_step)

    
def model_f2(start, x_range=[0,1], t_range=None, time_steps=1, ):
    F0 = start[(0,)]
    F1 = start[(1,)]
    
    p = F0 + F1
    N0 = 1 - p
    N1 = N0 / (1 - F0)
    f0 = N0 * ( 1/N0 - 1/N1 )
    f1 = N0 * ( 1/N1 - 1 )
    
    q = f1
    
    ### d_t p + d_x q = 0
    ### d_t N1 + d_x N1 = 0
    pass
    

def correct_fN(start, x, t_end, T, EASY=True, bounds='free', plotting=True):
    
    N, nx = start.shape
    nx_inner = len(x)
    nbc = (nx - nx_inner)//2 
    
    dx = (x[-1] - x[0]) / nx_inner
    
    x_long = np.array(x)
    
    for _ in range(nbc):
        x_long = np.append( x_long[0]-dx, np.append(x_long, x_long[-1]+dx) )
        
    try:
        f = sv.boundary(nbc, start[:, nbc:nbc+nx_inner], cond=bounds)
    except:
        f = sv.boundary(nbc, start, cond=bounds)
    
    u = (1 - sum(f)).reshape(1,nx)
    V = np.linspace(0, 1, num=N)
    
    q = np.dot(f.T, V)
       
    psum = np.zeros((1, nx))
    
    for k in range(1, N):
        psum += f[k-1, :]
        u = np.append(u, u[0,:] / (1-psum), axis=0)
    
    time = 0
    it = 1
    
    if plotting:
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
        
        fig.suptitle(f'time = {time}')
        
        line0, = ax[0].plot(x_long, 1-u[0,:] )
        ax[0].set_ylabel('rho')
        ax[0].set_xlabel('x')
        ax[0].grid('true')
        ax[0].set_ylim((0,1))
        
        line1, = ax[1].plot(x_long, q)
        ax[1].set_ylabel('q')
        ax[1].set_xlabel('x')
        ax[1].grid('true')
        ax[1].set_ylim((0,1))
        
        plt.show()
    
    while time < t_end:
        
        u = sv.boundary(nbc, u[:, nbc:nbc+nx_inner], cond=bounds)
        
        dt = 0.5 * 0.9 * dx / schemes.lambda_max(f,V)
        dt = min( dt , t_end-time )
        
        u = schemes.step_easy(u, dt, dx, V, (nbc, nbc+nx_inner))
        
        f[-1,:] = u[0,:] * (1/u[-1,:] - 1)
        for k in range(N-1):
            f[k,:] = u[0,:] * (1/u[k,:] - 1/u[k+1,:])

        #f = sv.boundary(nbc, f[:, nbc:nbc+nx_inner], cond=bounds)
        
        r = 1-u[0,:]
        q = np.dot(f.T, V)
        rel = schemes.feq(V, r)
        
        #f = ( T*f + dt*rel ) / (T + dt);
        
        u[0,:] = 1 - f.sum(axis=0)
        
        psum = np.zeros((1,nx))
        
        for k in range(1, N):
            psum += f[k-1, :]
            
            if not EASY:
                u[k,:] = u[0,:] / (1-psum)
        
        time += dt
        it += 1
        #timeline = np.append(timeline, [time])
        
        pics = 20
        
        if (it % pics == 0 or time >= t_end) and plotting:
            line0.set_ydata( 1-u[0,:] )
            line1.set_ydata( q )
            fig.suptitle(f'time = {time}')
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            #print(max((1-u[0,:])[:-2*nbc]), min((1-u[0,:])[:-2*nbc]))
           # break
       
    #plt.savefig('Test.png')
    return f[:,nbc:-nbc], u[:,nbc:-nbc], q[nbc:-nbc]
    
    
def error_analysis(x1, v1, x2, v2, norm='L1', steps=1, bounds='periodic', 
                   T = 0.01, EASY=False):
    
    N, nx1 = v1.shape
    N2, nx2 = v2.shape
    assert N2 == N
    
    if nx1 < nx2:
        nx_inner = len(x1)
        dx = (x1[-1] - x1[0]) / nx_inner
        f = v1
        X1 = x1
        X2 = x2
        N1 = nx1
        N2 = nx2
        V1 = v1
        V2 = v2
        
    else:
        nx_inner = len(x2)
        dx = (x2[-1] - x2[0]) / nx_inner
        f = v2
        X1 = x2
        X2 = x1
        N1 = nx2
        N2 = nx1
        V1 = v2
        V2 = v1
        
    V = np.linspace(0, 1, num=N)
    dt = 0.5 * 0.9 * dx / schemes.lambda_max(f,V)
    
    f1, u1, q1 = correct_fN(V1, X1, dt*steps, T, EASY=EASY, bounds=bounds, plotting=False)
    f2, u2, q2 = correct_fN(V2, X2, dt*steps, T, EASY=EASY, bounds=bounds, plotting=False)
    
    M, m = f1.shape
    nbc = ( N1 - m )
    
    save = True
    if save:
        
        s1 = np.append(f1, u1, axis=0)
        s2 = np.append(f2, u2, axis=0)
        
        np.save(f'data/Array_{N1}_{bounds}_{steps}steps', s1)
        np.save(f'data/Array_{N2}_{bounds}_{steps}steps', s2)
    
    if norm == 'L1':
        index = np.linspace(0, N2-1-nbc, N1-nbc).astype(int)
        f2 = np.take(f2, indices=index, axis=1)
        
        diffs = f1 - f2
        
        l1 = max(np.linalg.norm(diffs, ord=1, axis=0))
        l2 = max(np.linalg.norm(diffs, ord=2, axis=0))
        
        print(f'L1 norm is {l1}')
        
        print(f'L2 norm is {l2}')
            
        
def error_plot(*args, x_int=[0,1], x_min=100, x_max=100000, nums=10, steps=1, dims=2,
               start=sv.start_sinus, bounds='periodic', T=0.01, EASY=False, 
                **kwargs):
    
    x = np.linspace(x_int[0],x_int[1],num=x_min+1)[:-1]
    f = start(x, *args, **kwargs)
    V = np.linspace(0, 1, num=dims)
    dt = 0.5 * 0.9 * ((x_int[1]-x_int[0])/x_min) / schemes.lambda_max(f,V)
    
    F_array = np.empty((0,2))
    U_array = np.empty((0,2))
    
    NUMS = np.linspace(x_max, x_min, nums).astype(int)
    for idx, num in enumerate(NUMS):
        
        X = np.linspace(x_int[0], x_int[1], num=num+1)[:-1]
        f = start(X, *args, **kwargs)
        v = sv.boundary(1, f, cond=bounds)
        
        if idx == 0:
            _F, _U, _Q = correct_fN(v, X, dt*steps, T, EASY=EASY, bounds=bounds, 
                                    plotting=False)
            print('Finished first calculations')
            
            s = np.append(_F, _U, axis=0)
            np.save(f'data/Array_{num}_{bounds}_{steps}steps_{dims}dim', s)
            
        else:
            F, U, Q = correct_fN(v, X, dt*steps, T, EASY=EASY, bounds=bounds, 
                                 plotting=False)
            s = np.append(F, U, axis=0)
            np.save(f'data/Array_{num}_{bounds}_{steps}steps_{dims}dim', s)
            
            print(f'Finished {idx} numbers of iterations')
            
            M, m = F.shape
            nbc = 0
            
            index = np.linspace(0, x_max-1-nbc, m-nbc).astype(int)
            _f = np.take(_F, indices=index, axis=1)
            _u = np.take(_U, indices=index, axis=1)
            
            diff_F = _f - F
            diff_U = _u - U
            
            F1 = max(np.linalg.norm(diff_F, ord=1, axis=0))
            F2 = max(np.linalg.norm(diff_F, ord=2, axis=0))
            
            U1 = max(np.linalg.norm(diff_U, ord=1, axis=0))
            U2 = max(np.linalg.norm(diff_U, ord=2, axis=0))
            
            F_array = np.append(F_array, [[F1, F2]], axis=0)
            U_array = np.append(U_array, [[U1, U2]], axis=0)
  
            
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f'Norms of errors')
    
    line0, = ax[0].plot(NUMS[1:], F_array[:,0], 'o-', label='L1' )
    line0, = ax[0].plot(NUMS[1:], F_array[:,1], 'o-', label='L2' )
    ax[0].set_xlabel('num of x')
    ax[0].set_ylabel('error in F')
    ax[0].grid('true')
    ax[0].legend()
    ax[0].set_yscale('log')

    
    line1, = ax[1].plot(NUMS[1:], U_array[:,0], 'o-', label='L1' )
    line1, = ax[1].plot(NUMS[1:], U_array[:,1], 'o-', label='L2' )
    ax[1].set_xlabel('num of x')
    ax[1].set_ylabel('error in U')
    ax[1].grid('true')
    ax[1].legend()
    ax[1].set_yscale('log')
    
    plt.show()
    
    plt.savefig(f'data/Plot_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.eps')
    plt.savefig(f'data/Plot_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.png')
        
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f'Norms of errors')
    
    line0, = ax[0].loglog(NUMS[1:], F_array[:,0], 'o-', label='L1' )
    line0, = ax[0].loglog(NUMS[1:], F_array[:,1], 'o-', label='L2' )
    ax[0].set_xlabel('num of x')
    ax[0].set_ylabel('error in F')
    ax[0].grid('true')
    ax[0].legend()
    ax[0].set_yscale('log')

    
    line1, = ax[1].loglog(NUMS[1:], U_array[:,0], 'o-', label='L1' )
    line1, = ax[1].loglog(NUMS[1:], U_array[:,1], 'o-', label='L2' )
    ax[1].set_xlabel('num of x')
    ax[1].set_ylabel('error in U')
    ax[1].grid('true')
    ax[1].legend()
    ax[1].set_yscale('log')
    
    plt.show()
    
    plt.savefig(f'data/Loglog_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.eps')
    plt.savefig(f'data/Loglog_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.png')

if __name__ == '__main__':
    
    '''
    steps = 5
    
    fs = sv.start_rieman_rand(5, )

    p = fs[(0,)] + fs[(1,)]
    q = fs[(1,)]
    z = q/(1-p)
    
    #start2 = sv.start_riemann(100, [0.59, 0.2], [0,0.1])
    start_3 = sv.start_barrier(100, [0.1, 0.2], [0.5, 0.3])
    start2 = sv.start_riemann(100, [0.5, 0.2], [0.2,0.4])
    start2 = sv.start_rieman_easy(1000, 0.3, 0.475, dim=2)
    
    p2 = start2[0,:] + start2[1,:]
    q2 = start2[1,:]
    z2 = q2/(1-p2)
    
    
    #start = np.append(p2.reshape(1, len(p2)), z2.reshape(1, len(p2)), axis=0)
    
    #model_pz(start, time_steps=20)
    
    #model_fN(start2, t_range=.01, time_steps=200)
    
    nx = 10000
    dim = 2
    
    s = sv.start_rieman_easy(nx+2, 0.7, 0.45, dim=dim)/dim
    s = sv.start_riemann(nx+2, [0.8, 0.9], [0.2, 0.2])/dim
    x = np.linspace(0,1,num=nx)
    
    x_more = np.append([ 2*x[0]-x[1] ], np.append(x, [ 2*x[-1]-x[-2] ]))
    y = x
    
    x_sin = np.linspace(0,1,num=nx+1)[:-1]
    x = x_sin
    
    x_more = np.append([ 2*x[0]-x[1] ], np.append(x, [ 2*x[-1]-x[-2] ]))
    
    sinus = (np.sin(2*np.pi*3*x_more)/5 + 0.7).reshape((1,nx+2))
    sinus_array = np.zeros((0,nx+2))
    for _ in range(dim):
        sinus_array = np.append(sinus_array, sinus/dim, axis=0)
        
    sin = sinus[:,1:-1]
    long = sv.boundary(1, sin, cond='periodic')
    
    #plt.plot(x_more,long.T)
    '''
    
    nx = 1000
    y = np.linspace(0,1,num=nx)
    
    start2 = sv.start_riemann(nx, [0.3, 0.3, 0.3], [0.03,0.03, 0.03])
    
    new_x = np.linspace(0,1,num=nx)
    new_sinus = sv.start_sinus(new_x, 0.4, 0.35, 7)
    
    #new_sinus = sv.start_amp_sinus(new_x, 0.6, 0.01, 0.35, 7)
    
    new = sv.boundary(1, new_sinus, cond='periodic')
    #new = sv.boundary(1, new, cond='periodic')
    
    #new = sv.boundary(1, start2, cond='periodic')
    
    correct_fN(new, y, 5.05, T=0.01, bounds='periodic', EASY=False)
    
    '''
    nx1 = 10000
    nx2 = 10000000
    
    x1 = np.linspace(0,1,num=nx1+1)[:-1]
    x2 = np.linspace(0,1,num=nx2+1)[:-1]
    
    s1 = sv.start_sinus(x1, 0.49, 0.05, 6)
    s2 = sv.start_sinus(x2, 0.49, 0.05, 6)
    
    v1 = sv.boundary(1, s1, cond='periodic')
    v2 = sv.boundary(1, s2, cond='periodic')
    
    #error_analysis(x1, v1, x2, v2)
    
    
    '''
    #error_plot(0.49, 0.05, 6, x_max=10000, nums=20)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    