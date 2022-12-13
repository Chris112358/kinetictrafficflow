# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:18:19 2022

@author: chris
"""

import method_class as mc
import class_solution as cs
import numpy as np

import matplotlib.pyplot as plt

import model_velocity as mv
import start_velocity as sv

eps = 10**-9 

def convergence(x_min, x_max, nums, **kwargs):
    
    space = np.geomspace(x_max, x_min, num=nums).astype(int)
    #space = np.linspace(x_max, x_min, num=nums).astype(int)
    
    kwargs['x'] = None
    kwargs['plot'] = -1
    kwargs['nx'] = int(space[-1])
    
    print(kwargs)
    
    v = cs.Values(**kwargs)
    v.init(**kwargs)
    
    V = np.linspace(0, 1, num=v.ndim)
    t_end = 0.5 * 0.9 * (v.dx / cs.lambda_max(v.v ,V))
    #t_end = 0.01
    
    F_array = np.empty((0,4))
    U_array = np.empty((0,2))
    
    kwargs['t_end'] = t_end
    
    for idx, num in enumerate(space):
        
        kwargs['nx'] = int(num)
        v = cs.Values(**kwargs)
        v.init(**kwargs)    
        
        m = mc.Method(v)
        m.set_val(**kwargs)
        m.start()
        print(f'Finished {idx+1} numbers of iterations')

        if idx == 0:
            
            _F = m.f
            _U = m.u
            _Q = m.q
            
            s = np.append(_F, _U, axis=0)
            np.save(f'data/Array_{v.nx}_{m.bound}_{m.t_end}__{v.ndim}dim', s)
            
        else:
            F = m.f
            U = m.u
            Q = m.q
            
            s = np.append(_F, _U, axis=0)
            np.save(f'data/Array_{v.nx}_{m.bound}_{m.t_end}__{v.ndim}dim', s)
            
            M, m = F.shape
            nbc = 0
            
            index = np.linspace(0, x_max-1-nbc, m-nbc).astype(int)
            _f = np.take(_F, indices=index, axis=1)
            _u = np.take(_U, indices=index, axis=1)
            
            diff_F = _f - F
            diff_U = _u - U
            
            #print(np.amax(diff_F), np.amax(diff_U))
            
            F1 = sum(np.linalg.norm(diff_F, ord=1, axis=1)) /m
            F2 = sum(np.linalg.norm(diff_F, ord=2, axis=1)) /m
            F3 = sum(np.linalg.norm(diff_F, ord=np.inf, axis=1)) /m 
            F4 = sum( abs(sum(diff_F)) ) /m
            
            print(diff_F.shape)
            
            U1 = max(np.linalg.norm(diff_U, ord=1, axis=0))
            U2 = max(np.linalg.norm(diff_U, ord=2, axis=0))
            
            F_array = np.append(F_array, [[F1, F2, F3, F4]], axis=0)
            U_array = np.append(U_array, [[U1, U2]], axis=0)
    
    '''fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f'Norms of errors')
    
    line0, = ax[0].plot(space[1:], F_array[:,0], 'o-' , label='L1' )
    line0, = ax[0].plot(space[1:], F_array[:,1], 'o-' , label='L2' )
    line0, = ax[0].plot(space[1:], F_array[:,2], 'o-' , label='L3' )
    line0, = ax[0].plot(space[1:], F_array[:,3], 'o-' , label='L4' )
    ax[0].set_xlabel('num of x')
    ax[0].set_ylabel('error in F')
    ax[0].grid('true')
    ax[0].legend()
    ax[0].set_yscale('log')

    
    line1, = ax[1].plot(space[1:], U_array[:,0], 'o-' , label='L1')
    line1, = ax[1].plot(space[1:], U_array[:,1], 'o-' , label='L2' )
    ax[1].set_xlabel('num of x')
    ax[1].set_ylabel('error in U')
    ax[1].grid('true')
    ax[1].legend()
    ax[1].set_yscale('log')
    
    plt.show()  '''
    
    #plt.savefig(f'data/Plot_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.eps')
    #plt.savefig(f'data/Plot_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.png')
        
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(f'Norms of errors')
    
    line0, = ax[0].loglog(space[1:], F_array[:,0], 'o-' , label='L1' )
    line0, = ax[0].loglog(space[1:], F_array[:,1], 'o-' , label='L2' )
    line0, = ax[0].loglog(space[1:], F_array[:,2], 'o-' , label='L3' )
    line0, = ax[0].loglog(space[1:], F_array[:,3], 'o-' , label='L4' )
    ax[0].set_xlabel('num of x')
    ax[0].set_ylabel('error in F')
    ax[0].grid('true')
    ax[0].legend()
    ax[0].set_yscale('log')

    
    line1, = ax[1].loglog(space[1:], U_array[:,0], 'o-' , label='L1' )
    line1, = ax[1].loglog(space[1:], U_array[:,1], 'o-' , label='L2' )
    ax[1].set_xlabel('num of x')
    ax[1].set_ylabel('error in U')
    ax[1].grid('true')
    ax[1].legend()
    ax[1].set_yscale('log')
    
    plt.show()
    
    #plt.savefig(f'data/Loglog_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.eps')
    #plt.savefig(f'data/Loglog_{x_min}-{x_max}_{nums}_{steps}_{dims}_{bounds}_concave.png')
    #plt.savefig(f'data/loglog_{kwargs}'.replace(' ','').replace('}','').replace('{',
     #           '').replace('.','').replace("'","") + '.png')

a = 0.1
A = 1*abs(a)
#A = 1.2 * max(1, abs(lambda_max(u, V)))

def Fsys(u, V):
    if u.shape[0] == 2:
        m = np.array([[a, 0], [0, -a]])
        return np.matmul(m,u) 
    else:
        return a*u


def fluxRelax(u, v, dt, dx, V):
    ### v = uj+1, u = uj
    
    #A = 1.01 * abs(a) 
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
    
    if np.any(denom == 0) and np.any(nom == 0):
        return slope((denom+eps)/(nom+eps))
    
    return slope(denom/nom)


def sigma(u, v, w, dx, A, V, sign):
    s = 1/dx * (Fsys(w, V) + sign * A * w - Fsys(u, V) - sign * A * u)  
    t = theta(u,v,w,V,A,sign)

    return s*t
    

def fluxRelax2(U, nx_inner, dt, dx, V):
    
    n_s, n_e = nx_inner
    
    u = U[:, n_s-1:n_e+0]
    v = U[:, n_s-2:n_e-1]
    w = U[:, n_s+0:n_e+1]
    z = U[:, n_s+1:n_e+2]
    
    part1 = fluxRelax(u, w, dt, dx, V)
    #A = 1.01 * abs(a)
    sigp = sigma(u, v, w, dx, A, V, 1)
    sigm = sigma(w, u, z, dx, A, V, -1)
    part2 = dx/4 * (sigp-sigm)
    
    return part1 + part2


def fluxJX(u, v, dt, dx, V):
    
    ustar = (u + v)/2 - dt/dx/2 * ( Fsys(v,V) - Fsys(u,V) );  
    flx = 1/2 * (Fsys(ustar,V) + 1/2 * (Fsys(u,V) + Fsys(v,V))) \
            - dx/dt/4 * (v - u)
            
    return flx


def step_easy(u, dt, dx, V, nx_inner, BOUND='periodic'):
    ''' same as step, but only one call is necessary / all calculation is done here''' 

    n_s, n_e = nx_inner
    
    u1 = np.copy(u)
    u2 = np.copy(u)
    
    flux1 = fluxRelax2(u, nx_inner, dt, dx, V)
    u1[:,n_s:n_e] = u[:,n_s:n_e] - dt/dx * (flux1[:,1:]-flux1[:,:-1])
    
    u1 = sv.boundary(2, u1[:,n_s:n_e], cond=BOUND)
    
    flux2 = fluxRelax2(u1, nx_inner, dt, dx, V)
    u2[:,n_s:n_e] = u1[:,n_s:n_e] - dt/dx * (flux2[:,1:]-flux2[:,:-1])
    
    u = (u + u2)/2
        
    return u    


def step_easy2(U, dt, dx, V, nx_inner):
    ''' same as step, but only one call is necessary / all calculation is done here''' 

    n_s, n_e = nx_inner
    u = U[:, n_s-1:n_e+0]
    v = U[:, n_s-2:n_e-1]
    w = U[:, n_s+0:n_e+1]
    
    flux = fluxJX(u, w, dt, dx, V)
    U[:,n_s:n_e] = U[:,n_s:n_e] - dt/dx * (flux[:,1:]-flux[:,:-1])
    
    return U
    
Print = False

def easy_fun(v, t_end, x=[0,1], solve='JX', BOUND='periodic'):
    
    dim, nx = v.shape
    
    #BOUND = 'periodic'
        
    dx = ( x[1] - x[0] ) / (nx - 1)
    x = np.linspace(x[0], x[1], num=nx)

    u = sv.boundary(2, v, cond=BOUND)
    
    t = 0
    
    #print(x.shape, u.shape) 
    if Print:
            
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    
        fig.suptitle(f'time = {t}')
        
        line0, = ax.plot(x, u[0,2:2+nx].T )
        ax.set_ylabel('u')
        ax.set_xlabel('x')
        ax.grid('true')

        plt.show()
        
    while t < t_end:
        dt = min(t_end-t, abs(dx/a)/10 )
        t += dt
        
        if solve== 'JX':
            u = step_easy2(u, dt, dx, 1, [2,nx+2])
        elif solve == 'Relax2':
            u = step_easy(u, dt, dx, 1, [2,nx+2], BOUND=BOUND)
        
        
        u = sv.boundary(2, u[:,2:2+nx], cond=BOUND)
            
        if Print == True:
            
            if dim>1:
                line0.set_ydata( sum(u[:,2:2+nx])/dim )
            else:
                line0.set_ydata( u[0,2:2+nx].T )
            
            fig.suptitle(f'time = {t}')
            fig.canvas.draw()
            fig.canvas.flush_events()
    if not Print == True:
        return u
    
    
def sinus_real(points, c, t,  m=0.4, a=0.3, p=9):
    V = np.linspace(0, 1, num=points) -c*t
    
    S = a * np.sin(2 * np.pi * p * V) + m
    
    return S
    

def test_conv(x_min, x_max, nums=15, method='sinus', solve='JX'):
    #dx = 1/(x_min -1)
   # t_end = abs(dx/a)
    
    t_end = 5
    
    space = np.geomspace(x_max, x_min, num=nums).astype(int)
            
    F_array = np.empty((0,4))
    
    for idx, num in enumerate(space):
        
        if method == 'sinus':
            v = sv.start_sinus(np.linspace(0,1,num=num), 0.4, 0.3, 5, dim=1)
        else:
            v = sv.start_rieman_easy(num, 0.4, 0.8, dim=1)
        
        u = easy_fun(v, t_end, x=[0,1], solve=solve)[:,2:-2]
        
        
        print(f'Finished {idx+1} numbers of iterations')

        if idx == 0:
            
            _F = u.copy()
            
            
        else:
            F = u.copy()
            
            M, m = F.shape
            nbc = 0
            
            index = np.linspace(0, x_max-1-nbc, m-nbc).astype(int)
            _f = np.take(_F, indices=index, axis=1)
            #_f = sinus_real(num, a, t_end)
            
            diff_F = _f - F
            
            #print(diff_F.shape)
            
            #print(np.amax(diff_F), np.amax(diff_U))
            
            F1 = max(np.linalg.norm(diff_F, ord=1, axis=1)) /(m-1)
            F2 = max(np.linalg.norm(diff_F, ord=2, axis=1)) /(m-1)
            F3 = max(np.linalg.norm(diff_F, ord=np.inf, axis=1)) #/m 
            F4 = max( abs(max(diff_F)) )/m
            
            print(F1, F2, F3, F4, '\n')
            
            F_array = np.append(F_array, [[F1, F2, F3, F4]], axis=0)
        
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle(f'Norms of errors')
    
    line0, = ax.loglog(space[1:], F_array[:,0], 'o-' , label='L1' )
    line0, = ax.loglog(space[1:], F_array[:,1], 'o-' , label='L2' )
    line0, = ax.loglog(space[1:], F_array[:,2], '*-' , label='L3' )
    line0, = ax.loglog(space[1:], F_array[:,3], 'o-' , label='L4' )
    ax.set_xlabel('num of x')
    ax.set_ylabel('error in F')
    ax.grid('true')
    ax.legend()
    ax.set_yscale('log')
    
    plt.show()
    return space[1:], F_array



def convriem():
    
    L, R = 0.4, 0.8
    t_end = 4
    x_min, x_max, nums = 50, 10000, 40
    
    space = np.geomspace(x_max, x_min, num=nums).astype(int)
            
    F_array1 = np.empty((0,2))
    F_array2 = np.empty((0,2))
    
    for idx, num in enumerate(space):

        v = sv.start_rieman_easy(num, L, R, dim=1)
        
        mid, amp, per = 0.5, 0.3, 5
        #v = sv.start_sinus(np.linspace(0,1,num=num), mid, amp, per, dim=1)
        
        
        u1 = easy_fun(v, t_end, x=[0,1], solve='JX', BOUND='free')[:,2:-2]
        u2 = easy_fun(v, t_end, x=[0,1], solve='Relax2', BOUND='free')[:,2:-2]
        
        print(f'Finished {idx+1} numbers of iterations')

        if idx == 0:
            _F1 = u1.copy()
            _F2 = u2.copy()
            
        else:
            F1 = u1.copy()
            F2 = u2.copy()
            
            index = np.linspace(0, x_max-1, num).astype(int)
            _f1 = np.take(_F1, indices=index, axis=1)
            _f2 = np.take(_F2, indices=index, axis=1)
            
            diff_F1 = _f1 - F1
            diff_F2 = _f2 - F2
            
            F_1 = max(np.linalg.norm(diff_F1, ord=1, axis=1)) /num
            F_2 = max(np.linalg.norm(diff_F1, ord=2, axis=1)) /num
            F_3 = max(np.linalg.norm(diff_F1, ord=np.inf, axis=1)) #/m 
            F_4 = max( abs(max(diff_F1)) )/num
            
            F_5 = max(np.linalg.norm(diff_F2, ord=1, axis=1)) /num
            F_6 = max(np.linalg.norm(diff_F2, ord=2, axis=1)) /num
            
            F_array1 = np.append(F_array1, [[F_1, F_2]], axis=0)
            F_array2 = np.append(F_array2, [[F_5, F_6]], axis=0)
    
    plt.loglog(space[:-1], F_array1[:,0], label=(f'Ordnung 1, $p = 1$') )
    plt.loglog(space[:-1], F_array1[:,1], label=(f'Ordnung 1, $p = 2$') )
    
    plt.loglog(space[:-1], F_array2[:,0], label=(f'Ordnung 2, $p = 1$') )
    plt.loglog(space[:-1], F_array2[:,1], label=(f'Ordnung 2, $p = 2$') )
    
    plt.xlabel('Anzahl an Diskretisierungspunkten')
    plt.ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_p$')
    plt.grid('true')
    plt.legend()
    plt.title('Konvergenzvergleich mit Riemann Startbedingungen')
    
    plt.savefig('plots/riemannverg.eps', format='eps', bbox_inches = "tight")
    

    
def convsinus():
    
    t_end = .1
    x_min, x_max, nums = 20, 10000, 40
    mid, amp, per = 0.5, 0.3, 5
    
    space = np.geomspace(x_max, x_min, num=nums).astype(int)
            
    F_array1 = np.empty((0,2))
    F_array2 = np.empty((0,2))
    
    for idx, num in enumerate(space):
        
        v = sv.start_sinus(np.linspace(0,1,num=num), mid, amp, per, dim=1)
        u1 = easy_fun(v, t_end, x=[0,1], solve='JX', BOUND='periodic')[:,2:-2]
        u2 = easy_fun(v, t_end, x=[0,1], solve='Relax2', BOUND='periodic')[:,2:-2]
        
        print(f'Finished {idx+1} numbers of iterations')

        if idx == 0:
            _F1 = u1.copy()
            _F2 = u2.copy()
            
        else:
            F1 = u1.copy()
            F2 = u2.copy()
            
            index = np.linspace(0, x_max-1, num).astype(int)
            _f1 = np.take(_F1, indices=index, axis=1)
            _f2 = np.take(_F2, indices=index, axis=1)
            
            diff_F1 = _f1 - F1
            diff_F2 = _f2 - F2
            
            F_1 = max(np.linalg.norm(diff_F1, ord=1, axis=1)) /num
            F_2 = max(np.linalg.norm(diff_F1, ord=2, axis=1)) /num
            F_3 = max(np.linalg.norm(diff_F1, ord=np.inf, axis=1)) #/m 
            F_4 = max( abs(max(diff_F1)) )/num
            
            F_5 = max(np.linalg.norm(diff_F2, ord=1, axis=1)) /num
            F_6 = max(np.linalg.norm(diff_F2, ord=2, axis=1)) /num
            
            F_array1 = np.append(F_array1, [[F_1, F_2]], axis=0)
            F_array2 = np.append(F_array2, [[F_5, F_6]], axis=0)
    
    plt.loglog(space[:-1], F_array1[:,0], '-r', label=(f'Ordnung 1, $p = 1$') )
    plt.loglog(space[:-1], F_array1[:,1], '-b', label=(f'Ordnung 1, $p = 2$') )
    
    plt.loglog(space[:-1], F_array2[:,0], '--g', label=(f'Ordnung 2, $p = 1$') )
    plt.loglog(space[:-1], F_array2[:,1], '--c', label=(f'Ordnung 2, $p = 2$') )
    
    #plot convergence lines
    O15 = 3 / space[:-1]**1.5
    O1 = 2 / space[:-1]**1
    
    plt.loglog(space[:-1], O1, '-k', label=r'$\mathcal{O}(h^{1})$')
    plt.loglog(space[:-1], O15, '--k', label=r'$\mathcal{O}(h^{1.5})$')
    
    plt.xlabel('Anzahl an Diskretisierungspunkten')
    plt.ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_p$')
    plt.grid('true')
    plt.legend()
    plt.title('Konvergenzvergleich mit Sinus Startbedingungen')
    
    plt.savefig(f'plots/sinusverg_short.eps', format='eps', bbox_inches = "tight")
    

def convsinus_long():
    
    t_end = 4
    x_min, x_max, nums = 20, 10000, 40
    mid, amp, per = 0.5, 0.3, 5
    
    space = np.geomspace(x_max, x_min, num=nums).astype(int)
            
    F_array1 = np.empty((0,2))
    F_array2 = np.empty((0,2))
    
    for idx, num in enumerate(space):
        
        v = sv.start_sinus(np.linspace(0,1,num=num), mid, amp, per, dim=1)
        u1 = easy_fun(v, t_end, x=[0,1], solve='JX', BOUND='periodic')[:,2:-2]
        u2 = easy_fun(v, t_end, x=[0,1], solve='Relax2', BOUND='periodic')[:,2:-2]
        
        print(f'Finished {idx+1} numbers of iterations')

        if idx == 0:
            _F1 = u1.copy()
            _F2 = u2.copy()
            
        else:
            F1 = u1.copy()
            F2 = u2.copy()
            
            index = np.linspace(0, x_max-1, num).astype(int)
            _f1 = np.take(_F1, indices=index, axis=1)
            _f2 = np.take(_F2, indices=index, axis=1)
            
            diff_F1 = _f1 - F1
            diff_F2 = _f2 - F2
            
            F_1 = max(np.linalg.norm(diff_F1, ord=1, axis=1)) /num
            F_2 = max(np.linalg.norm(diff_F1, ord=2, axis=1)) /num
            F_3 = max(np.linalg.norm(diff_F1, ord=np.inf, axis=1)) #/m 
            F_4 = max( abs(max(diff_F1)) )/num
            
            F_5 = max(np.linalg.norm(diff_F2, ord=1, axis=1)) /num
            F_6 = max(np.linalg.norm(diff_F2, ord=2, axis=1)) /num
            
            F_array1 = np.append(F_array1, [[F_1, F_2]], axis=0)
            F_array2 = np.append(F_array2, [[F_5, F_6]], axis=0)
    
    plt.loglog(space[:-1], F_array1[:,0], '-r', label=(f'Ordnung 1, $p = 1$') )
    plt.loglog(space[:-1], F_array1[:,1], '-b', label=(f'Ordnung 1, $p = 2$') )
    
    plt.loglog(space[:-1], F_array2[:,0], '--g', label=(f'Ordnung 2, $p = 1$') )
    plt.loglog(space[:-1], F_array2[:,1], '--c', label=(f'Ordnung 2, $p = 2$') )
    
    #plot convergence lines
    O15 = 50 / space[:-1]**1.5
    O2 = 10 / space[:-1]**1.8
    
    plt.loglog(space[:-1], O15, '-k', label=r'$\mathcal{O}(h^{1.5})$')
    plt.loglog(space[:-1], O2, '--k', label=r'$\mathcal{O}(h^{1.8})$')
    
    plt.xlabel('Anzahl an Diskretisierungspunkten')
    plt.ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_p$')
    plt.grid('true')
    plt.legend()
    plt.title('Konvergenzvergleich mit Sinus Startbedingungen')
    
    plt.savefig(f'plots/sinusverg_long.eps', format='eps', bbox_inches = "tight")


def VerglRiem():
    
    a = 0.1
    t_end = 2
    Print = False
    
    num = [20, 50]
    sym = ['-r', '--b', ':g', '-.c']
    
    N = len(num)
    
    V = [sv.start_rieman_easy(n, 0.4, 0.8) for n in num] + \
         [sv.start_rieman_easy(n, 0.4, 0.8) for n in num]
    U = [easy_fun(v, t_end, solve='JX', BOUND='free')[0,2:-2] for v in V[:N]] + \
         [easy_fun(v, t_end, solve='Relax2', BOUND='free')[0,2:-2] for v in V[N:]]
    
    for idx, u in enumerate(U):
        if idx < N:
            plt.plot(np.linspace(0,1,num=num[idx]), u, sym[idx], label=f'{num[idx]} Punkte 1. Ordnung' )
        else:
            plt.plot(np.linspace(0,1,num=num[idx-N]), u, sym[idx], label=f'{num[idx-N]} Punkte 2. Ordnung' )
    
    plt.plot([0, a*t_end + 0.5, a*t_end + 0.5, 1], [0.4,0.4,0.8,0.8], '-k', label='Analytische Lösung')
    
    plt.legend()
    plt.grid('true')
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={t_end})')
    plt.title(f'Lösung unterschiedlicher Lösungen mit \n verschiedenen Diskretisierungen nach {t_end} Zeiteinheiten')

    plt.savefig('plots/riemanncompare.eps', format='eps', bbox_inches = "tight")


def VerglSinus():
    
    a = 0.1
    t_end = 2
    Print = False
    
    mid, amp, per = 0.5, 0.3, 2
    
    num = [20, 50]
    sym = ['-r', '--b', ':g', '-.c']
    
    N = len(num)
    
    V = [sv.start_sinus(np.linspace(0,1,num=n), mid, amp, per, dim=1) for n in num] + \
         [sv.start_sinus(np.linspace(0,1,num=n), mid, amp, per, dim=1) for n in num]
    U = [easy_fun(v, t_end, solve='JX', BOUND='periodics')[0,2:-2] for v in V[:N]] + \
         [easy_fun(v, t_end, solve='Relax2', BOUND='periodics')[0,2:-2] for v in V[N:]]
    
    for idx, u in enumerate(U):
        if idx < N:
            plt.plot(np.linspace(0,1,num=num[idx]), u, sym[idx], label=f'{num[idx]} Punkte 1. Ordnung' )
        else:
            plt.plot(np.linspace(0,1,num=num[idx-N]), u, sym[idx], label=f'{num[idx-N]} Punkte 2. Ordnung' )
    
    num = 10000
    plt.plot(np.linspace(0, 1, num=num),
             sinus_real(num, a, t_end, m=mid, a=amp, p=per),
             '-k', label='Analytische Lösung')
    plt.legend(loc='lower right')
    plt.grid('true')
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={t_end})')
    plt.title(f'Lösung unterschiedlicher Lösungen mit \n verschiedenen Diskretisierungen nach {t_end} Zeiteinheiten')

    plt.savefig('plots/sinuscompare.eps', format='eps', bbox_inches = "tight")


def Riemann1(sol='JX'):
    a = 0.1
    t_end = 2
    Print = False
    
    num = [20, 50, 100, 500]
    sym = ['-r', '--b', ':g', '-.c']
    
    V = [sv.start_rieman_easy(n, 0.4, 0.8) for n in num]
    U = [easy_fun(v, t_end, solve=sol, BOUND='free')[0,2:-2] for v in V]
    
    for idx, u in enumerate(U):
        plt.plot(np.linspace(0,1,num=num[idx]), u, sym[idx], label=f'{num[idx]} Punkte' )
        
    plt.plot([0, a*t_end + 0.5, a*t_end + 0.5, 1], [0.4,0.4,0.8,0.8], '-k', label='Analytische Lösung')
    
    plt.legend()
    plt.grid('true')
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={t_end})')
    plt.title(f'Lösung verschiedener Diskretisierungen nach {t_end} Zeiteinheiten')
    
    if sol == 'JX':
        plt.savefig('plots/riemannsolves1.eps', format='eps', bbox_inches = "tight")
    elif sol == 'Relax2':
        plt.savefig('plots/riemann2solves1.eps', format='eps', bbox_inches = "tight")


def Sinus1(sol='JX'):
    a = 0.1
    t_end = 2
    Print = False
    
    mid, amp, per = 0.5, 0.3, 2
    
    num = [20, 50, 100, 500]
    sym = ['-r', '--b', ':g', '-.c']
    
    V = [sv.start_sinus(np.linspace(0,1,num=n), mid, amp, per, dim=1) for n in num]
    U = [easy_fun(v, t_end, solve=sol, BOUND='periodic')[0,2:-2] for v in V]
    
    for idx, u in enumerate(U):
        plt.plot(np.linspace(0,1,num=num[idx]), u, sym[idx], label=f'{num[idx]} Punkte' )
        
    num = 10000
    plt.plot(np.linspace(0, 1, num=num),
             sinus_real(num, a, t_end, m=mid, a=amp, p=per),
             '-k', label='Analytische Lösung')
    
    plt.legend()
    plt.grid('true')
    plt.xlabel('x')
    plt.ylabel(f'u(x, t={t_end})')
    plt.title(f'Lösung verschiedener Diskretisierungen nach {t_end} Zeiteinheiten')
    if sol == 'JX':
        plt.savefig('plots/sinussolves1.eps', format='eps', bbox_inches = "tight")
    elif sol == 'Relax2':
        plt.savefig('plots/sinus2solves1.eps', format='eps', bbox_inches = "tight")
    

def Riemann2():
    t_end = 2
    a = 0.1
    Print = False
    
    testnum = [500, 1500, 5000] #+ (echte Lösung)
    againnum = [5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    sym = ['--r', '-.g', ':b', '-k']
    
    testv = [sv.start_rieman_easy(n, 0.4, 0.8) for n in testnum]
    againv = [sv.start_rieman_easy(n, 0.4, 0.8) for n in againnum]
    testu = [easy_fun(v, t_end, solve='JX', BOUND='free')[0,2:-2] for v in testv]
    againu = [easy_fun(v, t_end, solve='JX', BOUND='free')[0,2:-2] for v in againv]
    
    testu.append(np.array([0.4]*700+[0.8]*300))
    testnum.append(1000)
    
    F_1 = np.zeros((len(testnum), len(againnum)))
    F_2 = np.zeros((len(testnum), len(againnum)))

    for idx, u1 in enumerate(testu):
        for jdx, u2 in enumerate(againu):
            
            ut = u1.copy()
            ua = u2.copy()
            
            index = np.linspace(0, testnum[idx]-1, againnum[jdx]).astype(int)
            _f = np.take(ut, indices=index, axis=0)
            
            diff_F = _f - ua
            F2 = (np.linalg.norm(diff_F, ord=2, axis=0)) /(againnum[jdx]-1)
            F1 = (np.linalg.norm(diff_F, ord=1, axis=0)) /(againnum[jdx]-1)
            
            F_1[idx, jdx] = F1
            F_2[idx, jdx] = F2
            
    plt.rcParams['text.usetex'] = True
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
    fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
    
    ax = [ax1, ax2]
    
    #ax[1].yaxis.tick_right()
    #ax[1].yaxis.set_label_position("right")
    
    for idx, row in enumerate(F_1):
        if not idx == len(F_1) - 1:
            ax[0].loglog(againnum, row, sym[idx], label=(f'Test gegen {testnum[idx]} Punkte') )
        else:
            ax[0].loglog(againnum, row, sym[idx], label=(f'Test gegen analytische Lösung') )
        ax[0].set_xlabel('Anzahl an Diskretisierungspunkten')
        ax[0].set_ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_1$')
        ax[0].grid('true')
        ax[0].legend()
    
    for idx, row in enumerate(F_2):
        if not idx == len(F_2)-1:
            ax[1].loglog(againnum, row, sym[idx], label=(f'Test gegen {testnum[idx]} Punkte') )
        else:
            ax[1].loglog(againnum, row, sym[idx], label=(f'Test gegen analytische Lösung') )
        ax[1].set_xlabel('Anzahl an Diskretisierungspunkten')
        ax[1].set_ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_2$')
        ax[1].grid('true')
        ax[1].legend()
    
    fig1.savefig('plots/riemannsolves2_1.eps', format='eps', bbox_inches = "tight")
    fig2.savefig('plots/riemannsolves2_2.eps', format='eps', bbox_inches = "tight")
    
            
    
def Sinus2():
    t_end = 2
    a = 0.1
    Print = False
    
    mid, amp, per = 0.5, 0.3, 2
    
    testnum = [500, 1500, 5000] #+ (echte Lösung)
    againnum = [5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    againnum = list(np.geomspace(50, 300, num=20).astype(int))
    sym = ['--r', '-.g', ':b', '-k']
    
    testv = [sv.start_sinus(np.linspace(0,1,num=n), mid, amp, per, dim=1) for n in testnum]
    againv = [sv.start_sinus(np.linspace(0,1,num=n), mid, amp, per, dim=1) for n in againnum]
    
    testu = [easy_fun(v, t_end, solve='JX', BOUND='periodic')[0,2:-2] for v in testv]
    againu = [easy_fun(v, t_end, solve='JX', BOUND='periodic')[0,2:-2] for v in againv]
    
    testu.append( sinus_real(1000, a, t_end, m=mid, a=amp, p=per) )
    testnum.append(1000)
    
    F_1 = np.zeros((len(testnum), len(againnum)))
    F_2 = np.zeros((len(testnum), len(againnum)))

    for idx, u1 in enumerate(testu):
        for jdx, u2 in enumerate(againu):
            
            ut = u1.copy()
            ua = u2.copy()
            
            index = np.linspace(0, testnum[idx]-1, againnum[jdx]).astype(int)
            _f = np.take(ut, indices=index, axis=0)
            
            diff_F = _f - ua
            F2 = (np.linalg.norm(diff_F, ord=2, axis=0)) /(againnum[jdx]-1)
            F1 = (np.linalg.norm(diff_F, ord=1, axis=0)) /(againnum[jdx]-1)
            
            F_1[idx, jdx] = F1
            F_2[idx, jdx] = F2
            
    #plt.rcParams['text.usetex'] = True

    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
    fig2, ax2 = plt.subplots(1,1, constrained_layout=True)
    
    ax = [ax1, ax2]
    
    #ax[1].yaxis.tick_right()
    #ax[1].yaxis.set_label_position("right")
    
    for idx, row in enumerate(F_1):
        if not idx == len(F_1)-1:
            ax[0].loglog(againnum, row, sym[idx], label=f'Test gegen {testnum[idx]} Punkte' )
        else:
            ax[0].loglog(againnum, row, sym[idx], label=f'Test gegen analytische Lösung' )
        ax[0].set_xlabel('Anzahl der Diskretisierungspunkten')
        ax[0].set_ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_1$')
        ax[0].grid('true')
        ax[0].legend()
    
    for idx, row in enumerate(F_2):
        if not idx == len(F_2)-1:
            ax[1].loglog(againnum, row, sym[idx], label=(f'Test gegen {testnum[idx]} Punkte') )
        else:
            ax[1].loglog(againnum, row, sym[idx], label=(f'Test gegen analytische Lösung') )
        ax[1].set_xlabel('Anzahl der Diskretisierungspunkten')
        ax[1].set_ylabel(r'Fehler in $\vert\vert \cdot\vert\vert_2$')
        ax[1].grid('true')
        ax[1].legend()
    
    fig1.savefig('plots/sinussolves2_1.eps', format='eps', bbox_inches = "tight")
    fig2.savefig('plots/sinussolves2_2.eps', format='eps', bbox_inches = "tight")


Print = False

    
if __name__ == '__main__':
    
    
    ''' these functions are used to plot the graphics of the numeric part of the thesis
    
    uncomment a functions to see the plot result. 
    
    Attention as some functions use the current figure, 
    so uncomment only one funtion at a time. '''
    
    
    #convriem()
    convsinus()
    #convsinus_long()
    #VerglRiem()
    #VerglSinus()
    #Riemann1()
    #Sinus1()
    #Sinus2()
    #Riemann2()
    #Riemann1(sol='Relax2')
    #Sinus1(sol='Relax2')
    
    
    '''
    convergence(10, 10000, 30, method='sinus', mid=0.4, amp=0.3, periods=5, 
                Easy=True, nb=2, T=0.1, method_solve='Relax2')
    
    convergence(10, 10000, 30, method='sinus', mid=0.4, amp=0.1, periods=5, 
                Easy=True, nb=2, T=0.1, method_solve='JX')
    '''
    
    ''' the next part is only for tests and to find the ideas for the thesis '''
    
    '''
    
    nums = 100
    LH = 0.4
    RH = 0.8
    end = 2
    
    v = sv.start_sinus(np.linspace(0,1,num=nums), 0.5, 0.3, 5, dim=1)
    #v = sv.start_rieman_easy(nums, LH, RH, dim=1)
    l1 = easy_fun(v, end, solve='JX')
    l2 = easy_fun(v, end, solve='Relax2')
    
    numl = int(nums/2 + end * a * nums)
    l3 = [[LH] * numl + [RH] * (nums - numl) ]
    l3 = v
    
    x = np.linspace(0,1,num=nums)

    
    if False:
        plt.plot(x, l1[0,2:-2], '--r', label='Ordnung 1')
        plt.plot(x, l2[0,2:-2], ':b', label='Ordnung 2')
        plt.plot(x, l3[0], 'g', label='Echte Lösung')
        plt.legend()
        plt.grid('true')
        plt.title(f'Lösung nach {end} Zeitschritten.')
        plt.xlabel('x')
        plt.ylabel(f'u(x, t={end})')
    
    #plt.savefig('plots/sinusvergleich1.eps', format='eps')
    
    if False:
        S1, F1 = test_conv(10, 10000, nums=20, solve='JX', method='rieman')
        S2, F2 = test_conv(10, 10000, nums=20, solve='Relax2', method='rieman')
    
    
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    