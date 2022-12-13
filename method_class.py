# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:57:10 2022

@author: chris
"""

import numpy as np
import class_solution as cs
import schemes

from matplotlib import pyplot as plt


class MethodError(Exception):
    pass


class Method():
    def __init__(self, values):
        if not isinstance(values, cs.Values):
            raise MethodError(f'Values should be an instance of Values class, but got {type(values)}' )
        
        self.x_short = values.x
        self.dx = values.dx
        self.nx = values.nx
        self.ndim = values.ndim
        self.v = values.v
        
        self.plot = 100
        self.Easy = False
        self.bound = 'periodic'
        self.t_end = 0.5
        self.T = 0.01
        self.nb = 1
        self.f = None
        self.method_solve = 'JX'
        self.feq = 1
        
        DX = [i * self.dx for i in range(1, self.nb+1)]
        DX.reverse()
        t1 = [self.x_short[0] - d for d in DX]
        t2 =  list(self.x_short)
        DX.reverse()
        t3 = [self.x_short[-1] + d for d in DX]
        self.x = np.array( t1 + t2 + t3 )
        
        self.V = np.linspace(0,1, num=self.ndim)
        self.u = np.zeros((self.ndim, self.nx + 2*self.nb))
        self.time = 0
        self.it = 0
        self.xdim = self.nx + 2*self.nb
        
        self.fig, self.ax = [None, None]
        self.line0, self.line1 = [None, None]
              
    def set_val(self, **kwargs):     
        for key, value in kwargs.items():
            if key.lower() == 'plot':
                if not isinstance(value, int):
                    raise MethodError(f'plot needs to be an int value but {value} was given')
                else:
                    self.plot = value
            if key.lower() == 'feq':
                if not isinstance(value, int):
                    raise MethodError(f'plot needs to be an int value but {value} was given')
                else:
                    self.feq = value
            if key.lower() == 'easy':
                if not isinstance(value, bool):
                    raise MethodError(f'easy needs to be an bool value but {value} was given')
                else:
                    self.Easy = value
                    
            if key.lower() == 'bound':
                if not isinstance(value, str):
                    raise MethodError(f'bound needs to be an string value but {value} was given')
                else:
                    self.bound = value
            if key.lower() == 't_end':
                if not isinstance(value, float) and not isinstance(value, int):
                    raise MethodError(f't_end needs to be a float or int value but {value} with type {value} was given')
                else:
                    self.t_end = value
            if key.lower() == 't':
                try: 
                    v = float(value)
                except:
                    raise MethodError(f't needs to be a float value but {value} was given')
                self.T = v
            if key.lower() == 'method_solve':
                if not isinstance(value, str):
                    raise MethodError(f'method_solve needs to be a string value but {value} was given')
                else:
                    self.method_solve = value
            if key.lower() == 'nb':
                if not isinstance(value, int):
                    raise MethodError(f'nb needs to be an int value but {value} was given')
                else:
                    self.nb = value  
                    DX = [i * self.dx for i in range(1, self.nb+1)]
                    DX.reverse()
                    t1 = [self.x_short[0] - d for d in DX]
                    t2 =  list(self.x_short)
                    DX.reverse()
                    t3 = [self.x_short[-1] + d for d in DX]
                    self.x = np.array( t1 + t2 + t3 )
                    
                    self.u = np.zeros((self.ndim, self.nx + 2*self.nb))
                    self.xdim = self.nx + 2*self.nb
                        
    def boundary_v(self):
        if self.bound == 'free':
        
            left = np.array(list(self.v[:,0]) * self.nb).reshape((self.nb, self.ndim)).T
            right = np.array(list(self.v[:,-1]) * self.nb).reshape((self.nb, self.ndim)).T
            
        if self.bound == 'periodic':
            
            left = np.array(list(self.v[:,-self.nb:])).reshape((self.nb, self.ndim)).T
            right = np.array(list(self.v[:,:self.nb])).reshape((self.nb, self.ndim)).T
        
        self.f = np.append(left, np.append(self.v, right, axis=1), axis=1)
        
    def boundary_u(self):
        
        if self.bound == 'free':
            for idx in range(self.nb):
                self.u[:,idx] = self.u[:,self.nb]
                self.u[:,-1-idx] = self.u[:,-1-self.nb]
                
        if self.bound == 'periodic':
            self.u[:,:self.nb] = self.u[:,-self.nb*2:-self.nb]
            self.u[:,-self.nb:] = self.u[:,self.nb:self.nb*2]
        
    def step(self):
        
        self.boundary_u()
        
        dt = 0.9 * self.dx / cs.lambda_max(self.f,self.V) * 1/ self.A
        dt = min( dt , self.t_end - self.time ) 

        t_step = dt       
        
        
        self.u = cs.step_easy(self.u, dt, self.dx, self.V, (self.nb, self.nb+self.nx),
                              method=self.method_solve, A=self.A, bound=self.bound)
        
        self.f[-1,:] = self.u[0,:] * (1/self.u[-1,:] - 1)
        
        for k in range(self.ndim - 1):
            self.f[k,:] = self.u[0,:] * (1/self.u[k,:] - 1/self.u[k+1,:])

        
        self.r = 1-self.u[0,:]
        self.q = np.dot(self.f.T, self.V)
        
        self.A = 1.1 * max(np.maximum(1, abs(self.q/(1-self.r))))
        
        rel = cs.feq(self.V, self.r, eq=self.feq)
        
        if not self.Easy:
            self.f = ( self.T*self.f + dt*rel ) / (self.T + dt);
        
        self.u[0,:] = 1 - self.f.sum(axis=0)
        
        psum = np.zeros((1,self.xdim))
        
        for k in range(1, self.ndim):
            psum += self.f[k-1, :]
            
            if not self.Easy:
                self.u[k,:] = self.u[0,:] / (1-psum)
        
        self.time += t_step  #dt
        self.it += 1
                      
    def start(self):
        
        #self.dx *= 10 
        
        if self.plot == 0:
            self.plot = -1
        
        if self.time == 0:
            self.boundary_v()
        
        self.u[0,:] = (1 - sum(self.f)).reshape(1,self.xdim)
        self.q = np.dot(self.f.T, self.V)
        
        psum = np.zeros((1, self.xdim))
        
        for k in range(1, self.ndim):
            psum += self.f[k-1, :]
            self.u[k,:] =  self.u[0,:] / (1-psum)
            
        if self.plot > 0:
            self.plotting()
            
        self.A = 1.2 * max(1, abs(cs.lambda_max(self.u, self.V))) #** (1/2)
        #self.A = 1
        
        while self.time < self.t_end:
            self.step()
            
            if self.it % self.plot == 1 or (self.plot > 0 and self.time >= self.t_end):
                self.plotting()
                
    def plotting(self):
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(2, 1, constrained_layout=True)
            self.fig.suptitle(f'time = {self.time}')
            
            self.line0, = self.ax[0].plot(self.x, 1-self.u[0,:] )
            self.ax[0].set_ylabel('rho')
            self.ax[0].set_xlabel('x')
            self.ax[0].grid('true')
            self.ax[0].set_ylim((0,1))
            
            self.line1, = self.ax[1].plot(self.x, self.q)
            self.ax[1].set_ylabel('q')
            self.ax[1].set_xlabel('x')
            self.ax[1].grid('true')
            self.ax[1].set_ylim((0,1))
            
            plt.show()
        
        else:
            self.line0.set_ydata( 1-self.u[0,:] )
            self.line1.set_ydata( self.q )
            self.fig.suptitle(f'time = {self.time}')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        pass
                
    

    
if __name__ == '__main__':
    print('Whoo')
    
    v = cs.Values(nx=10000)
    v.init('sinus', LHS=0.2, RHS=0.8, mid=0.49, amp=0.35, periods=5)
    
    m = Method(v)
    m.set_val(t_end=3.10, T=1.1, Easy=False, nb=2, method_solve='JX')
    m.start()
    #m.boundary()
    #print(m.f.shape, m.x.shape)
    
    #plt.plot(m.x, m.f[1,:])
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    