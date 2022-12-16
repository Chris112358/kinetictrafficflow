# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:48:25 2022

@author: chris
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit

import start_velocity as sv
  

'''
x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)
  
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')
  
for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
'''

n=3

PSUM_MULT = np.ones((n,n)) - np.triu(np.ones((n,n)))
PMULT = np.append(np.ones((n,1)), np.zeros((n,n-1)), axis=1)

NX = 100000

u = np.random.rand(n, NX)
f = np.random.rand(n, NX)
u[0,:] = (1 - sum(f)).reshape(1,NX)

def matrix_mult(u, f):
    Pmult = 1/(1 - (PSUM_MULT @ f))
    u = (PMULT @ u) * Pmult
    return u


def sum_mult(u, f, n, NX):
    psum = np.zeros((1,NX))
    u = u[0,:].reshape(1,NX)
    for k in range(1, n):
        psum += f[k-1, :]
        u = np.append(u, u[0,:] / (1-psum), axis=0)
        
    return u

reps = 10

#t1 = timeit.repeat(lambda: matrix_mult(u, f), number=reps,repeat=5)
#t2 = timeit.repeat(lambda: sum_mult(u, f, n, NX), number=reps,repeat=5)

#print(t1, t2)

#print((matrix_mult(u, f) - sum_mult(u, f, n, NX)).sum(), '\n')
#print(matrix_mult(u, f) == sum_mult(u, f, n, NX), '\n')
#print((matrix_mult(u, f) != sum_mult(u, f, n, NX))[:,:].sum())



x = np.linspace(0, 1 , num=100)

sin = sv.start_sinus(x, 0.4, 0.3, 5)

plt.plot(x, sin)































