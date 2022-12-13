# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:49:21 2022

@author: chris
"""

import numpy as np
import random
import matplotlib.pyplot as plt


class myexception(Exception):
    pass


def start_riemann(points, LHS, RHS):
   ''' builds a reiman starting problem for len(RHS)=len(LHS) dimeensions '''
   
   lenr = len(RHS)
   lenl = len(LHS)
   
   if not lenr == lenl:
       raise myexception('dimension of LHS must be equal to dimension of RHS')
   
   mid = points//2
   array = []
   for i in range(points):
       if i < mid:
           array.append(LHS)
       else:
           array.append(RHS)
           
   return np.array(array).transpose()


def start_barrier(points, LHS, MHS, RHS=None):
    ''' builds a starting problem with 3 plateaus for len(LHS)=len(MHS) dimensions.
        if RHS is not given RHS=LHS is assumed'''
    
    lenl = len(LHS)
    lenm = len(MHS)
    if RHS is None:
        RHS = LHS
        lenr = lenl
    else:
        lenr = len(RHS)
    
    if not (lenr == lenl and lenr == lenm):
        raise myexception('dimension of LHS must be equal to dimension of RHS and MHS')
    
    mid = points // 3
    mid2 = (2 * points) // 3
    array = []
    
    for i in range(points):
        if i < mid:
            array.append(LHS)
        elif i < mid2:
            array.append(MHS)
        else:
            array.append(RHS)
            
    return np.array(array).transpose()


def start_barrier_easy(points, LHS, MHS, RHS=None, dim=2):
    
    L = [LHS] * dim
    M = [MHS] * dim
    
    if not RHS is None:
        R = [RHS] * dim
    else:
        R = None
    
    return start_barrier(points, L, M, R)


def start_rieman_rand(points, dim=2, rule='R<L<1'):
    '''builds a rieman problem based on rule:
        following rules are allowed:
        R<L<1 : RHS[i] < LHS[i] both add up to ad most 1
        L<R<1 : LHS[i] < RHS[i] both add up to ad most  '''
        
    reverse = False 
    if rule == 'R<L<1':
        reverse = False
    elif rule == 'L<R<1':
        reverse = True
    else:
        raise myexception('The rule {} is not defined'.format(rule))
    
    rest = 1
    LHS = []
    RHS = []
    for _ in range(dim):
        right = random.uniform(0.1*rest, 0.8*rest)
        left = random.uniform(0.0, right)
        rest -= right
        
        LHS.append(left)
        RHS.append(right)
        
    if reverse:
        return start_riemann(points, LHS, RHS)
    else:
        return start_riemann(points, RHS, LHS)
    raise myexception('somethings is wrong')


def start_rieman_easy(points, LHS, RHS, dim=2,):
    L = [LHS] * dim
    R = [RHS] * dim
    return start_riemann(points, L, R)
    

def boundary(num_b, f, cond='free'):
    
    n, nx = f.shape
    
    if cond == 'free':
    
        left = np.array(list(f[:,0])*num_b).reshape((num_b,n)).T
        right = np.array(list(f[:,-1])*num_b).reshape((num_b,n)).T
        
    if cond == 'periodic':
        
        left = np.array(list(f[:,-num_b:])).reshape((num_b,n)).T
        right = np.array(list(f[:,:num_b])).reshape((num_b,n)).T
        
    if cond == 'periodics':
        
        left = np.array(list(f[:,-num_b-1:-1])).reshape((num_b,n)).T
        right = np.array(list(f[:,1:num_b+1])).reshape((num_b,n)).T
    
    return np.append(left, np.append(f, right, axis=1), axis=1)


def start_sinus(x, mid, amp, periods, dim=2):
    ''' returns sin with periods periodes and midvalue mid and amplitude amp.
    Observe that the first and last value are identical = mid '''
    x_min = x[0]
    x_max = x[-1]

    x_dif = x_max - x_min - x[0]  #+x[1]
    
    sinus = np.sin( 2 * np.pi * periods * (x - x_min)/x_dif ) * amp + mid
    
    return np.array( [sinus/dim] * dim )


def start_amp_sinus(x, mid, amp_min, amp_max, periods, dim=2):
    
    sinus = start_sinus(x, 0, 1, periods, dim=dim)
    
    amp_min = amp_min/1
    amp_max = amp_max/1
    
    x0 = x[0]
    x1 = x[-1]
    
    m = 2 * (amp_max - amp_min)/(x1-x0)
    
    mid_x = len(x) // 2
    
    arrx1 =  m * x[:mid_x] + amp_min
    arrx2 = -m * (x[mid_x:] - x[mid_x]) + amp_max    
    
    arrx = np.append(arrx1, arrx2)
    
    amp_sinus = sinus * arrx + mid/dim

    return amp_sinus


if __name__ == '__main__':
    LHS = [1,2]
    RHS = [2,4]   
    MHS = [5,6]
    print(start_riemann(9,LHS, RHS))
    print(start_barrier(13,LHS, MHS, RHS))
    x = (start_rieman_rand(10,dim=3))
    
    
    plt.plot(x[0])
    plt.plot(x[1])
    plt.plot(x[2])
    
    
    