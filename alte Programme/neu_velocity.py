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
    
    
    