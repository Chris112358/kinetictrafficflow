# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:25:32 2022

@author: chris
"""

import method_class as mc
import class_solution as cs
import numpy as np

import matplotlib.pyplot as plt

import model_velocity as mv
import start_velocity as sv

eps = 10**-9 


def beauty_plot(y_val, *args, x_val=None, fig=None, axis=None, name=None, title=None, 
                y_label=None, x_label=None, **kwargs):
    
    if fig is None and axis is None:
        fig, axis = plt.subplots(1,1, constrained_layout=True)
    elif axis is None:
        axis = fig.axes[0]
    elif fig is None: 
        fig = axis.get_figure()
        
    if x_val is None:
        length = len(y_val)
        x_val = np.linspace(0, 1, length)

    axis.plot(x_val, y_val, *args, **kwargs)
    
    if x_label is None:
        axis.set_xlabel('x')
    else:
        axis.set_xlabel(x_label)
    
    if y_label is None:
        axis.set_ylabel(r'$ \rho $ ')
    else:
        axis.set_ylabel(y_label)
        
    axis.set_ylim((0,1))
    axis.grid('true')
    axis.legend()
    axis.set_title(title)
    
    fig.show()
    
    if name is None:
        return fig
    else:
        fig.savefig(f'plots/{name}.eps', format='eps', bbox_inches = "tight")


def main():
    ''' main function of this script. UNcomment one or multiple functions to see the corresponding plot.
    These plots are used in the "Auswertung" chapter of my thesis. '''
    
    #riemann_2_easy()
    #riemann_2_easy_alt()
    #riemann_3_easy()
    #riemann_4_easy()
    #riemann_10_easy()
    #riemann_100_easy()
    
    #riemann_2_easy_dif()
    #riemann_3_easy_dif()
    
    riemann_2_easy_split()
    #riemann_3_easy_split()
    
    #riemann_2()
    #riemann_2_small()
    #riemann_2_verysmall()
    #riemann_10()
    
    #riemann_seq3_alt_hard()
    #riemann_seq3_hard()
    #riemann_seq2_hard()
    #riemann_seq2_hard2()
    #riemann_seq2_hard3()

    #sinus_2_easy()
    #sinus_2_easy_long()
    #sinus_2_easy_long2()
    sinus_2_easy_split()
    
    
    #sinus_3_easy()
    #sinus_3_easy_split()

    #sinus_5_easy()
    
    #sinus_2()
    
    
    #sinus_seq3_long()
    #sinus_seq3_long_conv()
    
    
    #sinus_2_hard()
    #sinus_2_hard_long()
    
    #sinus_2_hard_easy_1()
    #sinus_2_hard_easy_2()
    #sinus_2_hard_easy_3()
    
    #sinus_2_hard_comp_1()
    #sinus_2_hard_comp_2()
    #sinus_2_hard_comp_3()
    
    #sinus_2_meth2_ending()
    #sinus_2_meth3_ending()
    
    riemann_meth_compare()
    riemann_hard_meth_compare()
    sinus_meth_compare()
    sinus_meth_compare_long()
    sinus_hard_meth_compare()

    #run()
    
    #relax1()
    #relax2()
    #relax3()
    pass



def riemann_2_easy_alt():
    
    t_end = 0.3
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=[0.8,0.1], RHS=[0.05,0.05], mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_easy_riemann_alt', 
                      label='Berechnerter Wert')
    
    
def riemann_2_easy():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    #real analytic solution
    x_p = [0, 0.5-4.5*t_end, 0.5-4.5*t_end, 0.5+t_end, 0.5+t_end, 1]
    y_p = [0.9, 0.9, 4.55/5.5, 4.55/5.5, 0.1, 0.1]
    fig = beauty_plot(y_p, '-.k', x_val=x_p, label='Analytische Lösung', fig=fig, linewidth=1.2 )
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_easy_riemann', 
                      label='Berechnerter Wert')
    
def riemann_2_easy_dif():
    
    t_end = 0.3
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.1, RHS=0.9, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_easy_riemann_dif', 
                      label='Berechnerter Wert')
    
    
def riemann_2_easy_split():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig, axis = plt.subplots(2,1, constrained_layout=True)
    
    fig = beauty_plot(v.v[0,:], '--r', label='Startwert von $f_1$' , axis=axis[0])
    fig = beauty_plot(m.f[0,:], x_val=m.x, label='Berechnete Werte von $f_1$', 
                      axis=fig.axes[0], title=f'Berechnete Werte nach {t_end}', y_label='$f_1$')
    
    fig = beauty_plot(v.v[1,:], '--r', label='Startwert von $f_2$' , axis=fig.axes[1])
    fig = beauty_plot(m.f[1,:], x_val = m.x, 
                      axis=fig.axes[1], 
                      name = 'model_easy_riemann_split', 
                      label='Berechnete Werte von $f_2$',
                      y_label='$f_2$')    

        
def riemann_3_easy():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0) , '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model3_easy_riemann', 
                      label='Berechnerter Wert')
    

def riemann_3_easy_dif():
    
    t_end = 0.3
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=0.1, RHS=0.9, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0) , '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model3_easy_riemann_dif', 
                      label='Berechnerter Wert')
    
    
def riemann_3_easy_split():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig, axis = plt.subplots(3,1, constrained_layout=True)
    
    fig = beauty_plot(v.v[0,:], '--r', label='Startwert von $f_1$' , axis=axis[0])
    fig = beauty_plot(m.f[0,:], x_val=m.x, label='Berechnete Werte von $f_1$', 
                      axis=fig.axes[0], title=f'Berechnete Werte nach {t_end}', y_label='$f_1$')
    
    fig = beauty_plot(v.v[1,:], '--r', label='Startwert von $f_2$' , axis=fig.axes[1])
    fig = beauty_plot(m.f[1,:], x_val=m.x, label='Berechnete Werte von $f_2$', 
                      axis=fig.axes[1], y_label='$f_2$')
    
    fig = beauty_plot(v.v[2,:], '--r', label='Startwert von $f_3$' , axis=fig.axes[2])
    fig = beauty_plot(m.f[2,:], x_val = m.x,  
                      axis=fig.axes[2], 
                      name = 'model3_easy_riemann_split', 
                      label='Berechnete Werte von $f_3$',
                      y_label='$f_3$')
    
    
def riemann_4_easy():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=4)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model4_easy_riemann', 
                      label='Berechnerter Wert')
    
    
def riemann_10_easy():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=10)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model10_easy_riemann', 
                      label='Berechnerter Wert')
    
def riemann_100_easy():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=100)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model100_easy_riemann', 
                      label='Berechnerter Wert')
    
 
def riemann_2():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_riemann', 
                      label='Berechnerter Wert')
    
def riemann_2_small():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_riemann_small', 
                      label='Berechnerter Wert')
    
def riemann_2_verysmall():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.01, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_riemannvsmall', 
                      label='Berechnerter Wert')
    
    
def riemann_10():
    
    t_end = 0.05
    
    v = cs.Values(nx=10000, dim_n=10)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)
    #v.init('riemann', LHS=0.8, RHS=0.4, mid=0.4, amp=0.3, periods=8)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model10_hard_riemann', 
                      label='Berechnerter Wert')
    
    
def riemann_seq3():
    
    t_step = 0.02
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_easy_riemann_seq', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def riemann_seq3_alt():
    
    t_step = 0.02
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=[0.5,0.3,0.1], RHS=[0.06, 0.04, 0.02], mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_easy_riemann_seq_alt', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def riemann_seq3_hard():
    
    t_step = 0.02
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_riemann_seq', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def riemann_seq3_alt_hard():
    
    t_step = 0.02
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('riemann', LHS=[0.5,0.3,0.1], RHS=[0.06, 0.04, 0.02], mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='free')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_riemann_seq_alt', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    


def riemann_seq2_hard():
    
    t_step = 0.3
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=6)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.001, Easy=False, nb=2, method_solve='JX', 
              plot=-1, bound='free', feq=1)
    m.start()
    print('test')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model2_hard_riemann_seq', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def riemann_seq2_hard2():
    
    t_step = 0.3
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=6)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.001, Easy=False, nb=2, method_solve='JX', 
              plot=-1, bound='free', feq=2)
    m.start()
    print('test')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model2_hard_riemann_seq2', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def riemann_seq2_hard3():
    
    t_step = 0.1
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=6)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.001, Easy=False, nb=2, method_solve='JX', 
              plot=-1, bound='free', feq=3)
    m.start()
    print('test')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model2_hard_riemann_seq3', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')


def run():
    
    t_end = 20.0
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=6)
    
    #v.init('sinus', LHS=0.9, RHS=0.1, mid=1.1, amp=0.6, periods=6)
    #v.v[0,:] = v.v[0,:] / 8
    
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=.001, Easy=True, nb=2, method_solve='Relax2', 
              plot=5, bound='periodic', feq=1)
    m.start()

    
def sinus_2_easy():
    
    t_step = 0.06
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_easy_sinus', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    

def sinus_2_easy_split():
    
    t_end = 0.18
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig, axis = plt.subplots(2,1, constrained_layout=True)
    
    fig = beauty_plot(v.v[0,:], '--r', label='Startwert von $f_1$' , axis=axis[0])
    fig = beauty_plot(m.f[0,:], x_val=m.x, label='Berechnete Werte von $f_1$', 
                      axis=fig.axes[0], title=f'Berechnete Werte nach {t_end}', y_label='$f_1$')
    
    fig = beauty_plot(v.v[1,:], '--r', label='Startwert von $f_2$' , axis=fig.axes[1])
    fig = beauty_plot(m.f[1,:], x_val = m.x, 
                      axis=fig.axes[1], 
                      name = 'model_easy_sinus_split', 
                      label='Berechnete Werte von $f_2$',
                      y_label='$f_2$') 
    
 
def sinus_2_easy_long():
    
    t_step = 8
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_easy_sinus_long', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    

def sinus_2_easy_long2():
    
    t_step = 8
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_easy_sinus_long2', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def sinus_3_easy():
    
    t_step = 0.08
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_easy3_sinus', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def sinus_3_easy_split():
    
    t_end = 0.24
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=10.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig, axis = plt.subplots(3,1, constrained_layout=True)
    
    fig = beauty_plot(v.v[0,:], '--r', label='Startwert von $f_1$' , axis=axis[0])
    fig = beauty_plot(m.f[0,:], x_val=m.x, label='Berechnete Werte von $f_1$', 
                      axis=fig.axes[0], title=f'Berechnete Werte nach {t_end}', y_label='$f_1$')
    
    fig = beauty_plot(v.v[1,:], '--r', label='Startwert von $f_2$' , axis=fig.axes[1])
    fig = beauty_plot(m.f[1,:], x_val=m.x, label='Berechnete Werte von $f_2$', 
                      axis=fig.axes[1], y_label='$f_2$')
    
    fig = beauty_plot(v.v[2,:], '--r', label='Startwert von $f_3$' , axis=fig.axes[2])
    fig = beauty_plot(m.f[2,:], x_val = m.x,  
                      axis=fig.axes[2], 
                      name = 'model3_easy_sinus_split', 
                      label='Berechnete Werte von $f_3$',
                      y_label='$f_3$') 

    
    
def sinus_5_easy():
    
    t_step = 0.08
    
    v = cs.Values(nx=10000, dim_n=5)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_easy5_sinus', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')

    
def sinus_2():
    
    t_end = 0.18
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=1.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus', 
                      label='Berechnerter Wert')
    

def sinus_seq3():
    
    t_step = 0.1
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {m.time} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {m.time} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {m.time} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_easy_sinus_seq', 
                      label=f'Wert nach {round(3*t_step, 4)} Zeiteinheiten')
    
    
def sinus_seq3_long():
    
    t_step = 2.1
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_easy_sinus_seq_long', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')   
 
    
def sinus_seq3_long_conv():
    
    t_step = 3 #shows convergence for int values
    
    v = cs.Values(nx=10000, dim_n=3)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=8)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model3_easy_sinus_seq_long_conv', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten') 
    
    
def sinus_2_hard():
    
    t_step = 0.06
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_hard_sinus', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def sinus_2_hard_long():
    
    t_step = 0.1
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1, 
              bound='periodic')
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig, 
                      name = 'model_hard_sinus_long', 
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten')
    
    
def sinus_2_hard_easy_1():

    t_end = 0.1
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label='Hyperbolisches Problem')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_comp1', 
                      label='Vollständiges Problem')
    
def sinus_2_hard_easy_2():

    t_end = 0.20
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label='Hyperbolisches Problem')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_comp2', 
                      label='Vollständiges Problem')
    
def sinus_2_hard_easy_3():

    t_end = 0.3
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label='Hyperbolisches Problem')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_comp3', 
                      label='Vollständiges Problem')
    
    
def sinus_2_hard_comp_1():

    t_end = 0.1
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.2, Easy=False, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    m3 = mc.Method(v)
    m3.set_val(t_end=t_end, T=0.5, Easy=False, nb=2, method_solve='JX', plot=-1)
    m3.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.1$')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.2$')
    fig = beauty_plot(1-m3.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_compT_1', 
                      label=r'$T=0.5$')
    
def sinus_2_hard_comp_2():

    t_end = 0.2
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.2, Easy=False, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    m3 = mc.Method(v)
    m3.set_val(t_end=t_end, T=0.5, Easy=False, nb=2, method_solve='JX', plot=-1)
    m3.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.1$')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.2$')
    fig = beauty_plot(1-m3.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_compT_2', 
                      label=r'$T=0.5$')
    
def sinus_2_hard_comp_3():

    t_end = 0.3
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1)
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.2, Easy=False, nb=2, method_solve='JX', plot=-1)
    m2.start()
    
    m3 = mc.Method(v)
    m3.set_val(t_end=t_end, T=0.5, Easy=False, nb=2, method_solve='JX', plot=-1)
    m3.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.1$')
    fig = beauty_plot(1-m2.u[0,:], x_val=m2.x, fig=fig, label=r'$T=0.2$')
    fig = beauty_plot(1-m3.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_compT_3', 
                      label=r'$T=0.5$')
    

def sinus_2_meth2_ending():
    t_step = 2.5
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.7, amp=0.27, periods=6)

    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    
    m = mc.Method(v)
    m.set_val(t_end=t_step, T=0.001, Easy=False, nb=2, method_solve='JX', plot=-1, 
              bound='periodic', feq=2)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], ':', x_val = m.x, label=f'Wert nach {round(m.time, 4)} Zeiteinheiten', fig=fig)
    
    m.set_val(t_end = m.time + t_step)
    m.start()
    
    fig = beauty_plot(1-m.u[0,:], '-.', x_val = m.x, 
                      title=f'Berechnete Werte nach {round(m.time, 4)} Zeiteinheiten', 
                      fig=fig,  
                      label=f'Wert nach {round(m.time, 4)} Zeiteinheiten',
                      name = 'model_hard_sinus_meth2')

    
def sinus_2_meth3_ending():
    t_end = 20.0
    
    v = cs.Values(nx=10000, dim_n=2)
    v.init('sinus', LHS=[0.29,0.7], RHS=[0.75,0.2], mid=0.4, amp=0.27, periods=6)
    
    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.001, Easy=False, nb=2, method_solve='JX', plot=-1, feq=3)
    m.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val = m.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'model_hard_sinus_meth3', 
                      label='Berechnerter Wert')
    

def riemann_meth_compare():
    
    t_end = 0.08
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1,
              bound='free')
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='Relax2', plot=-1,
               bound='free')
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m.x, fig=fig, label=r'Ordnung 1')
    fig = beauty_plot(1-m2.u[0,:], x_val = m2.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'riemann_solver_2', 
                      label=r'Ordnung 2')
    
def riemann_hard_meth_compare():
    
    t_end = 0.08
    
    v = cs.Values(nx=1000, dim_n=2)
    v.init('riemann', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1,
              bound='free')
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='Relax2', plot=-1,
               bound='free')
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m.x, fig=fig, label=r'Ordnung 1')
    fig = beauty_plot(1-m2.u[0,:], x_val = m2.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'riemann_solver_2_hard', 
                      label=r'Ordnung 2')
    
def sinus_meth_compare():
    
    t_end = 0.55
    
    v = cs.Values(nx=200, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1,
              bound='periodic')
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='Relax2', plot=-1,
               bound='periodic')
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m.x, fig=fig, label=r'Ordnung 1')
    fig = beauty_plot(1-m2.u[0,:], x_val = m2.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'sinus_solver_2', 
                      label=r'Ordnung 2')
    
def sinus_meth_compare_long():
    
    t_end = 4
    
    v = cs.Values(nx=200, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='JX', plot=-1,
              bound='periodic')
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=True, nb=2, method_solve='Relax2', plot=-1,
               bound='periodic')
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m.x, fig=fig, label=r'Ordnung 1')
    fig = beauty_plot(1-m2.u[0,:], x_val = m2.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'sinus_solver_2_long', 
                      label=r'Ordnung 2')
    
def sinus_hard_meth_compare():
    
    t_end = 0.2
    
    v = cs.Values(nx=200, dim_n=2)
    v.init('sinus', LHS=0.9, RHS=0.1, mid=0.4, amp=0.3, periods=2)

    m = mc.Method(v)
    m.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='JX', plot=-1,
              bound='periodic')
    m.start()
    
    m2 = mc.Method(v)
    m2.set_val(t_end=t_end, T=0.1, Easy=False, nb=2, method_solve='Relax2', plot=-1,
               bound='periodic')
    m2.start()
    
    fig = beauty_plot(np.sum(v.v, axis=0), '--r', label='Startwert')
    fig = beauty_plot(1-m.u[0,:], x_val=m.x, fig=fig, label=r'Ordnung 1')
    fig = beauty_plot(1-m2.u[0,:], x_val = m2.x, 
                      title=f'Berechnete Werte nach {t_end}', 
                      fig=fig, 
                      name = 'sinus_solver_2_hard', 
                      label=r'Ordnung 2')
    
    
    
def relax1():
    
    r = np.linspace(0,1, num=1000)
    func = r * (1-r)
    
    fig = beauty_plot(func, x_val = r, 
                      title=r'Funktion $f(\rho) = \rho(1-\rho)$',  
                      name = 'relax_func_1',
                      y_label = r'$f(\rho) $',
                      x_label = r'$ \rho $'
                      )
    
def relax2():
    
    r = np.linspace(0,1, num=1000)
    func = r * (1-r) ** 2
    
    wx = 2/3
    wy = wx * (1-wx) ** 2
    
    fig = beauty_plot(func, x_val = r, 
                      y_label = r'$f(\rho) $',
                      x_label = r'$ \rho $'
                      )
    fig = beauty_plot(wy, 'or', x_val=wx, fig=fig,
                      title=r'Funktion $f(\rho) = \rho(1-\rho)^2$',  name= 'relax_func_2')
    
def relax3():
    
    r = np.linspace(0,1, num=1000)
    func = (-16*(r**4) + 32*(r**3) - 21*(r**2) + 5*r)
    
    w1_x = 0.323
    w2_x = 0.677
    
    w1_y = (-16*(w1_x**4) + 32*(w1_x**3) - 21*(w1_x**2) + 5*w1_x)
    w2_y = (-16*(w2_x**4) + 32*(w2_x**3) - 21*(w2_x**2) + 5*w2_x)
    
    fig = beauty_plot(func, x_val = r,   
                      y_label = r'$f(\rho) $',
                      x_label = r'$ \rho $'
                      )

    fig = beauty_plot(w1_y, 'or', x_val=w1_x , fig=fig)
    fig = beauty_plot(w2_y, 'or', x_val=w2_x , fig=fig, 
                      title=r'Funktion $f(\rho) = -16\rho^4 + 32\rho^3 - 21\rho^2 + 5\rho$',
                      name = 'relax_func_3')
    


if __name__ == '__main__':
    
    main()