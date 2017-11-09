
# coding: utf-8

import pandas as pd
import numpy as np


if True:
    s_one = np.array([1, 1]) 
    s_two = np.array([1, 2])
    s_three = np.array([2, -1])
    s_four = np.array([2, 0])
    s_five = np.array([-1, 2])
    s_six = np.array([-2, 1])
    s_seven = np.array([-1, -1])
    s_eight = np.array([-2, -2])
    
    S = np.array([s_one, s_two, s_three, s_four, s_five, s_six, s_seven, s_eight])
    
    t_one = np.array([-1, -1]) 
    t_two = np.array([-1, -1])
    t_three = np.array([-1, 1])
    t_four = np.array([-1, 1])
    t_five = np.array([1, -1])
    t_six = np.array([1, -1])
    t_seven = np.array([1, 1])
    t_eight = np.array([1, 1])
    
    t = np.array([t_one, t_two, t_three, t_four, t_five, t_six, t_seven, t_eight])

def activation(y, theta):
    theta = theta
    for idx in range(len(y)):
        if y[idx] > theta:
            y[idx] = 1
        elif (-theta <= y[idx]) and (y[idx] <= theta):
            y[idx] = 0
        elif y[idx] < -theta:
            y[idx] = -1
    return y


def neural(al, theta):
    alpha = al
    w = np.zeros((2,2),dtype=int)
    b = np.zeros((2),dtype=int)

    y_in = np.zeros((len(S),len(S[0])), dtype=int)

    Q = 0
    iteration = 0

    while Q < 8:
        for q in range(len(S)):
            y_in[q] = np.dot(S[q].reshape(1,2), w) + b.reshape(1,2)

            activated_op = activation(y_in[q], theta)

            if not activated_op.tolist() == t[q].tolist():
                Q = 0
                w = w + alpha* np.dot((S[q].reshape(2, 1)), t[q].reshape(1, 2))
                b = b + alpha * t[q]
            else:
                Q += 1
        print 'weight', w
        print 'bias', b
        iteration += 1
    return iteration


it = []
alpha = np.linspace(0.001, 1, 100)
for al in alpha:
    it.append(neural(al, 0.5))

import matplotlib.pylab as pl
get_ipython().magic(u'pylab inline')

pl.plot(alpha, it)

alpha



