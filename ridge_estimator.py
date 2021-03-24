# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:11:52 2021

@author: maxim
"""
import numpy as np 

def ridge_regression(X,Y,l_penalty):
    (i,j)=np.shape(X)
    b=np.linalg.inv((np.transpose(X)@X+l_penalty*np.eye(j)))@np.transpose(X)@Y
    return b
