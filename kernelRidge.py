# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:04:45 2021

@author: maxim
"""

import numpy as np

from math import *



sgma = 1

#Test of different values of lambda
lbda_vec = [0,1e-5,1e-4,0.001,0.005,0.1,0.5,1,2,5,10]


def kernel_matrix(fun,X):

    """Compute the kernel matrix of any function fun"""

    n = X.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j]=fun(X[i],X[j])
    return K


def gaussian_kernel(x,y,sigma = sgma):
  g = 1/(2 * (sigma**2) )
  norme2 = ( np.linalg.norm(x-y,2) )**2
  return exp(-g * norme2)

def gaussian_kernel(x,y,sigma = sgma):
  g = 1/(2 * (sigma**2) )
  norme2 = ( np.linalg.norm(x-y,2) )**2
  return exp(-g * norme2)

def accuracy(yreal,ypredicted):
    (size,_)=np.shape(yreal)
    res=0
    for i in range(size):
        if yreal[i][0]>0:
            if ypredicted[i][0]>0:
                res+=1
        else:
            if ypredicted[i][0]<0:
                res+=1
    return round(res/size,4)

def kernel_ridge_regression_train(Ktrain,y,lamb_vec):
    n = Ktrain.shape[0]
    I = np.eye(n,n)
    for idx,lbda in enumerate(lamb_vec):
      K_l = Ktrain+lbda*n*I
      alpha = np.linalg.solve(K_l,y)
      y_KRR = np.dot(Ktrain,alpha)
      print("Précision pour lambda = " + str(lbda) + " :", accuracy(y_KRR,y))

def kernel_ridge_regression_test(Ktrain,Y,Ktest,lamb):
    n = Ktrain.shape[0]
    I = np.eye(n,n)
    K_l = Ktrain + lamb*n*I
    alpha = np.linalg.solve(K_l,Y)
    return np.dot(alpha.T,Ktest)