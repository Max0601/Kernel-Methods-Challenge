# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:05:54 2021

@author: maxim
"""
import numpy as np

from cvxopt import matrix, solvers, spmatrix

#SVM solving a quadratic program in alpha
lbda_vec = [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.5,1,2,5,10]

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
    return round(res/size,2)

def SVM_train(Ktrain,y,lbda_vec):

  """ Solve the QP with cxvopt solvers and returns alpha as an array"""
  n = Ktrain.shape[0]
  for idx, lbda in enumerate(lbda_vec):  
    C = 1/(2*lbda*n)
    P = matrix(Ktrain, tc="d")
    q = - matrix(y,tc="d")
    G = matrix( np.concatenate( (np.diagflat(y) , -np.diagflat(y) ), axis=0 ),tc="d" )
    h1 = C * np.ones((n,1))
    h2 = np.zeros((n,1)) 
    h = matrix(np.concatenate((h1,h2),axis=0))

    solvers.options['show_progress'] = False
  
    sol = solvers.qp(P,q,G,h) 
    a = np.asarray(sol['x'])

    #alpha is sparse
    a[np.where(np.abs(a) < 1e-4)] = 0
    y_svm = np.dot(Ktrain,a)

    print("Précision pour lambda = " + str(lbda) + " :", accuracy(y_svm,y))


def SVM_test(Ktrain,y,Ktest,lbda):

  """ Solve the QP with cxvopt solvers and returns alpha as an array"""
  n = Ktrain.shape[0]
  C = 1/(2*lbda*n)
  P = matrix(Ktrain, tc="d")
  q = - matrix(y,tc="d")
  G = matrix( np.concatenate( (np.diagflat(y) , -np.diagflat(y) ), axis=0 ),tc="d" )
  h1 = C * np.ones((n,1))
  h2 = np.zeros((n,1)) 
  h = matrix(np.concatenate((h1,h2),axis=0))

  solvers.options['show_progress'] = False
  
  sol = solvers.qp(P,q,G,h) 
  a = np.asarray(sol['x'])

  #alpha is sparse
  a[np.where(np.abs(a) < 1e-4)] = 0
  y_svm = np.dot(a.T,Ktest)

  return y_svm