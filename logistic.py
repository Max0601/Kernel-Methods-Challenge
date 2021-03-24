# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:10:52 2021

@author: maxim
"""

# Defining the gradient function of the logistic loss 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit

def L(s,y): 
    return (1/n) * np.sum( np.log( 1 + expit(-s*y) ) )
def E(w,X,y): 
    return L(X.dot(w),y);

def theta(v): 

    return 1 / (1+expit(-v))

def nablaL(s,y): 
    return -1/n * y* theta(-s * y)
def nablaE(w,X,y): return X.transpose()@nablaL(X@w,y)



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



def gradient_descent(X,y,w0, tau, iteration):
    w=w0
    global n
    n,p=np.shape(X)
    res1=[]
    res2=[]
    for i in range(iteration):
        res1.append(w)
        res2.append(E(w,X,y))
        
        w=w-tau*nablaE(w,X,y)
        
        print("training Epoch:"+str(i))
        print('the loss on the training set is ' +str(E(w,X,y)))
        print()
        #print("the loss on the testing set is " +str(const*E(w,X_test,y_test)))
        #y_predicted=X_test@w
        #print("the accuracy on the testing set is "+str(accuracy(y_test,y_predicted)))
    return res1, res2


#a,b=gradient_descent(Xtr0_mat100,Ytrain0,np.random.random((100,1)),0.384,100000)