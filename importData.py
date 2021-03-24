# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:09:11 2021

@author: maxim
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit


Xtrain0=pd.read_csv('dataset/Xtr0.csv')
Ytrain0=pd.read_csv('dataset/Ytr0.csv')
Xtrain1=pd.read_csv('dataset/Xtr1.csv')
Ytrain1=pd.read_csv('dataset/Ytr1.csv')
Xtrain2=pd.read_csv('dataset/Xtr2.csv')
Ytrain2=pd.read_csv('dataset/Ytr2.csv')

Xtest0=pd.read_csv('dataset/Xte0.csv')
Xtest1=pd.read_csv('dataset/Xte1.csv')
Xtest2=pd.read_csv('dataset/Xte2.csv')

Xtr0_mat100=pd.read_csv('dataset/Xtr0_mat100.csv',sep=' ',header=None)
Xtr1_mat100=pd.read_csv('dataset/Xtr1_mat100.csv',sep=' ',header=None)
Xtr2_mat100=pd.read_csv('dataset/Xtr2_mat100.csv',sep=' ',header=None)
Xte0_mat100=pd.read_csv('dataset/Xte0_mat100.csv',sep=' ',header=None)
Xte1_mat100=pd.read_csv('dataset/Xte1_mat100.csv',sep=' ',header=None)
Xte2_mat100=pd.read_csv('dataset/Xte2_mat100.csv',sep=' ',header=None)


Xtrain0=Xtrain0.to_numpy()[:,1].reshape(-1,1)
Xtrain1=Xtrain1.to_numpy()[:,1].reshape(-1,1)
Xtrain2=Xtrain2.to_numpy()[:,1].reshape(-1,1)

Xtr0_mat100=Xtr0_mat100.to_numpy()
Xtr1_mat100=Xtr1_mat100.to_numpy()
Xtr2_mat100=Xtr2_mat100.to_numpy()
Xte0_mat100=Xte0_mat100.to_numpy()
Xte1_mat100=Xte1_mat100.to_numpy()
Xte2_mat100=Xte2_mat100.to_numpy()

Ytrain0=Ytrain0.to_numpy()[:,1].reshape(-1,1)
Ytrain1=Ytrain1.to_numpy()[:,1].reshape(-1,1)
Ytrain2=Ytrain2.to_numpy()[:,1].reshape(-1,1)

Xtest0=Xtest0.to_numpy()[:,1].reshape(-1,1)
Xtest1=Xtest1.to_numpy()[:,1].reshape(-1,1)
Xtest2=Xtest2.to_numpy()[:,1].reshape(-1,1)

Ytrain0 = 2*Ytrain0 -1
Ytrain1 = 2*Ytrain1 -1 
Ytrain2 = 2*Ytrain2 -1 

Ktrain0_3n = np.load('dataset/spec_train0_n3.npy')
Ktest0_3n = np.load('dataset/spec_test0_n3.npy')
Ktrain1_3n = np.load('dataset/spec_train1_n3.npy')
Ktest1_3n = np.load('dataset/spec_test1_n3.npy')
Ktrain2_3n = np.load('dataset/spec_train2_n3.npy')
Ktest2_3n = np.load('dataset/spec_test2_n3.npy')

Ktrain0_7n = np.load('dataset/spec_train0_n7.npy')
Ktest0_7n = np.load('dataset/spec_test0_n7.npy')
Ktrain1_7n = np.load('dataset/spec_train1_n7.npy')
Ktest1_7n = np.load('dataset/spec_test1_n7.npy')
Ktrain2_7n = np.load('dataset/spec_train2_n7.npy')
Ktest2_7n = np.load('dataset/spec_test2_n7.npy')

Ktrain_mis_0_3n = np.load('dataset/mis_train0_n3.npy')
Ktrain_mis_0_5n = np.load('dataset/mis_train0_n5.npy')
Ktrain_mis_0_7n = np.load('dataset/mis_train0_n7.npy')

def csv(y0,y1,y2):
    
    y0 = np.array(y0>0, dtype=int).reshape(1000).tolist()
    y1 = np.array(y1>0, dtype=int).reshape(1000).tolist()
    y2 = np.array(y2>0, dtype=int).reshape(1000).tolist()
    y_pred = y0 + y1 + y2

    with open("Yte.csv", 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(y_pred)):
            f.write(str(i)+','+str(y_pred[i])+'\n')