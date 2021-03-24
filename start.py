# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:08:36 2021

@author: maxim
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from importData import Ytrain0, Ytrain1, Ytrain2, Ktrain0_3n, Ktrain1_3n, Ktrain2_3n, Ktest0_3n, Ktest1_3n, Ktest2_3n, Xtrain0, Xtrain1, Xtrain2, Xtest1, Xtest2, Xtest0
from importData import Ktrain0_7n, Ktrain1_7n, Ktrain2_7n, Ktest0_7n, Ktest1_7n, Ktest2_7n
from bioKernel import *
from kernelRidge import *
from SVM import *
from importData import *

##Kernel Ridge
y0 = kernel_ridge_regression_test(Ktrain0_7n,Ytrain0,Ktest0_7n,lamb = 0.5)
y1 = kernel_ridge_regression_test(Ktrain1_7n,Ytrain1,Ktest1_7n,lamb = 0.5)
y2 = kernel_ridge_regression_test(Ktrain2_7n,Ytrain2,Ktest2_7n,lamb = 0.5)

csv(y0,y1,y2)

##SVM

y0_SVM = SVM_test(np.double(Ktrain0_7n),Ytrain0,Ktest0_7n,lbda = 0.1)
y1_SVM = SVM_test(np.double(Ktrain1_7n),Ytrain1,Ktest1_7n,lbda = 0.1)
y2_SVM = SVM_test(np.double(Ktrain2_7n),Ytrain2,Ktest2_7n,lbda = 0.1)

#csv(y0_SVM,y1_SVM,y2_SVM) #Pour utiliser la méthode SVM