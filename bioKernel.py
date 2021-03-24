# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:12:41 2021

@author: maxim
"""

#spectrum and mismatch kernels

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from itertools import product

#spectrum and mismatch kernels

def char_combin(n, char_list):
    '''
    Computes all the ngrams (of length n) that we can obtain from a list of char
    '''
    return list(product(char_list,repeat=n))

def mapping_spectrum(n,seq,combin):
    '''
    Creates the mapping seen in the course for the spectrum kernel, using for the pre-indexation the function above
    '''
    ngrams_seq = list(zip(*[seq[i:] for i in range(n)]))
    phi = np.zeros([len(combin),])
    for ngram in ngrams_seq:
        index_ngram = combin.index(ngram)
        phi[index_ngram] = phi[index_ngram]+1
    return phi

def mapping_mismatch(n,seq,combin):
    '''
    Creates the mapping for the one-mismatch kernel 
    '''
    letters = ['A','C','G','T']
    ngrams_seq = list(zip(*[seq[i:] for i in range(n)]))
    phi = np.zeros([len(combin),])
    for ngram in ngrams_seq:
        index_ngram = combin.index(ngram)
        phi[index_ngram] = phi[index_ngram]+1
        copy_ngram = list(ngram)
        for ind,cur_letter in enumerate(copy_ngram):
            for letter in letters:
                if letter!=cur_letter:
                    new_ngram = list(copy_ngram)
                    new_ngram[ind]= letter
                    mismatch_ngram = tuple(new_ngram)
                    index_ngram = combin.index(mismatch_ngram)
                    phi[index_ngram] = phi[index_ngram]+1
    return phi

def linK(phi1,phi2):
    """ 
    Computes the scalar product between phi1 and phi2 (linear kernel in the embedding given above)
    """
    k = np.dot(phi1,phi2)
    return k

#string kernel inspired from https://github.com/timshenkao/StringKernelSVM

def K(n, seq1, seq2, lamb):
    if min(len(seq1), len(seq2))< n:
        return 0
    else:
        sum = 0
        for j in range(1, len(seq2)):
            if seq2[j] == seq1[-1]:
                sum += B(n - 1, seq1[:-1], seq2[:j],lamb)
        result = K(n, seq1[:-1], seq2,lamb) + lamb ** 2 * sum
        return result


def B(n, seq1, seq2,lamb):
    if n == 0:
        return 1
    elif min(len(seq1), len(seq2)) < n:
        return 0
    else:
        sum = 0
        for j in range(1, len(seq2)):
            if seq2[j] == seq1[-1]:
                sum += B(n - 1, seq1[:-1], seq2[:j],lamb) * (lamb ** (len(seq2) - (j + 1) + 2))
        result = lamb * B(n, seq1[:-1], seq2,lamb) + sum
        return result
    
def gram_matrix(n,Xtrain,combin,kernel,Xtest=[],lamb=0.5):
    '''
    This function computes the kernel gram matrix for the spectrum, mismatch and string kernels.
    If Xtest=[], this is the training, if not this is the testing 
    '''
    len_Xtrain = len(Xtrain)
    len_Xtest = len(Xtest)

    if len_Xtest == 0:
      
        mapping_train = {}

        if kernel == "spectrum":
            for i in range(len_Xtrain):
                if i%100==0:
                    print(f'{i} / {len(Xtrain)}')
                mapping_train[i] = mapping_spectrum(n,Xtrain[i,0],combin)
            matrix = np.zeros((len_Xtrain, len_Xtrain), dtype=np.float32)
            for i in range(len_Xtrain):
                for j in range(i, len_Xtrain):
                    matrix[i, j] = linK(mapping_train[i],mapping_train[j])
                    matrix[j, i] = matrix[i, j]
            return matrix
        
        if kernel == "mismatch":
            for i in range(len_Xtrain):
                if i%100==0:
                    print(f'{i} / {len(Xtrain)}')
                mapping_train[i] = mapping_mismatch(n,Xtrain[i,0],combin)
            matrix = np.zeros((len_Xtrain, len_Xtrain), dtype=np.float32)
            for i in range(len_Xtrain):
                for j in range(i, len_Xtrain):
                    matrix[i, j] = linK(mapping_train[i],mapping_train[j])
                    matrix[j, i] = matrix[i, j]
            return matrix

        if kernel == "string":
            matrix = np.zeros((len_Xtrain, len_Xtrain), dtype=np.float32)
            for i in range(len_Xtrain):
                if i%100==0:
                  print(f'{i} / {len(Xtrain)}')
                for j in range(i, len_Xtrain):
                    matrix[i, j] = K(n,Xtrain[i,0],Xtrain[j,0],lamb)
                    matrix[j, i] = matrix[i, j]
            return matrix

    else:

        mapping_train = {}
        mapping_test = {}
        
        if kernel == "spectrum":
            for i in range(len_Xtrain):
                if i%100==0:
                    print(f'{i} / {len(Xtrain) + len(Xtest)}')
                mapping_train[i] = mapping_spectrum(n,Xtrain[i,0],combin)
            for i in range(len_Xtest):
                if i%100==0:
                    print(f'{i + len(Xtrain)} / {len(Xtrain) + len(Xtest)}')
                mapping_test[i] = mapping_spectrum(n,Xtest[i,0],combin)
            matrix = np.zeros((len_Xtrain, len_Xtest), dtype=np.float32)
            for i in range(len_Xtrain):
                for j in range(len_Xtest):
                    matrix[i, j] = linK(mapping_train[i],mapping_test[j])
            return matrix
        
        if kernel == "mismatch":
            for i in range(len_Xtrain):
                if i%100==0:
                    print(f'{i} / {len(Xtrain) + len(Xtest)}')
                mapping_train[i] = mapping_mismatch(n,Xtrain[i,0],combin)
            for i in range(len_Xtest):
                if i%100==0:
                    print(f'{i + len(Xtrain)} / {len(Xtrain) + len(Xtest)}')
                mapping_test[i] = mapping_mismatch(n,Xtest[i,0],combin)
            matrix = np.zeros((len_Xtrain, len_Xtest), dtype=np.float32)
            for i in range(len_Xtrain):
                for j in range(len_Xtest):
                    matrix[i, j] = linK(mapping_train[i],mapping_test[j])
            return matrix


        if kernel == "string":
            matrix = np.zeros((len_Xtrain, len_Xtest), dtype=np.float32)
            for i in range(len_Xtrain):
                if i%10==0:
                  print(f'{i} / {len(Xtrain)}')
                for j in range(len_Xtest):
                    matrix[i, j] = K(n,Xtrain[i,0],Xtest[j,0],lamb)
            return matrix