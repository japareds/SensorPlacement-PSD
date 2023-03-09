#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:13:39 2023

@author: jparedes
"""
import pandas as pd
import numpy as np
import os
import LoadDataSet as LDS
import matplotlib as mpl
import matplotlib.pyplot as plt
#%%
def create_dataSet():
    dataset = LDS.dataSet(pollutant,start_date,end_date)
    dataset.load_dataSet(RefStations,file_path)
    dataset.cleanMissingvalues(strategy='interpolate')
    dataset.cleanMissingvalues(strategy='remove')# remove persisting missing values
    return dataset

def Snapshots_matrix(df):
    print('Rearranging data set to form snapshots matrix')
    X = df.T.values
    print(f'Snapshots matrix has dimensions {X.shape}')
    return X
def Covariance(X):
    return

def low_rank_decomposition(X):
    avg = np.mean(X,axis=1)
    X_ = X - avg[:,None]
    print(f'Snapshots matrix created: {X_.shape[1]} measurements of vector space {X_.shape[0]}')
    print(f'Matrix size in memory {X_.__sizeof__()} bytes')
    # NaN values
    print(f'Snapshots matrix has {np.isnan(X).sum()} NaN entries')
    U, S, V = np.linalg.svd(X_,full_matrices=False)
    
    return X_,U,S,V

def svd_threshold(X,Sigma,mu_b):
    """
    Determine threshold for reconstruction
    X = X_true + gX_noisse
    """
    n, m = X.shape
    beta = m/n
    f1 = 2*(beta+1)
    f2 = 8*beta
    f3 = (beta+1)+np.sqrt(beta**2 + 14*beta+1)
    lam = np.sqrt(f1 + f2/f3)
    omega = lam/mu_b
    tau = omega*np.median(Sigma)
    return tau

#%%
def main():
    ds = create_dataSet()
    X = Snapshots_matrix(ds.ds)
    X_,U,S,V = low_rank_decomposition(X)
    mu_b = 1.4149 # from octave-script paper
    tau = svd_threshold(X,S,mu_b)
    
    return X
if __name__ == '__main__':
    # dataset parameters
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    RefStations = ['Badalona','Ciutadella','Eixample','El-Prat','Fabra','Gracia','Manlleu','Palau-Reial','Sant-Adria','Tona','Vall_Hebron','Vic']
    pollutant = 'O3'
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    # data set
    X = main()
    
