#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral properties of sensor placement

Created on Mon Mar 27 11:14:17 2023

@author: jparedes
"""
import os
import pandas as pd
import scipy
import itertools
import numpy as np
from scipy import linalg
import cvxpy as cp
import matplotlib.pyplot as plt
import LoadDataSet as LDS
import LowRankDecomposition as LRD
import SensorPlacementMethods as SPM
import Plots
#%% Load dataset
            
def create_dataSet():
    dataset = LDS.dataSet(pollutant,start_date,end_date, RefStations)
    dataset.load_dataSet(file_path)
    #dataset.cleanMissingvalues(strategy='interpolate')
    # remove persisting missing values
    dataset.cleanMissingvalues(strategy='remove')
    
    return dataset

def TrainTestSplit(df,end_date_train,end_date_test):
    """
    Splits data set into train and testing set.
    Everything up to the date specified will be used for training. The remaining will be testing set
    """
    time_range = pd.date_range(start=df.index[0],end=end_date_train,freq='1H',closed='left')
    df_train = df.loc[df.index.isin(time_range)]
    time_range = pd.date_range(start=end_date_train,end=end_date_test,freq='1H',closed='left')
    df_test = df.loc[df.index.isin(time_range)]
    
    print(f'Training set generated: from {df_train.index[0]} until {df_train.index[-1]}. {df_train.shape[0]} measurements')
    print(f'Testing set generated: from {df_test.index[0]} until {df_test.index[-1]}. {df_test.shape[0]} measurements')
    return df_train, df_test

#%% locations algorithms
def class_locations(M,n):
    """
    Sensor placement loctions for a specific class

    Parameters
    ----------
    M : np.array (n,m)
        Matrix with canonical vectors in R^n to sample from
    n : int
        Number of locations to sample
    Returns
    -------
    arr : list
        list of all possible sensor combinations
    """
    placements = itertools.combinations(M,n)
    arr = []
    for m in placements:
        arr.append(np.array(m))
    return arr

def class_complementary_locations(C1,n):
    """
    Locations of complementary positions of distribution given by C
    """
    Id =  np.identity(n)
    arr = []
    for C1_ in C1:
        C2 = Id.copy()
        for i in np.argwhere(C1_==1):
            C2[i[1],i[1]] = 0
        C2 = C2[C2.sum(axis=1)==1,:]
        arr.append(C2)
    return arr

def empty_locations(In,C1,C2):
    """
    Return locations not ocupied neither by class 1 nor by class 2

    Parameters
    ----------
    In : np array (n,n)
        Identity matrix
    C1 : np array(p1,n)
    C2 : np.array (p2,n)

    Returns
    -------
    C3 : np.array(p3,n)
    """
    
    C3 = In.copy()
    for i in np.argwhere(C1==1):
        C3[i[1],i[1]] = 0
    for i in np.argwhere(C2==1):
        C3[i[1],i[1]] = 0
    C3 = C3[C3.sum(axis=1)==1,:]

    return C3
    
def sensors_distributions(n,p1,p2,p3):
    """
    All possible distributions of sensor locations
    """
    In = np.identity(n)
    C1 = class_locations(In,p1)
    C1c = class_complementary_locations(C1,n)
    C2 = []
    C3 = []
    if p3 != 0:
        for idx in range(len(C1c)):
            c1_ = C1c[idx]
            c2 = class_locations(c1_,p2)
            c3 = []
            for c2_ in c2:
                c3_ = empty_locations(In,C1[idx],c2_)
                c3.append(c3_)
                C2.append(c2)
                C3.append(c3)
    else:
        C2 = C1c.copy()
    return C1,C2,C3

#%% sensor placement algorithms
def D_optimal_limit(Psi,C1,C2,sigma1):
    """
    D-optimal case in the limit sigma2 = 0
    Variance = sigma1 V@V.T; V = ker(Phi2)
    """
    n,r = Psi.shape
    results = []
    for c2 in C2:
        Theta2 = c2@Psi
        Phi2 = Theta2.T@Theta2
        V = scipy.linalg.null_space(Phi2)
        Var_limit = sigma1*V@V.T
        ld = -np.log(np.abs(np.linalg.det(Var_limit)))
        results.append(ld)
        
    return results

def E_optimal_multiclass(Psi,C1,C2,C3,p1,p2,p3,sigma1 = 1e-1,sigma2 =1e-1/10,sigma3=10*1e-1,save_iter=False,results_path='.'):
    """
    E-optimal sensor placement algorithm

    Parameters
    ----------
    Psi : np array(n,r)
        Eigenmodes basis of r vectors in R^n
    C1 : np array
        distribution of class 1
    C2 : np array
        distribution of class 2 (ref stations)
    C3 : np array
        distribution of class 3 (empty)
    sigma1 : float, optional
        Class 1 sensors variance. The default is 1e-1.
    sigma2 : float, optional
        Class 2 sensors variance. The default is 1e-1/10.
    sigma3 : float, optional
        Class 3 sensors variance. The default is 10*1e-1.
    save_iter : bool, optional
        save search results. The default is False.
    results_path : str, optional
        path to save directory. The default is '.'.

    Returns
    -------
    results: np array
        results of the exhaustive search
    """
    n,r = Psi.shape 
    # iterate over possibilities
    results = []
    for c1,c2_,c3_ in zip(C1,C2,C3):
        for c2,c3 in zip(c2_,c3_):
            Theta1 = c1@Psi
            Theta2 = c2@Psi
            Theta3 = c3@Psi
            
            Phi1= Theta1.T@Theta1
            Phi2 = Theta2.T@Theta2
            Phi3 = Theta3.T@Theta3
            M_ = (1/sigma1)*Phi1 + (1/sigma2)*Phi2 + (1/sigma3)*Phi3
            Var = np.linalg.inv(M_)
            lambdas,_ = np.linalg.eigh(Var)
            lmax = lambdas.max()
            results.append(lmax)
    if save_iter:
        fname = f'E-optimal_SensorReplacement_r{r}_p1_{p1}_p2_{p2}_p3_{p3}_sigma1_{sigma1}_sigma2_{sigma2}_sigma3_{sigma3}.csv'
        np.savetxt(f'{results_path}{fname}',results,delimiter=',')
        
    # compute optimal placement
    #  loc = np.argmax(results)
    # C_optimal = C1[loc]
    # locations = np.argwhere(C_optimal==1)[:,1]
        
    return np.array(results)

def D_optimal_multiclass(Psi,C1,C2,C3,p1,p2,p3,sigma1 = 1e-1,sigma2 =1e-1/10,sigma3=10*1e-1,save_iter=False,results_path='.'):
    """
    E-optimal sensor placement algorithm

    Parameters
    ----------
    Psi : np array(n,r)
        Eigenmodes basis of r vectors in R^n
    C1 : np array
        distribution of class 1
    C2 : np array
        distribution of class 2 (ref stations)
    C3 : np array
        distribution of class 3 (empty)
    sigma1 : float, optional
        Class 1 sensors variance. The default is 1e-1.
    sigma2 : float, optional
        Class 2 sensors variance. The default is 1e-1/10.
    sigma3 : float, optional
        Class 3 sensors variance. The default is 10*1e-1.
    save_iter : bool, optional
        save search results. The default is False.
    results_path : str, optional
        path to save directory. The default is '.'.

    Returns
    -------
    results: np array
        results of the exhaustive search
    """
    n,r = Psi.shape 
    # iterate over possibilities
    results = []
    for c1,c2_,c3_ in zip(C1,C2,C3):
        for c2,c3 in zip(c2_,c3_):
            Theta1 = c1@Psi
            Theta2 = c2@Psi
            Theta3 = c3@Psi
            
            Phi1= Theta1.T@Theta1
            Phi2 = Theta2.T@Theta2
            Phi3 = Theta3.T@Theta3
            M_ = (1/sigma1)*Phi1 + (1/sigma2)*Phi2 + (1/sigma3)*Phi3
            ld = np.log(np.linalg.det(M_))
            results.append(ld)
    if save_iter:
        fname = f'D-optimal_SensorReplacement_r{r}_p1_{p1}_p2_{p2}_p3_{p3}_sigma1_{sigma1}_sigma2_{sigma2}_sigma3_{sigma3}.csv'
        np.savetxt(f'{results_path}{fname}',results,delimiter=',')
        
    # compute optimal placement
    #  loc = np.argmax(results)
     # C_optimal = C1[loc]
     # locations = np.argwhere(C_optimal==1)[:,1]
        
    return np.array(results)


def D_optimal_convex(Psi,C1,C2,C3,p1,p2,p3,sigma1 = 1e-1,sigma2 =1e-1/10,sigma3=10*1e-1,save_iter=False,results_path='.'):
    """
    Convex relaxation for D-optimal

    Parameters
    ----------
    Psi : np array(n,r)
        Eigenmodes basis of r vectors in R^n
    C1 : np array
        distribution of class 1
    C2 : np array
        distribution of class 2 (ref stations)
    C3 : np array
        distribution of class 3 (empty)
    sigma1 : float, optional
        Class 1 sensors variance. The default is 1e-1.
    sigma2 : float, optional
        Class 2 sensors variance. The default is 1e-1/10.
    sigma3 : float, optional
        Class 3 sensors variance. The default is 10*1e-1.
    save_iter : bool, optional
        save search results. The default is False.
    results_path : str, optional
        path to save directory. The default is '.'.

    Returns
    -------
    results: np array
        results of the exhaustive search
    """
    n,r = Psi.shape 
    # iterate over possibilities
    # problem data
    beta = cp.Variable((n,classes),pos=True,name='beta')
    F = []
    for eq in range(classes):
        F.append(k[eq]*cp.sum([beta[i,eq]*f[i] for i in range(n)],axis=0))
    Total_sum = cp.sum([F[i] for i in range(len(F))])
    
    objective = cp.Minimize(-1*cp.log_det(Total_sum))
    constraints = [
        beta >= np.zeros((n,classes)),# betas >= 0
        cp.sum(beta,axis=1) <= 1, # sum of probabilities at most 1
        cp.sum(beta,axis=0) == sensors # sum betas_i = num_sensors_class_i
        ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(gp=False)
    
    
    results = []
    for c1,c2_,c3_ in zip(C1,C2,C3):
        for c2,c3 in zip(c2_,c3_):
            Theta1 = c1@Psi
            Theta2 = c2@Psi
            Theta3 = c3@Psi
            
            Phi1= Theta1.T@Theta1
            Phi2 = Theta2.T@Theta2
            Phi3 = Theta3.T@Theta3
            M_ = (1/sigma1)*Phi1 + (1/sigma2)*Phi2 + (1/sigma3)*Phi3
            ld = np.log(np.linalg.det(M_))
            results.append(ld)
    if save_iter:
        fname = f'D-optimal_SensorReplacement_r{r}_p1_{p1}_p2_{p2}_p3_{p3}_sigma1_{sigma1}_sigma2_{sigma2}_sigma3_{sigma3}.csv'
        np.savetxt(f'{results_path}{fname}',results,delimiter=',')
        
    # compute optimal placement
    #  loc = np.argmax(results)
     # C_optimal = C1[loc]
     # locations = np.argwhere(C_optimal==1)[:,1]
        
    return np.array(results)

#%%
def full_space_positioning(U,r,p,p1,p2,sigma1,sigma2):
    """
    Sensor placement covering the whole space with two classes of sensors
    
    U: Eigenmodes basis
    p1: number of LCSs
    p2: number of stations with varying variance
    """
    # sensor parameters
    Psi = U[:,:r]
    # relevant matrices
    C = np.identity(n)
    C1 = C[-p1:,:]
    C2 = C[:p2,:]
    H1 = C1.T@C1
    H2 = C2.T@C2
    Theta1 = C1@Psi
    Theta2 = C2@Psi
    Phi1= Theta1.T@Theta1
    Phi2 = Theta2.T@Theta2
    Var= np.linalg.inv((1/sigma1)*Phi1 + (1/sigma2)*Phi2)
    a = np.linalg.inv((1/sigma1)*Phi1 + (1/sigma2)*Phi2)@(np.block([(1/sigma1)*Theta1.T,(1/sigma2)*Theta2.T]))
    return Psi,C1,H1,Theta1,Phi1,sigma1,C2,H2,Theta2,Phi2,sigma2,Var,a

def spectral_information(Phi1,Phi2,Var):
    lambda1,v1= np.linalg.eigh(Phi1)
    lambda1[lambda1.real<1e-10] = 0.0
    diag1 = np.diag(lambda1)
    lambda2,v2= np.linalg.eigh(Phi2)
    lambda2[lambda2.real<1e-10] = 0.0
    diag2 = v1.T@Phi2@v1
    diag2[diag2<1e-10] = 0.0
    print('Lambda1, Lambda2')
    print(*[[diag1[i,i],diag2[i,i]] for i in range(diag1.shape[0])],sep='\n')
  
    lambdaVar,vVar= np.linalg.eigh(Var)
    

    return lambda1,lambda2,diag1,diag2,lambdaVar,v1,v2,vVar

#%% KKT method
def KKT_variance(Phi1,Phi2,Theta1,Theta2,sigma1):
    """
    Compute variance of a in the exact case of sigma2 = 0 via KKT
    """
    A = Phi1@Phi1+Phi2
    A_inv = np.linalg.inv(A)
    Schur = Theta2@Theta2.T - Theta2@Phi1@A_inv@Phi1@Theta2.T
    Schur_inv = np.linalg.inv(Schur)
    # (K^T@K)^-1
    E = A_inv + A_inv@Phi1@Theta2.T@Schur_inv@Theta2@Phi1@A_inv
    F = -A_inv@Phi1@Theta2.T@Schur_inv
    G = -Schur_inv@Theta2@Phi1@A_inv
    H = Schur_inv.copy()
    # pseudo inverse: eq to np.linalg.inv(K) but the first block has inverse for K^T@K
    Z = np.zeros((Theta2.shape[0],Theta2.shape[0]))
    K = np.block([[Phi1,Theta2.T],[Theta2,Z]])
    
    K_pinv = np.block([[E@Phi1+F@Theta2,E@Theta2.T],[G@Phi1+H@Theta2,G@Theta2.T]])
    # variance of estimator
    M = E@Phi1@Theta1.T + F@Theta2@Theta1.T
    Var_KKT = sigma1*M@M.T

#%% plots

def plot_iterations(results):
    fs = 15
    x =[np.round(i) for i in results]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(results)
    ax.set_title('E-optimal selection',fontsize=fs)
    ax.set_xlabel('Iteration number',fontsize=fs)
    ax.set_ylabel('minimum eigenvalue',fontsize=fs)
    
def plot_lambdamax_sigma2_zero(results_E_r5,results_E_r8,results_E_r10):
    """
    Plot E-optimal (lambda max of min Sigma) results for p2=r and different values of r
    as sigma converges to zero
    """
    fs = 15
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(results_E_r5,color='#1F618D',label='r=5',marker='o')
    ax.plot(results_E_r8,color='#117A65',label='r=8',marker='o')
    ax.plot(results_E_r10,color='#CA6F1E',label='r=10',marker='o')
    
    ax.set_yscale('log')
    
    yrange = np.logspace(-1,-15,15)
    ax.set_yticks([i for i in yrange[::2]])
    exps = np.arange(-1,-16,-2)
    ax.set_yticklabels([str('$10^{%i}$'%i) for i in exps],fontsize=fs-2)
    ax.set_ylabel('$\min\ \lambda_{max} \mathbf{\Sigma_{\hat{a}}}$',fontsize=fs)
    
    xrange = np.arange(0,len(results_E_r8)+1,2)
    ax.set_xticks(xrange)
    ax.set_xticklabels([str('$10^{-%i}$'%(i+1)) for i in xrange],fontsize=fs-2)
    ax.set_xlabel('$\sigma_2^{2}/\sigma_1^{2}$',fontsize=fs)
    
    ax.legend(loc='best',fontsize=fs)
    ax.grid()
    
    return fig

def plot_lambda_Spectra_sigma2_zero(results_2,results_5,results_10,results_100):
    """
    Plot Lambda max of Sigma for different iterations as sigma2 -> 0
    For the results not to be a single line it is necessary that p2>=r
    """
    # Plot simultaneously different results for different values of sigma2 to show that the results
    # converge to zero
    fs = 20
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(141)
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)
    ax.plot(results_2,color='#117A65',alpha=0.7,label='$\sigma_2 = \sigma_1/2$')
    ax1.plot(results_5,color='#D35400',alpha=0.7,label='$\sigma_2 = \sigma_1/5$')
    ax2.plot(results_10,color='#1A5276',label='$\sigma_2 = \sigma_1/10$')
    ax3.plot(results_100,color='#CB4335',label='$\sigma_2 = \sigma_1/100$')
    
    ax.set_yticks(ax3.get_yticks())
    ax1.set_yticks(ax3.get_yticks())
    ax2.set_yticks(ax3.get_yticks())
    ax3.set_yticks(ax3.get_yticks())
    
    ax.set_ylim(0.0,sigma1+0.01)
    ax1.set_ylim(0.0,sigma1+0.01)
    ax2.set_ylim(0.0,sigma1+0.01)
    ax3.set_ylim(0.0,sigma1+0.01)   
    
    ax.set_yticklabels([f'{i:.2f}' for i in ax.get_yticks()],fontsize=fs)
    ax1.set_yticklabels('')
    ax2.set_yticklabels('')
    ax3.set_yticklabels('')
    ax.set_ylabel('$\lambda_{max} \Sigma_{\hat{a}}$\n',fontsize=fs+5)
    
    ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],fontsize=fs)
    ax1.set_xticklabels([int(i+1) for i in ax.get_xticks()],fontsize=fs)
    ax2.set_xticklabels([int(i+1) for i in ax.get_xticks()],fontsize=fs)
    ax3.set_xticklabels([int(i+1) for i in ax.get_xticks()],fontsize=fs)
    ax.set_xlabel('Iteration number',fontsize=fs)
    ax1.set_xlabel('Iteration number',fontsize=fs)
    ax2.set_xlabel('Iteration number',fontsize=fs)
    ax3.set_xlabel('Iteration number',fontsize=fs)
    
    ax.legend(loc='upper right',fontsize=fs)
    ax1.legend(loc='upper right',fontsize=fs)
    ax2.legend(loc='upper right',fontsize=fs)
    ax3.legend(loc='upper right',fontsize=fs)
    
    
    
    plt.tight_layout()
    
    return

def plot_approx_vs_KKT(var_diff):
    """
    Plot difference in the solutions obtained via converging sigma2 -> 0 vs KKT
    """
    fs=15
    x = np.arange(1,len(var_diff)+1)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(x,var_diff,color='#1F618D',marker='o')
    
    ax.set_yscale('log')
    yrange = np.logspace(-1,-7,7)
    ax.set_yticks(yrange)
    exps = np.arange(-1,-8,-1)
    ax.set_yticklabels([str('$10^{%i}$'%i) for i in exps],fontsize=fs-2)
    ax.set_ylabel('max |$\Sigma_{\hat{a}}(KKT) - \Sigma_{\hat{a}}(approx)$|',fontsize=fs)
       
    xrange = np.arange(1,len(var_diff)+1,2)
    ax.set_xticks(xrange)
    
    ax.set_xticklabels([str('$10^{-%i}$'%i) for i in xrange],fontsize=fs-2)
    ax.set_xlabel('$\sigma_2/\sigma_1$',fontsize=fs)
    ax.grid()
    
#%%
def two_classes_full_space():
    """
    Two classes that span the whole space
    The study is testing sigma2 converging to zero and study the eigenvalues
    """
    r=8
    n=11
    p = n
    p2 = 5 #ref st
    p1 = p-p2 #lcs
    sigma1 = 1
    results_E_r10 = []
    sigma2 = sigma1/10

    # functions
    Psi,C1,H1,Theta1,Phi1,sigma1,C2,H2,Theta2,Phi2,sigma2,Var,a_hat = full_space_positioning(U,r,p,p1,p2,sigma1,sigma2)
    lambda1,lambda2,diag1,diag2,lambdaVar = spectral_information(Phi1,Phi2,Var)
    results = E_optimal_multiclass(Psi,p1,sigma1,sigma2,save_iter=False)
    lambda1,lambda2,diag1,diag2,lambdaVar,v1,v2,vVar = spectral_information(Phi1,Phi2,Var)
    # Variance eigenvalues as a function of lambda2
    print(20*'-')
    print('Variance eigenvalues directly computed:')
    print(np.sort(lambdaVar))
    print('Variance eigenvalues as a function of lambda2:')
    print(np.sort(sigma1*( (1+lambda2*((sigma1/sigma2)-1))**(-1) )))
    print(20*'-')
    print(results)
    
    
    return
    
#%%
def main():
    
    return

if __name__=='__main__':
    # data set parameters
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    # data set parameters
    RefStations = ['Badalona','Ciutadella','Eixample','El-Prat','Gracia','Manlleu','Palau-Reial','Sant-Adria','Tona','Vall_Hebron','Vic']#Full_list: ['Badalona','Ciutadella','Eixample','El-Prat','Fabra','Gracia','Manlleu','Palau-Reial','Sant-Adria','Tona','Vall_Hebron','Vic']
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    # training and testing set division
    trainingSet_end = '2019-01-01'
    testingSet_end = '2020-01-01'
    
    # data set
    ds = create_dataSet()
    df_train, df_test = TrainTestSplit(ds.ds,trainingSet_end,testingSet_end)
    #df_train.drop(df.index[(df==0.0).any(1)],inplace=True) # remove entries with zero value
    # SVD decomposition
    X_train = LRD.Snapshots_matrix(df_train)
    X_,U,S,V = LRD.low_rank_decomposition(X_train)
    
    r=8
    n=11
    p = n

    # number of sensors per class
    p2 = 5 #ref st
    p2_c = p-p2 #empty+lcs
    p1 = p2_c #lcs. if p1 = p2_c -> p3=0
    p3 = p2_c-p1 #empty
    # sensors parameters
    sigma1 = 1
    sigma2 = sigma1/10
    sigma3 = sigma1*10
    
    # sensors distributions and sparse basis
    C1,C2,C3 = sensors_distributions(n,p1,p2,p3)
    Psi = U[:,:r]
    
    # functions
    #Psi,C1,H1,Theta1,Phi1,sigma1,C2,H2,Theta2,Phi2,sigma2,Var,a_hat = full_space_positioning(U,r,p,p1,p2,sigma1,sigma2)
    #lambda1,lambda2,diag1,diag2,lambdaVar,v1,v2,vVar = spectral_information(Phi1,Phi2,Var)
    results_limit = D_optimal_limit(Psi,C1,C2,sigma1)
    results_E = E_optimal_multiclass(Psi,C1,C2,C3,p1,p2,p3,sigma1,sigma2,sigma3,save_iter=False)
    results_D = D_optimal_multiclass(Psi,C1,C2,C3,p1,p2,p3,sigma1,sigma2,sigma3,save_iter=False)
    
    
    
    
    main()
    
    

