#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization proposition tested on synthetic data
Created on Tue May  2 15:41:33 2023

@author: jparedes
"""
import os
import pandas as pd
import scipy
import networkx as nx
import itertools
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle
import warnings

import SensorPlacementMethods as SPM
#%% data pre-processing
def createData(mean,n,m):
    G=nx.fast_gnp_random_graph(n,0.5,seed=92)
    adj_matrix = nx.adjacency_matrix(G)
    M = adj_matrix.toarray()
    np.fill_diagonal(M, 1)
    
    rng = np.random.default_rng(seed=92)
    sigmas = rng.random(size=(n,n))
    sigmas = (sigmas + sigmas.T)/2
    CovMat = np.multiply(sigmas,M)
    np.fill_diagonal(CovMat,np.ones(CovMat.shape[0]))
    X = rng.multivariate_normal(mean, CovMat, m).T
    
    return X

def Perturbate_data(signal_refst,noise=0.1):
    """
    Add noise to reference station data to simulate a slightly perturbated reference station
    based on the ratio of the variances LCS/refSt, var(LCS) >> var(RefSt) s.t. original var(RefSt) = 1
    variances_ratio = var(RefSt)/var(LCS)
    """
    rng = np.random.default_rng(seed=92)
    noise = np.array([rng.normal(loc=0.,scale=np.sqrt(noise),size=signal_refst.shape[1])])
    signal_perturbated = signal_refst + noise
    
    return signal_perturbated
#%% random placement functions
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

def randomPlacement(n,p_eps,p_zero,p_empty,sigma_eps,sigma_zero,Psi,signal_lcs,signal_refst,var_ratio,num_samples=10):
    """
    Draw random locations for reconstruction
    """
    
    locations = {}
    C_eps_all, C_zero_all, C_empty_all = sensors_distributions(n,p_eps,p_zero,p_empty)
    
    rng = np.random.default_rng(seed=92)
    locs = rng.integers(0,len(C_eps_all),num_samples)
    
    if p_eps!= 0:
        reconstruction_error_empty = {('loc_eps','loc_empty'):'val'}
        for i in locs:
            C_eps = C_eps_all[i]
            C_zero_ = C_zero_all[i]
            C_empty_ = C_empty_all[i]
            
            j = rng.integers(0,len(C_empty_))
            C_zero = C_zero_[j]
            C_empty = C_empty_[j]

            # define theta
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            # sample at locations
            y_empty = C_empty@signal_refst
                
            beta_hat = beta_KKT(C_eps,C_zero,Theta_eps,Theta_zero,signal_refst,signal_lcs)
            y_empty_hat = Theta_empty@beta_hat
            
            loc_zero = np.argwhere(C_zero==1)[:,-1]
            loc_eps = np.argwhere(C_eps==1)[:,-1]
            loc_empty = np.argwhere(C_empty==1)[:,-1]
            locations[i] = [loc_eps,loc_zero,loc_empty]
            
            reconstruction_error_empty[loc_eps,loc_empty] = [np.round(np.sqrt(mean_squared_error(y_empty[i,:],y_empty_hat[i,:])),2) for i in range(y_empty.shape[0])]

    else:
        reconstruction_error_empty = {('loc_empty'):'val'}
        C_zero_ = C_zero_all[0]
        C_empty_ = C_empty_all[0]
        
        for i in np.arange(len(C_empty_)):
            
            C_empty=C_empty_[i]
            C_zero =C_zero_[i]
            
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            y_zero = C_zero@signal_refst
            y_empty = C_empty@signal_refst
            
            beta_hat = np.linalg.inv(Theta_zero.T@Theta_zero)@Theta_zero.T@y_zero
            y_empty_hat = Theta_empty@beta_hat
            
            loc_empty = np.argwhere(C_empty==1)[0][-1]
            reconstruction_error_empty[loc_empty] = [np.round(np.sqrt(mean_squared_error(y_empty[i,:],y_empty_hat[i,:])),2) for i in range(y_empty.shape[0])]
    
    
        
        
        
    return reconstruction_error_empty

#%% estimations
def beta_KKT(C_eps,C_zero,Theta_eps,Theta_zero,signal_refst,signal_lcs):
    """
    Compute estimator in the exact case of variance ref.st. = 0 via KKT
    """
    Phi_eps = Theta_eps.T@Theta_eps
    Phi_zero = Theta_zero.T@Theta_zero
    
    A = 4*Phi_eps@Phi_eps+Phi_zero
    B = 2*Phi_eps@Theta_zero.T
    D = Theta_zero@Theta_zero.T
    
    A_inv = np.linalg.inv(A)
    Schur = D - B.T@A_inv@B
    Schur_inv = np.linalg.inv(Schur)
    
    # (K^T@K)^-1
    F = -A_inv@B@Schur_inv
    E = A_inv -F@B.T@A_inv
    
    # pseudo inverse: eq to np.linalg.inv(K) but the first block has inverse for K^T@K
    #G = F.T
    #H = Schur_inv.copy()
    Z = np.zeros((Theta_zero.shape[0],Theta_zero.shape[0]))
    K = np.block([[2*Phi_eps,Theta_zero.T],[Theta_zero,Z]])
    #K_pinv = np.block([[2*E@Phi_eps+F@Theta_zero,E@Theta_zero.T],[2*G@Phi_eps+H@Theta_zero,G@Theta_zero.T]])
    
    # estimator 
    T1 = (2*E@Phi_eps + F@Theta_zero)@(2*Theta_eps.T)
    T2 = E@Theta_zero.T
    y_eps = C_eps@signal_lcs
    y_zero = C_zero@signal_refst
    beta_hat = T1@y_eps + T2@y_zero
    
    return beta_hat

def beta_GLS(Theta_eps,Theta_zero,y_eps,y_zero,sigma_eps,sigma_zero,p_eps,p_zero):
    """
    Compute estimator using GLS.
    Variance of ref.st. is almost zero but != 0
    """
    Theta = np.concatenate([Theta_eps,Theta_zero],axis=0)
    y = np.concatenate([y_eps,y_zero],axis=0)
    PrecisionMat = np.diag(np.concatenate([np.repeat(1/sigma_eps,p_eps),np.repeat(1/sigma_zero,p_zero)]))
    beta_hat = np.linalg.inv(Theta.T@PrecisionMat@Theta)@Theta.T@PrecisionMat@y
    
    return beta_hat

def beta_cov(Theta_eps,Theta_zero,Psi,var_eps):
    """
    Return KKT estimator covariance
    """
    Phi_eps = Theta_eps.T@Theta_eps
    Phi_zero = Theta_zero.T@Theta_zero
    
    # KKT matrix
    A = 4*Phi_eps@Phi_eps+Phi_zero
    B = 2*Phi_eps@Theta_zero.T
    D = Theta_zero@Theta_zero.T
    
    # inverse necessary matrices
    A_inv = np.linalg.inv(A)
    Schur = D - B.T@A_inv@B
    Schur_inv = np.linalg.inv(Schur)
    
    F = -A_inv@B@Schur_inv
    E = A_inv -F@B.T@A_inv
    
    
    # projector
    T_eps = (2*E@Phi_eps + F@Theta_zero)@(2*Theta_eps.T)
    
    # covariance
    #F = --2*A_inv@Phi_eps@Theta_zero.T@Schur_inv
    #M = (2*F.T@Phi_eps + Schur_inv@Theta_zero)@(2*Theta_eps.T)
    #Sigma_beta = var_eps*M@M.T
    Sigma_beta = var_eps*(T_eps@T_eps.T)
    
    return Sigma_beta

def check_consistency(n,r,p_zero,p_eps,p_empty):
    if p_zero + p_eps < r:
        raise Warning(f'The assumptions on the KKT probelm (left-invertibility of [Theta_eps;Theta_zero] == linearly independent columns) impose that the number of eigenmodes has to be smaller than the number of sensors.\nReceived:Eigemodes: {r}\nSensors: {p_zero + p_eps}')
    if p_zero > r:
        raise Warning(f'The assumptions on the KKT problem (right-invertibility of Theta_zero == linearly independent rows) impose that the number of reference stations has to be smaller than the number of eigenmodes.\nReceived:\nRef.St: {p_zero}\nEigenmodes: {r}')
    if p_zero <= r and r <= p_zero + p_eps:
        print(f'Number of sensors pass consistency check: Ref.St ({p_zero}) <= r ({r}) <= Ref.St ({p_zero}) + LCSs ({p_eps})')
    return

#%%

def save_results(objective_function,reconstruction_error_empty,optimal_locations,p_eps,p_zero,p_empty,r,var_ratio,results_path):
    print('Saving results')
    # save objective function
    fname = f'ObjectiveFunction_vs_lambda_RefSt{p_zero}_LCS{p_eps}_Empty{p_empty}_r{r}_varRatio{var_ratio}.pkl'
    with open(results_path+fname, 'wb') as handle:
        pickle.dump(objective_function, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save optimal locations
    fname = f'OptimalLocations_vs_lambda_RefSt{p_zero}_LCS{p_eps}_Empty{p_empty}_r{r}_varRatio{var_ratio}.pkl'
    with open(results_path+fname, 'wb') as handle:
        pickle.dump(optimal_locations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save reconstruction error
    fname = f'ReconstructionErrorEmpty_vs_lambda_RefSt{p_zero}_LCS{p_eps}_Empty{p_empty}_r{r}_varRatio{var_ratio}.pkl'
    with open(results_path+fname, 'wb') as handle:
        pickle.dump(reconstruction_error_empty, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Results saved: {results_path}')
    
    return
#%%
def locations_vs_lambdas(Psi,p_eps,p_zero,p_empty,sigma_eps,sigma_zero,signal_lcs,signal_refst):
    """
    Optimal locations fopund via D-optimal convex relaxation
    Refernce stations are simulated for the limit of low variances
    
    Results for different regularization values
    """
    
    lambdas = np.concatenate(([0],np.logspace(-3,3,7)))
    # results to save
    weights = {el:0 for el in lambdas}
    locations = {el:0 for el in lambdas}
    objective_metric =  {el:0 for el in lambdas}
    
    for lambda_reg in lambdas:
        H1_optimal,H2_optimal,obj_value = SPM.ConvexOpt_limit(Psi, sigma_eps, sigma_zero, p_eps, p_zero,lambda_reg)
        if np.abs(obj_value) == float('inf'):
            warnings.warn("Convergence not reached")
            continue    
        
        if p_eps != 0 and p_zero != 0:# using both type of sensors
            
            loc2 = np.sort(H2_optimal.argsort()[-p_zero:])
            loc1 = np.sort([i for i in np.argsort(H1_optimal) if i not in loc2][-p_eps:])
            loc_empty = np.sort([i for i in np.arange(0,n) if i not in loc1 and i not in loc2])
            
            
        elif p_eps==0:# only ref.st. no LCSs
            loc1 = []
            loc2 = np.sort(H2_optimal.argsort()[-p_zero:])
            loc_empty = np.sort([i for i in np.arange(0,n) if i not in loc2])

        # store results
        weights[lambda_reg] = [H1_optimal,H2_optimal]
        locations[lambda_reg] = [loc1,loc2,loc_empty]
        objective_metric[lambda_reg] = obj_value
         
            
    return weights, locations, objective_metric

def KKT_estimations(locations,p_eps,p_zero,n,sigma_eps,signal_refst,signal_lcs):
    """
    Reconstruct signal at specific locations and compute error
    """
    reg_param = locations.keys()
    reconstruction_error_empty = {el:0 for el in reg_param}
    reconstruction_error_eps = {el:0 for el in reg_param}
    reconstruction_error_zero = {el:0 for el in reg_param}
    reconstruction_error_full = {el:0 for el in reg_param}
    
   
    for l in reg_param:
        loc_eps,loc_zero,loc_empty = locations[l]
        
        if p_eps != 0 and p_zero != 0:# reconstruct using both type of sensors
          
            # placement
            In = np.identity(n)
            C_eps = In[loc_eps,:]
            C_zero = In[loc_zero,:]
            C_empty = In[loc_empty,:]
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            
            # measurement
            y_zero = C_zero@signal_refst
            y_empty = C_empty@signal_refst
            y_eps = C_eps@signal_lcs
            
            # estimations
            beta_hat = beta_KKT(C_eps,C_zero,Theta_eps,Theta_zero,signal_refst,signal_lcs)
            y_empty_hat = Theta_empty@beta_hat
            y_zero_hat = Theta_zero@beta_hat
            y_eps_hat = Theta_eps@beta_hat
            y_all_hat = Psi@beta_hat
   
        elif p_eps==0:# reconstruct only using Ref.St.
            In = np.identity(n)
            C_zero = In[loc_zero,:]
            C_empty = In[loc_empty,:]
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            
            # measurement
            y_zero = C_zero@signal_refst
            y_empty = C_empty@signal_refst
            
            # estimations
            beta_hat = np.linalg.inv(Theta_zero.T@Theta_zero)@Theta_zero.T@y_zero
            y_empty_hat = Theta_empty@beta_hat
            y_zero_hat = Theta_empty@beta_hat
            y_all_hat = Psi@beta_hat
            
        
        reconstruction_error_empty[l] = [np.sqrt(mean_squared_error(y_empty[i,:],y_empty_hat[i,:])) for i in range(y_empty.shape[0])]
        reconstruction_error_eps[l] = [np.sqrt(mean_squared_error(y_eps[i,:],y_eps_hat[i,:])) for i in range(y_eps.shape[0])]
        reconstruction_error_zero[l] = [np.sqrt(mean_squared_error(y_zero[i,:],y_zero_hat[i,:])) for i in range(y_zero.shape[0])]
        reconstruction_error_full[l] = [np.sqrt(mean_squared_error(signal_refst[i,:],y_all_hat[i,:])) for i in range(signal_refst.shape[0])]
        
   
        
    return reconstruction_error_full, reconstruction_error_zero, reconstruction_error_eps, reconstruction_error_empty

def KKT_cov(locations,Psi,sigma_eps):
    """
    Compute residuals covariance for the KKT algorithm of the estimator
    """
    reg_param = locations.keys()
    residuals_cov = {el:0 for el in reg_param}
    
    # covariances
    for l in reg_param:
        loc_eps,loc_zero,loc_empty = locations[l]
        
        if p_eps != 0 and p_zero != 0:# reconstruct using both type of sensors
        
            # placement
            In = np.identity(n)
            C_eps = In[loc_eps,:]
            C_zero = In[loc_zero,:]
            C_empty = In[loc_empty,:]
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            
            Sigma_beta = beta_cov(Theta_eps,Theta_zero,Psi,sigma_eps)
            Sigma_residuals = Psi@Sigma_beta@Psi.T
            
        elif p_eps == 0: # reconstruct using only Ref.St.
            
            # Exact OLS
            
            Sigma_beta = np.zeros(shape=(Psi.shape[0],Psi.shape[0])) 
            Sigma_residuals = Psi@Sigma_beta@Psi.T

    return Sigma_beta, Sigma_residuals

#%%

if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    n,m = 20,100# num_stations,num_snapshots

    X = createData(np.zeros(shape=(n)), n, m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    U,S,Vt = np.linalg.svd(X_scaled,full_matrices=False)
    a = np.diag(S)@Vt
    
    #sparsity
    r = 18
    Psi = U[:,:r]
    a_sparse = a.copy()
    a_sparse[r:,:] = np.zeros(shape=(a_sparse[r:,:].shape))
    X_sparse = U@a_sparse # = Psi@a[:r,:]
    
    # Refst and LCS
    
    var_eps = 1
    var_zero = 1e-3
    var_ratio = var_zero/var_eps #should be <1
    
    X_eps = Perturbate_data(X_sparse,noise=var_eps)
    X_zero = Perturbate_data(X_sparse,noise=var_zero)
    
    # number of sensors
    p = n-1 #total
    p_zero = p-2
    p_eps = 2
    p_empty = n-(p_zero+p_eps)
    
    print(f'Sensor placement parameters\n{n} locations\n{r} eigenmodes\n{p_eps+p_zero} sensors in total\n{p_eps} LCSs - variance = {var_eps:.2e}\n{p_zero} Ref.St. - variance = {var_zero:.2e}\n{p_empty} Empty locations')
    check_consistency(n,r,p_zero,p_eps,p_empty)
    
    input('Press Enter to continue...')
    
    print('Solving Convex Optimization problem')
    weights, optimal_locations, obj_function = locations_vs_lambdas(Psi,
                                                                   p_eps,
                                                                   p_zero,
                                                                   p_empty,
                                                                   var_eps,
                                                                   var_zero,
                                                                   X_eps,
                                                                   X_sparse)
    
    reconstruction_error_full, reconstruction_error_zero, reconstruction_error_eps, reconstruction_error_empty = KKT_estimations(optimal_locations,p_eps,p_zero,n,var_eps,X_sparse,X_eps)
   
    Sigma_beta, Sigma_residuals = KKT_cov(optimal_locations,Psi,var_eps)
    

    print('Objective function\nLambda\t val')
    for lambda_reg,val in zip(obj_function.keys(),obj_function.values()):
        print(f'{lambda_reg}\t{val}')
       
    print(f'\n{p_empty} Reconstruction errors\nLambda\t RMSE')
    for lambda_reg,val in zip(reconstruction_error_full.keys(),reconstruction_error_full.values()):
        print(f'{lambda_reg}\t {val}')

    #save_results(objective_func,reconstruction_error_empty,locations,p_eps,p_zero,p_empty,r,var_ratio,results_path)
    
    # print('Random placement')
    # reconstruction_error_empty_random = randomPlacement(
    #     n,
    #     p_eps,
    #     p_zero,
    #     p_empty,
    #     var_eps,
    #     var_zero,
    #     Psi,
    #     X_eps,
    #     X_sparse,
    #     var_ratio=var_ratio,
    #     save=True)
    
    # print('empty_loc\t RMSE')
    # for k,val in zip(reconstruction_error_empty_random.keys(),reconstruction_error_empty_random.values()):
    #     print(f'{k}\t {val}')
        
    # fname = f'ReconstructionErrorEmpty_Randomplacement_RefSt{p_zero}_LCS{p_eps}_Empty{p_empty}_r{r}_varRatio{var_ratio}.pkl'
    # with open(results_path+fname, 'wb') as handle:
    #     pickle.dump(reconstruction_error_empty, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print('------------\nAll Finished\n------------')
    
    
