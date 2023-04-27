#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:17:27 2023

@author: jparedes
"""
import os
import pandas as pd
import scipy
import itertools
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle
import warnings

import LoadDataSet as LDS
import LowRankDecomposition as LRD
import SensorPlacementMethods as SPM
import Plots
#%%
          
def create_dataSet(pollutant,start_date,end_date, RefStations):
    dataset = LDS.dataSet(pollutant,start_date,end_date, RefStations)
    dataset.load_dataSet(file_path)
    # remove persisting missing values
    dataset.cleanMissingvalues(strategy='stations',tol=0.1,RefStations=RefStations)
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
#%%
def loadStations():
    # data set parameters
    RefStations_list = {
                   'Badalona':'08015021',
                   'Eixample':'08019043',
                   'Gracia':'08019044',
                   'Ciutadella':'08019050',
                   'Vall-Hebron':'08019054',
                   'Palau-Reial':'08019057',
                   'Fabra':'08019058',
                   'Berga':'08022006',
                   'Gava':'08089005',
                   'Granollers':'08096014',
                   'Igualada':'08102005',
                   'Manlleu':'08112003',
                   'Manresa':'08113007',
                   'Mataro':'08121013',
                   'Montcada':'08125002',
                   'Montseny':'08125002',
                   'El-Prat':'08169009',
                   'Rubi':'08184006',
                   'Sabadell':'08187012',
                   'Sant-Adria':'08194008',
                   'Sant-Celoni':'08202001',
                   'Sant-Cugat':'08205002',
                   'Santa-Maria':'08259002',
                   'Sant-Vicenç':'08263001',
                   'Terrassa':'08279011',
                   'Tona':'08283004',
                   'Vic':'08298008',
                   'Viladecans':'08301004',
                   'Vilafranca':'08305006',
                   'Vilanova':'08307012',
                   'Agullana':'17001002',
                   'Begur':'17013001',
                   'Pardines':'17125001',
                   'Santa-Pau':'17184001',
                   'Bellver':'25051001',
                   'Juneda':'25119002',
                   'Lleida':'25120001',
                   'Ponts':'25172001',
                   'Montsec':'25196001',
                   'Sort':'25209001',
                   'Alcover':'43005002',
                   'Amposta':'43014001',
                   'La-Senla':'43044003',
                   'Constanti':'43047001',
                   'Gandesa':'43064001',
                   'Els-Guiamets':'43070001',
                   'Reus':'43123005',
                   'Tarragona':'43148028',
                   'Vilaseca':'43171002'
                   }
    RefStations = [i for i in RefStations_list.keys()]
    
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    # training and testing set division
    trainingSet_end = '2019-01-01'
    testingSet_end = '2020-01-01'
    
    # data set
    ds = create_dataSet(pollutant,start_date,end_date, RefStations)
    df_train, df_test = TrainTestSplit(ds.ds,trainingSet_end,testingSet_end)
    
    return df_train,df_test

  
def createLCS(signal_refst,SNR_db):
    """
    Add noise to reference station measurements to simulate LCS
    """
    
    # # reference station estimated error variance (or std)
    # res = signal_true - signal_reconstructed
    # diff = res[locations[1]]# take places where will place a LCS
    # stdev = diff.std(axis=1)
    
    # # low-cost sensors error variance (or std)
    # scale = stdev/np.sqrt(kappa[1])
    # # add noise to measured signal at lcs locations
    # rng = np.random.default_rng(seed=92)
    # noise = np.array([rng.normal(loc=0.0,scale=np.sqrt(i),size=diff.shape[1]) for i in scale])
    # measurement_lcs_noisy = signal_true[locations[1]] + noise
    
    # power and decibels
    signal_power = signal_refst **2
    signal_power_avg = signal_power.mean(axis=1)
    signal_db = 10*np.log10(signal_power_avg)
    # noise
    noise_db = signal_db - SNR_db
    noise_power = 10**(noise_db/10)
    # white noise samples    
    rng = np.random.default_rng(seed=92)
    noise =  np.array([rng.normal(loc=0.,scale=np.sqrt(i),size=signal_refst.shape[1]) for i in noise_power])
    signal_lcs = signal_refst + noise
    RMSE = [np.round(np.sqrt(mean_squared_error(signal_refst[i,:],signal_lcs[i,:])),2) for i in range(0,signal_refst.shape[0])]
    print(f'SNR = {SNR_db} creates data set with RMSE:\n{RMSE}')
    # stdev = signal_refst.std(axis=1)
    # scale = stdev/np.sqrt(kappa[1])
    # rng = np.random.default_rng(seed=92)
    # noise = np.array([rng.normal(loc=0.0,scale=np.sqrt(i),size=signal_refst.shape[1]) for i in scale])
    # signal_lcs =  signal_refst + noise
    return signal_lcs,np.sqrt(noise_power)

def Perturbate_RefSt(signal_refst,noise=0.1):
    """
    Add noise to reference station data to simulate a slightly perturbated reference station
    based on the ratio of the variances LCS/refSt, var(LCS) >> var(RefSt) s.t. original var(RefSt) = 1
    variances_ratio = var(RefSt)/var(LCS)
    """
    rng = np.random.default_rng(seed=92)
    noise = np.array([rng.normal(loc=0.,scale=noise,size=signal_refst.shape[1])])
    signal_perturbated = signal_refst + noise
    
    return signal_perturbated

#%% plots
 
def plot_singularvalues(S,results_path):
    fs = 10
    figx,figy = 3.5,2.5
    # singular values
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    ax.plot([i+1 for i in range(len(S))],S/max(S),'o',label='$\sigma_i$')    
    #ax.set_yscale('log')
    yrange = np.logspace(-2,0,3)
    ax.set_yticks(yrange)
    ax.set_yticklabels(['$10^{-2}$','$10^{-1}$','$1$'],fontsize=fs)
    ax.set_ylabel('Normalizaed singular values',fontsize=fs)
    ax.set_xlabel('$i$-th singular value',fontsize=fs)
    yrange = np.arange(0.0,1.1,0.1)
    xrange = np.arange(0,len(S)+5,5)
    ax.set_xticks(xrange[1:])
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    
    #ax.set_title('Snapshots matrix singular values',fontsize=fs)
    ax.grid()
    
    ax.tick_params(axis='both', which='major', labelsize=fs)
    fig.tight_layout()
    
    fig1 = plt.figure(figsize=(figx,figy))
    ax1 = fig1.add_subplot(111)
    ax1.plot([i+1 for i in range(len(S))],np.cumsum(S)/np.sum(S),'o',color='orange',label='Cumulative energy')
    ax1.set_ylabel('Cumulative energy',fontsize=fs)
    yrange = np.arange(0.0,1.2,0.3)
    ax1.set_yticks(yrange)
    ax1.set_yticklabels(np.round(ax1.get_yticks(),decimals=1),fontsize=fs)
    ax1.set_xlabel('$i$-th singular value',fontsize=fs)
    xrange = np.arange(0,len(S)+5,5)
    ax1.set_xticks(xrange[1:])
    ax1.set_xticklabels(ax1.get_xticks(),fontsize=fs)

    ax1.grid()

    ax1.tick_params(axis='both', which='major', labelsize=fs)
    fig1.tight_layout()
    

    
    return fig,fig1

def plot_reconstructionErrors(reconstruction_error_empty,save=False):
    
    fs = 10
    figx,figy = 3.5,2.5
    # singular values
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    ax.errorbar(reconstruction_error_zero.keys(),reconstruction_error_zero.values(),color='#1a5276',label='Locations with Ref.St.',marker='o')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    ax.set_xlabel('Regularization parameter $\lambda$',fontsize=fs)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(),fontsize=fs)
    ax.set_ylabel('RMSE',fontsize=fs)
    ax.set_title('Reconstruction error',fontsize=fs)
    ax.grid()
    ax.legend(loc='upper right',fontsize=fs)
    ax.set_xscale('symlog')
    ax.set_xlim(list(reconstruction_error_empty.keys())[0]-1,2*list(reconstruction_error_empty.keys())[-1])
    ax.set_ylim(0,2500)
    fig.tight_layout()
   
    if save:
        plt.savefig('Plot_reconstruction_error_vs_lambda.png',dpi=600,format='png')
   
    return fig

#%% save
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

#%% Experiment
def main(Psi,p_eps,p_zero,sigma_eps,sigma_zero,signal_refst,signal_lcs,X_train):
    """
    Convex optimization experiment
    Reconstruction error of non-measured places vs actual value from library
    -----------------------------------
    number of sensors for each class is fixed
    eigenmodes are fixed
    variances are fixed
    regularization value ranges
    """
    lambdas = np.concatenate(([0],np.logspace(-3,3,7)))
    locations = {el:0 for el in lambdas}
    D_optimal_metric = {el:0 for el in lambdas}
    objective_metric =  {el:0 for el in lambdas}
    reconstruction_error_zero = {el:0 for el in lambdas}
    reconstruction_error_eps = {el:0 for el in lambdas}
    reconstruction_error_empty = {el:0 for el in lambdas}
    for lambda_reg in lambdas:
        # get optimal locations and regularized objective function
        H1_optimal,H2_optimal,obj_value = SPM.ConvexOpt_limit(Psi, sigma_eps, sigma_zero, p_eps, p_zero,lambda_reg)
        if np.abs(obj_value) == float('inf'):
            warnings.warn("Convergence not reached")
            continue
            
        # compute D-optimal non-regularized objective function
        Phi1_optimal = Psi.T@np.diag(H1_optimal)@Psi
        Phi2_optimal = Psi.T@np.diag(H2_optimal)@Psi
        D_optimal_obj = (1/sigma_eps)*Phi1_optimal + (1/sigma_zero)*Phi2_optimal
        
        # measurement
        loc1 = H1_optimal.argsort()[-p_eps:]
        loc2 = H2_optimal.argsort()[-p_zero:]
        loc_empty = [i for i in np.arange(0,n) if i not in loc1 and i not in loc2]
        In = np.identity(n)
        C_eps = In[loc1,:]
        C_zero = In[loc2,:]
        C_empty = In[loc_empty,:]
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        Theta_empty = C_empty@Psi
        
        y_eps = C_eps@signal_lcs
        y_zero = C_zero@signal_refst
        y_empty = C_empty@X_train
        
        # prediction at non-measured points
        Theta = np.concatenate([Theta_eps,Theta_zero],axis=0)
        y = np.concatenate([y_eps,y_zero],axis=0)
        PrecisionMat = np.diag(np.concatenate([np.repeat(1/sigma_eps,p_eps),np.repeat(1/sigma_zero,p_zero)]))
        beta_hat = np.linalg.inv(Theta.T@PrecisionMat@Theta)@Theta.T@PrecisionMat@y
        
        y_zero_hat = Theta_zero@beta_hat
        y_eps_hat = Theta_eps@beta_hat
        y_empty_hat = Theta_empty@beta_hat
        
        # store results
        locations[lambda_reg] = [H1_optimal,H2_optimal]
        D_optimal_metric[lambda_reg] = -1*np.log(np.linalg.det(D_optimal_obj))
        objective_metric[lambda_reg] = obj_value
        reconstruction_error_zero[lambda_reg] = [np.round(np.sqrt(mean_squared_error(y_zero[i,:], y_zero_hat[i,:])),2) for i in range(y_zero.shape[0])]
        reconstruction_error_eps[lambda_reg] = [np.round(np.sqrt(mean_squared_error(y_eps[i,:], y_eps_hat[i,:])),2) for i in range(y_eps.shape[0])]
        reconstruction_error_empty[lambda_reg] = [np.round(np.sqrt(mean_squared_error(y_empty[i,:],y_empty_hat[i,:])),2) for i in range(y_empty.shape[0])]
        
        

    return locations,objective_metric,D_optimal_metric,reconstruction_error_zero,reconstruction_error_eps,reconstruction_error_empty


if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    df_train,df_test = loadStations()
    # SVD decomposition
    X_train = LRD.Snapshots_matrix(df_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.T).T
    U,S,V = LRD.low_rank_decomposition(X_train_scaled)
    
    X_test = LRD.Snapshots_matrix(df_test)
    # create LCSs signal
    var_ratio = 0.001 #<1!
    if var_ratio>=1:
        raise Warning(f'Variances ratio is var(refSt)/var(LCS) but value is larger than one! (ratio = {var_ratio})')
    
    X_train_eps = Perturbate_RefSt(X_train_scaled,noise=0.1)
    X_train_zero = Perturbate_RefSt(X_train_scaled,noise=var_ratio*0.1)
    
    
    # parameters
    
    ## basis
    r=35 # 35: number of eigenmodes for 90% of cumulative energy
    n=U.shape[0]
    p = n
    Psi = U[:,:r]
    
    ## number of sensors per class
    p_zero = r-1 #ref st
    p2_c = p-p_zero #empty+lcs
    p_eps = r-p_zero
    p_empty = p2_c-p_eps #empty
    sigma_eps = 0.1**2
    sigma_zero = var_ratio*sigma_eps
    sigma_empty = sigma_eps*1000
    print(f'Sensor placement parameters\n{n} locations\n{r} eigenmodes\n{p_eps+p_zero} sensors in total\n{p_eps} LCSs - variance = {sigma_eps}\n{p_zero} Ref.St. - variance = {sigma_zero}\n{p_empty} Empty locations - variance = {sigma_empty}')
    input('Press Enter to continue...')
    # experiment
    optimal_locations,objective_func,D_optimal_metric,reconstruction_error_zero,reconstruction_error_eps,reconstruction_error_empty = main(
        Psi,
        p_eps,
        p_zero,
        sigma_eps,
        sigma_zero,
        X_train_zero,
        X_train_eps,
        X_train_scaled
        )
    # show results
    print('Objective metric\nLambda\t val')
    for lambda_reg,val in zip(objective_func.keys(),objective_func.values()):
        print(f'{lambda_reg}\t{val}')
        
    print(f'{p_zero} Reference station locations\nLambda\t RMSE')
    for lambda_reg,val in zip(reconstruction_error_zero.keys(),reconstruction_error_zero.values()):
        print(f'{lambda_reg}\t {val}')
        
    print(f'\n{p_eps} LCSs locations\nLambda\t RMSE')
    for lambda_reg,val in zip(reconstruction_error_eps.keys(),reconstruction_error_eps.values()):
        print(f'{lambda_reg}\t {val}')
    
    print(f'\n{p_empty} Empty locations\nLambda\t RMSE')
    for lambda_reg,val in zip(reconstruction_error_empty.keys(),reconstruction_error_empty.values()):
        print(f'{lambda_reg}\t {val}')
        
    save_results(objective_func,reconstruction_error_empty,optimal_locations,p_eps,p_zero,p_empty,r,var_ratio,results_path)
