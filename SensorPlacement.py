#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor placement for reconstruction

Created on Fri Feb 24 15:15:56 2023

@author: jparedes
"""
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error
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

#%% Sensor placement algorithms
def SensorPlacement(Psi,p,sensors_numbers,method='D-optimal'):
    """
    Determine locations of the p sensors 
    y = C*Psi*a + e
    """
    # if method not in ['QR','D-optimal']:
        
    print(f'chosen method: {method}')
    if p != sum(sensors_numbers):
        raise ArithmeticError('Total number of sensors does not match the distribution of sensors for each class.')
    if len(kappa) != len(sensors_numbers):
        raise ArithmeticError('Specify the ratio between sensors variances')
    
    if method == 'QR':
        """
        (Psi_r @ Psi_r.T)C.T = QR
        """
        C, locations = SPM.QR_pivoting(Psi, p)

    elif method == 'D-optimal':
        """
        argmax det Theta.T Theta
        """
        C, locations = SPM.D_optimal(Psi,p)
        
    elif method == 'ConvexOpt':
        """
        argmax log det sum(beta_i Theta_i.T Theta_i) st sum(beta_i)=p
        """
        C, locations = SPM.ConvexOpt(Psi,p,k=kappa,sensors = sensors_numbers)

    elif method == 'Gappy-POD':
        pass
    elif method == 'DEIM':
        pass
    
    Theta = C @ Psi # basis measured at those points
    print(f'Theta matrix condition number: {np.linalg.cond(Theta,p=2):.2f}')
    return Theta, C, locations
#%% Signal reconstruction functions

def measuredSignal(signal,locations):
    """
    Signal measured at certain locations given by positions
    """
    # signal measured at specific locations
    y = signal[locations]
    return y

def reconstructSignal(Psi,a):
    """
    Reconstruct signal in the whole space based on sparse measured signal
    """    
    signal_reconstructed = Psi @ a # whole signal reconstruction
    return signal_reconstructed
    
def reconstruct_NoisySignal(signal_true, signal_reconstructed,locations,kappa):
    """
    Add noise to LCS measurements at their locations
    """
    # reference station estimated error variance (or std)
    res = signal_true - signal_reconstructed
    diff = res[locations[1]]# take places where will place a LCS
    stdev = diff.std(axis=1)
    
    # low-cost sensors error variance (or std)
    scale = stdev/np.sqrt(kappa[1])
    # add noise to measured signal at lcs locations
    rng = np.random.default_rng(seed=92)
    noise = np.array([rng.normal(loc=0.0,scale=np.sqrt(i),size=diff.shape[1]) for i in scale])
    measurement_lcs_noisy = signal_true[locations[1]] + noise
    return measurement_lcs_noisy

def PlaceLCS(signal,locations, measurement_lcs_noisy):
    # signal measured by reference stations
    y_refst = signal[locations[0]]
    # signal measured by LCS
    y_lcs = measurement_lcs_noisy
    
    signal_ = np.concatenate((y_refst,y_lcs))
    df = pd.DataFrame(signal_,index = np.concatenate(locations))
    df.sort_index(inplace=True)
    signal_measured = df.values
    
    return signal_measured
#%% Scoring metrics
def scoring(signal,signal_reconstructed,locations_measured,locations_not_measured):
    # real values
    y_true =  signal# signal not measured at those points
    # predicted values
    y_pred = signal_reconstructed
    y_pred[y_pred<0] = 0.0
    # scoring metrics
    RMSE_all = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE of full network
    RMSE_stations = [np.sqrt(mean_squared_error(y_true[i,:],y_pred[i,:])) for i in range(y_true.shape[0])] # RMSE per station
    RMSE_locations_measured = np.sqrt(mean_squared_error(y_true[locations_measured], y_pred[locations_measured])) # RMSE of measurement locations
    if len(locations_not_measured)>1: # RMSE of not measured locations
        RMSE_locations_reconstructed = np.sqrt(mean_squared_error(y_true[locations_not_measured], y_pred[locations_not_measured]))
    else:
        RMSE_locations_reconstructed = []
    return y_true, y_pred, RMSE_all, RMSE_stations, RMSE_locations_measured, RMSE_locations_reconstructed

def save_results(r,p,locations,locations_measured,locations_not_measured,RefStations,RMSE_all_train,RMSE_stations_train,RMSE_locations_measured_train,RMSE_locations_reconstructed_train,RMSE_all_test,RMSE_stations_test,RMSE_locations_measured_test,RMSE_locations_reconstructed_test,placement_method):
    print(f'Saving results at {results_path}')
    # save RMSE
    fname = f'RMSE_{placement_method}_r{r}_p{p}.txt'
    f = open(results_path+fname,'a')
    print(15*'-',file=f)
    print(f'Results obtained {datetime.now()}',file=f)
    # save network architecture
    print(f'Refernce Stations: {RefStations}',file=f)
    print(f'Using {r} eigenmodes - {p} sensors - {len(sensors_numbers)} types of sensors - k = {kappa} distributed as {sensors_numbers}',file=f)
    print(f'Measurement stations: {[RefStations[i] for i in locations_measured]}',file=f)
    print(f'Reconstructed stations: {[RefStations[i] for i in locations_not_measured]}',file=f)

    # save training RMSE
    print('Training set scores',file=f)
    print(f'Full network RMSE = {RMSE_all_train}',file=f)
    print(f'Individual stations RMSE = {RMSE_stations_train}',file=f)
    print(f'RMSE at placement locations: {[RMSE_stations_train[i] for i in locations_measured]}',file=f)
    print(f'Average RMSE at placement locations: {RMSE_locations_measured_train}',file=f)
    if len(locations_not_measured)>=1:
        print(f'RMSE at reconstructed locations: {[RMSE_stations_train[i] for i in locations_not_measured]}',file=f)
        print(f'Average RMSE at reconstructed locations: {RMSE_locations_reconstructed_train}',file=f)
    
    # save testing RMSE
    print('\nTesting set scores',file=f)
    print(f'Full network RMSE = {RMSE_all_test}',file=f)
    print(f'Individual stations RMSE = {RMSE_stations_test}',file=f)
    print(f'Individual RMSE at placement locations test: {[RMSE_stations_test[i] for i in locations_measured]}',file=f)
    print(f'Average RMSE at placement locations: {RMSE_locations_measured_test}',file=f)
    if len(locations_not_measured)>=1:
        print(f'Individual RMSE at reconstructed locations: {[RMSE_stations_test[i] for i in locations_not_measured]}',file=f)
        print(f'Average RMSE at reconstructed locations: {RMSE_locations_reconstructed_test}',file=f)
    
    if len(sensors_numbers)>1:
        print(f'Locations with Reference stations: {[RefStations[locations[0][i]] for i in range(len(locations[0]))]}',file=f)
        print(f'Locations with LCS: {[RefStations[locations[1][i]] for i in range(len(locations[1]))]}',file=f)
    
    f.close()
    print(f'Results saved as {fname}')
    
    
    

#%%
def main():
    # load data set and create snapshots matrix
    ds = create_dataSet()
    df_train, df_test = TrainTestSplit(ds.ds,trainingSet_end,testingSet_end)
    #df_train.drop(df.index[(df==0.0).any(1)],inplace=True) # remove entries with zero value
    
    # SVD decomposition
    X_train = LRD.Snapshots_matrix(df_train)
    Plots.CovarianceMatrix_plot(X_train,RefStations)
    X_,U,S,V = LRD.low_rank_decomposition(X_train)
    fig_S, fig_U = Plots.visualize_singularvalues(S), Plots.visualize_singularVectors(U)
    # sensor placement
    print(f'Using {r} basis vectors')
    Psi = U[:,:r]
    print(f'Placement of {p} sensors')
    Theta, C, locations = SensorPlacement(Psi,p,sensors_numbers,method=placement_method)
    
    # Place sensors at those specific locations
    locations_measured = np.argwhere(C==1)[:,1]# locations in R^n
    locations_not_measured = [i for i in range(len(RefStations)) if i not in locations_measured]
    
    print(f'Sensors placed in: {[RefStations[i] for i in locations_measured]}')
    x_train_measured = measuredSignal(X_train,locations_measured) # ref station measurements
    a, residuals,rank, sing_vals = linalg.lstsq(Theta,x_train_measured) # modes coefficients for sparse representation
    signal_reconstructed_train = reconstructSignal(Psi,a)
    
    # Replace RefStation sensors with LCS sensors
    if sensors_numbers[1] != 0:
        measurements_lcs = reconstruct_NoisySignal(X_train, signal_reconstructed_train,locations,kappa)
        if sensors_numbers[1] != p: # reallocate LCS along with Reference Stations
            signal_measured = PlaceLCS(X_train,locations, measurements_lcs)
        else:
            signal_measured = measurements_lcs.copy()
        # reconstruct using LCS with noisy measurements at certain locations
        a, residuals,rank, sing_vals = linalg.lstsq(Theta,signal_measured)
        signal_reconstructed_train = reconstructSignal(Psi,a)
        
    
    # scoring
    
    y_true_train, y_pred_train, RMSE_all_train, RMSE_stations_train, RMSE_locations_measured_train, RMSE_locations_reconstructed_train = scoring(X_train,
                                                                              signal_reconstructed_train,
                                                                              locations_measured,
                                                                              locations_not_measured)
    
    
    # testing
    X_test = LRD.Snapshots_matrix(df_test)
    x_test_measured = measuredSignal(X_test,locations_measured)
    a, residuals,rank, sing_vals = linalg.lstsq(Theta,x_test_measured) # modes coefficients for sparse representation
    signal_reconstructed_test = reconstructSignal(Psi,a)
    
    # Replace RefStation sensors with LCS sensors
    if sensors_numbers[1] != 0:
        measurements_lcs = reconstruct_NoisySignal(X_test, signal_reconstructed_test,locations,kappa)
        if sensors_numbers[1] != p: # reallocate LCS along with Reference Stations
            signal_measured = PlaceLCS(X_test,locations, measurements_lcs)
        else:
            signal_measured = measurements_lcs.copy()
        # reconstruct using LCS with noisy measurements at certain locations
        a, residuals,rank, sing_vals = linalg.lstsq(Theta,signal_measured)
        signal_reconstructed_test = reconstructSignal(Psi,a)
    
    
    y_true_test, y_pred_test, RMSE_all_test, RMSE_stations_test, RMSE_locations_measured_test, RMSE_locations_reconstructed_test = scoring(X_test,
                                                                          signal_reconstructed_test,
                                                                          locations_measured,
                                                                          locations_not_measured)
    
    # plot: only for cases where there are places without sensors
    if len(locations_not_measured)>=1:
        Plots.plot_true_vs_predicted(y_true_train,y_pred_train,
                                                        RefStations,RMSE_stations_train,
                                                        locations_not_measured)
        Plots.plot_true_vs_predicted(y_true_test,y_pred_test,
                                                        RefStations,RMSE_stations_test,
                                                        locations_not_measured)
    
        Plots.scatter_true_vs_predicted(y_true_test,y_pred_test,RefStations,RMSE_stations_test,locations_not_measured)
        Plots.residuals_plot(y_true_test,y_pred_test,RefStations,RMSE_stations_test,locations_not_measured)
    
    # once RMSE for all r exists
    ## Plots.RMSE_testAll_vs_r()

    save_results(r,p,locations,locations_measured,locations_not_measured,RefStations,RMSE_all_train,RMSE_stations_train,RMSE_locations_measured_train,RMSE_locations_reconstructed_train,RMSE_all_test,RMSE_stations_test,RMSE_locations_measured_test,RMSE_locations_reconstructed_test,placement_method)
    
    return X_,S,Psi,Theta,C,locations_measured,fig_S,y_true_train,y_pred_train,RMSE_all_train,RMSE_stations_train,y_true_test,y_pred_test,RMSE_all_test,RMSE_stations_test

if __name__ == '__main__':
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
    
    # sensor placement parameters
    placement_method = 'ConvexOpt' # options = QR, D-optimal, ConvexOpt
    
    r = 11 #  low-rank basis
    p = 11 # Total number of sensors (at least r)
    # multi-class of sensors
    kappa = [1,2/3] # variances ratio
    for n_lcs in np.arange(0,p+1,1):
        sensors_numbers = [p-n_lcs,n_lcs] # number of sensors for each class
        X_,S,Psi,Theta,C,locations_measured,fig_S,y_true_train,y_pred_train,RMSE_all_train,RMSE_stations_train,y_true_test,y_pred_test,RMSE_all_test,RMSE_stations_test = main()
        plt.close('all')
    
