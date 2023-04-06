#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True and predicted values comparison: plots and metrics

Created on Mon Feb 27 15:09:20 2023

@author: jparedes
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% Covariance matrix of data set
def CovarianceMatrix_plot(X,RefStations):
    M = np.cov(X,bias=True)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    pos = ax.imshow(M/M.max(),cmap='Blues',interpolation='None',vmin=0.0,vmax=1.0)
    fig.colorbar(pos,ax=ax,shrink=0.8)
    ax.set_title('Normalized covariance matrix of reference stations')
    ax.set_xticks([i for i in range(len(RefStations))])
    ax.set_yticks([i for i in range(len(RefStations))])
    ax.set_xticklabels(RefStations,rotation=45)
    ax.set_yticklabels(RefStations)
    
    return fig

#%% Low-rank decomposition

def visualize_singularvalues(S):
    ls = 15
    # singular values
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.plot([i+1 for i in range(len(S))],S/max(S),'o',label='$\sigma_i$')
    ax1.plot([i+1 for i in range(len(S))],np.cumsum(S)/np.sum(S),'o',color='orange',label='Cumulative energy')
    #ax.set_yscale('log')
    ax.set_ylabel('Normalizaed singular values',fontsize=ls)
    ax1.set_ylabel('Cumulative energy',fontsize=ls)
    ax1.set_xlabel('$i$-th singular value',fontsize=ls)
    ax.set_title('Snapshots matrix singular values',fontsize=ls)
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax.set_yticks([1e0,1e-1,1e-2])
    ax.grid()
    ax1.grid()
    
    ax.tick_params(axis='both', which='major', labelsize=ls)
    ax1.tick_params(axis='both', which='major', labelsize=ls)
    plt.tight_layout()
    
    return fig
    
def visualize_singularVectors(U):
    # first singular vectors
    ls = 15
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    ax.plot(U[:,0],color='blue',label='First singular vector')
    ax1.plot(U[:,1],color='orange',label='Second singular vector')
    ax2.plot(U[:,2],color='green',label='Third singular vector')
    ax.legend()
    ax1.legend()
    ax2.legend()
    ax.grid()
    ax1.grid()
    ax2.grid()
    ax.set_title('N first singular vectors',fontsize=ls)
    return fig

#%% Algorithms results
def plot_E_optimal_p1_change(E_optimal_p2_3_s3_10,E_optimal_p2_5_s3_10,E_optimal_p2_7_s3_10,E_optimal_p2_3_s3_100,E_optimal_p2_5_s3_100,E_optimal_p2_7_s3_100,E_optimal_p2_3_s3_1000,E_optimal_p2_5_s3_1000,E_optimal_p2_7_s3_1000):
    """
    Plot E-optimal results for varying number of LCSs (p1) using a fixed number of reference stations(p2)
    Different subplots for different parametrization of empty places (sigma3)
    """
    fs = 15
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax1 = fig.add_subplot(312)
    ax2 = fig.add_subplot(313)
    
    ax.plot(np.arange(1,9),E_optimal_p2_3_s3_10,color='#117a65',label='3 reference stations',marker='o')
    ax.plot(np.arange(1,7),E_optimal_p2_5_s3_10,color='#d68910',label='5 reference stations',marker='o')
    ax.plot(np.arange(1,5),E_optimal_p2_7_s3_10,color='#1f618d',label='7 reference stations',marker='o')
        
    ax1.plot(np.arange(1,9),E_optimal_p2_3_s3_100,color='#117a65',label='3 reference stations',marker='o')
    ax1.plot(np.arange(1,7),E_optimal_p2_5_s3_100,color='#d68910',label='5 reference stations',marker='o')
    ax1.plot(np.arange(1,5),E_optimal_p2_7_s3_100,color='#1f618d',label='7 reference stations',marker='o')
    
    ax2.plot(np.arange(1,9),E_optimal_p2_3_s3_1000,color='#117a65',label='3 reference stations',marker='o')
    ax2.plot(np.arange(1,7),E_optimal_p2_5_s3_1000,color='#d68910',label='5 reference stations',marker='o')
    ax2.plot(np.arange(1,5),E_optimal_p2_7_s3_1000,color='#1f618d',label='7 reference stations',marker='o')
    
    ax.set_xticklabels(labels=[])
    ax1.set_xticklabels(labels=[])
    xrange = np.arange(1,9,1)
    ax2.set_xticks(xrange)
    ax2.set_xticklabels(xrange,fontsize=fs)
    
    yrange = [1,10]
    ax.set_yticks(yrange)
    ax.set_yticklabels(['$\sigma^2_{LCS}$','$10\sigma^2_{LCS}$'],fontsize=fs)
    yrange = [1,100]
    ax1.set_yticks(yrange)
    ax1.set_yticklabels(['$\sigma^2_{LCS}$','$10^2 \sigma^2_{LCS}$'],fontsize=fs)
    yrange = [1,1000]
    ax2.set_yticks(yrange)
    ax2.set_yticklabels(['$\sigma^2_{LCS}$','$10^3 \sigma^2_{LCS}$'],fontsize=fs)
    
    ax2.set_xlabel('Number of LCSs',fontsize=fs)
    ax.set_ylabel('$\min\ \lambda_{max} \mathbf{\Sigma_{\hat{a}}}$',fontsize=fs)
    ax1.set_ylabel('$\min\ \lambda_{max} \mathbf{\Sigma_{\hat{a}}}$',fontsize=fs)
    ax2.set_ylabel('$\min\ \lambda_{max} \mathbf{\Sigma_{\hat{a}}}$',fontsize=fs)
    
    ax.set_title('$\sigma^2_{empty} = 10$',fontsize=fs)
    ax1.set_title('$\sigma^2_{empty} = 10^2$',fontsize=fs)
    ax2.set_title('$\sigma^2_{empty} = 10^3$',fontsize=fs)
    ax.legend(fontsize=fs)
    ax1.legend(fontsize=fs)
    ax2.legend(fontsize=fs)
    
    
    plt.tight_layout()
    
    return




#%% Sensor placement reconstruction: true vs predicted
def plot_true_vs_predicted(y_true,y_pred,RefStations,RMSE_stations,locations_not_measured):
    """
    Plot time series of true vs predicted value.
    2 stations are plotted: the lowest and the highest RMSE
    
    y.shape = (num_stations,num_snapshots)
    """
    RMSE = [RMSE_stations[i] for i in locations_not_measured]
    rs = [RefStations[i] for i in locations_not_measured]
    
    rs_min = rs[np.argmin(RMSE)]
    rs_max = rs[np.argmax(RMSE)]
    loc_min = RefStations.index(rs_min)
    loc_max = RefStations.index(rs_max)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    
    ax.plot(y_true[loc_min,:],color='#1A5276',label='True value')
    ax.plot(y_pred[loc_min,:],color='orange',label='Predicted value',alpha=0.5)
    ax.set_title(f'{rs_min} reference station\nRMSE = {RMSE_stations[loc_min]:.2f}',fontsize=15)
    
    ax1.plot(y_true[loc_max,:],color='#0E6655',label='True value')
    ax1.plot(y_pred[loc_max,:],color='orange',label='Predicted value',alpha=0.5)
    ax1.set_title(f'{rs_max} reference station\nRMSE = {RMSE_stations[loc_max]:.2f}',fontsize=15)
    ax.legend()
    ax1.legend()
    ax.grid()
    ax1.grid()
    ax.set_xticks(ticks=[])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    
    plt.tight_layout()
    
    return fig

def scatter_true_vs_predicted(y_true,y_pred,RefStations,RMSE_stations,locations_not_measured):
    """
    Plot scatter plot of true vs predicted value.
    2 stations are plotted: the lowest and the highest RMSE
    
    y.shape = (num_stations,num_snapshots)
    """
    RMSE = [RMSE_stations[i] for i in locations_not_measured]
    rs = [RefStations[i] for i in locations_not_measured]
    
    rs_min = rs[np.argmin(RMSE)]
    rs_max = rs[np.argmax(RMSE)]
    loc_min = RefStations.index(rs_min)
    loc_max = RefStations.index(rs_max)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    
    ax.scatter(y_pred[loc_min,:],y_true[loc_min,:],color='#1A5276',label=f'RMSE = {RMSE_stations[loc_min]:.2f}')
    ax.plot(y_pred[loc_min,:],y_pred[loc_min,:],color='red')
    ax1.scatter(y_pred[loc_max,:],y_true[loc_max,:],color='#0E6655',label=f'RMSE = {RMSE_stations[loc_max]:.2f}')
    ax1.plot(y_pred[loc_max,:],y_pred[loc_max,:],color='red')
    
    ax.set_title(f'{rs_min} reference station',fontsize=15)
    ax1.set_title(f'{rs_max} reference station',fontsize=15)
    
    ax1.set_xlabel('Predicted',fontsize=15)
    ax.set_ylabel('Actual',fontsize=15)
    ax1.set_ylabel('Actual',fontsize=15)
    
    ax.legend()
    ax1.legend()
    ax.grid()
    ax1.grid()
    ax.set_xticks(ticks=[])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    plt.tight_layout()
    
    return fig

def residuals_plot(y_true,y_pred,RefStations,RMSE_stations,locations_not_measured):
    """
    Plot Residuals.
    2 stations are plotted: the lowest and the highest RMSE
    
    y.shape = (num_stations,num_snapshots)
    """
    
    RMSE = [RMSE_stations[i] for i in locations_not_measured]
    rs = [RefStations[i] for i in locations_not_measured]
    
    rs_min = rs[np.argmin(RMSE)]
    rs_max = rs[np.argmax(RMSE)]
    loc_min = RefStations.index(rs_min)
    loc_max = RefStations.index(rs_max)
    
    fs = 20
    fig = plt.figure(figsize=(8,8))
    ax= fig.add_subplot(2,3,(1,2))
    ax1 = fig.add_subplot(2,3,3)
    ax2 = fig.add_subplot(2,3,(4,5))
    ax3 = fig.add_subplot(2,3,6)
    
    # residuals
    ax.scatter(y_pred[loc_min,:],y_true[loc_min,:] - y_pred[loc_min,:],color='#1A5276',label=f'RMSE = {RMSE_stations[loc_min]:.2f}')
    ax.hlines(y=0.0,xmin=0.0,xmax=max(y_pred[loc_min,:]),color='black')
    
    ax2.scatter(y_pred[loc_max,:],y_true[loc_max,:] - y_pred[loc_max,:],color='#0E6655',label=f'RMSE = {RMSE_stations[loc_max]:.2f}')
    ax2.hlines(y=0.0,xmin=0.0,xmax=max(y_pred[loc_max,:]),color='black')
    
    # histogram
    ax1.hist(x = (y_true[loc_min,:] - y_pred[loc_min,:]),bins=30,orientation='horizontal',color='#1A5276')
    ax1.hlines(y=0.0,xmin=0.0,xmax=1500,color='black')
    
    ax3.hist(x = (y_true[loc_max,:] - y_pred[loc_max,:]),bins=30,orientation='horizontal',color='#0E6655')
    ax3.hlines(y=0.0,xmin=0.0,xmax=1500,color='black')
    
    
    ax.set_title(f'{rs_min} reference station',fontsize=fs)
    ax2.set_title(f'{rs_max} reference station',fontsize=fs)
    
    ax.set_xlabel('Predicted',fontsize=fs)
    ax2.set_xlabel('Predicted',fontsize=fs)
    ax1.set_xlabel('Distribution',fontsize=fs)
    ax3.set_xlabel('Distribution',fontsize=fs)
    
    ax.set_ylabel('Residuals',fontsize=fs)
    ax2.set_ylabel('Residuals',fontsize=fs)
    
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    
    ax.legend()
    ax2.legend()
    ax.grid()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    #ax.set_xticks(ticks=[])
    
    plt.tight_layout()
    
    return fig
#%% LCS time series
def TimeSeries_refSt_vs_LCS(measurements_refst,measurements_lcs,kappa,locations,RefStations,loc='Palau-Reial'):
    """
    Plot time series of Reference Station and perturbated LCS measurements.
    Consider using this function for a testing set which has many less points.
    """
    lcs_locations = [RefStations[i] for i in locations[1]]
    if loc not in lcs_locations:
        raise Exception(f'Station selected {loc} is not a location for LCS')
    idx = lcs_locations.index(loc)
    refst = measurements_refst[idx]
    lcs = measurements_lcs[idx]
    
    fs = 15
    time_range = pd.date_range(start='2019-01-01',end='2020-01-01',freq='5MS')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.plot(lcs,color='orange',label='LCS',alpha=0.5)
    ax.plot(refst,color='#1A5276',label='Reference station')
    
    ax.legend(fontsize=fs)
    ax.grid()
    ax.set_title(f'Comparison between reference station\n and LCS measurements at {loc} station\n$\kappa=${kappa[1]}',fontsize=fs)
    ax.set_ylabel('O3 concentration ($\mu$g/$m^3$)',fontsize=fs)
    ax.set_xlabel('ticks',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim(0,max(lcs))
    
    
    plt.tight_layout()
    
    return fig
    


#%% RMSE plots
def RMSE_fullNetwork_Increasing_r():
    RMSE_r_train = [21.92,15.75,12.74,12.25,12.20,11.59,7.06,6.05,4.86,2.77,2.39e-14] # RMSE for the whole network for increasing number of eigenmodes and p=r
    RMSE_r_test = [22.03,14.94,12.22,11.87,12.19,11.61,6.96,5.69,4.80,3.03,2.37e-14]
    fs = 15
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot([i+1 for i in range(len(RMSE_r_train))],RMSE_r_train,'-o',color='#1A5276',label='Training set')
    ax.plot([i+1 for i in range(len(RMSE_r_test))],RMSE_r_test,'-o',color='#B9770E',label='Testing set')
    ax.set_title('Whole network reconstruction error with p=r sensors',fontsize=fs)
    ax.set_xlabel('Number of eigenmodes',fontsize=fs)
    ax.set_ylabel('RMSE ($\mu$g/$m^3$)',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.legend(loc='best')
    ax.grid()

    return fig

def RMSE_fullNetwork_Increasing_LCS():
    """
    RMSE of reconstruction as number of LCS increases for a specific number of nodes
    """
    r=11
    RMSE_p_11 = [2.37e-14,1.7,2.34,4.39,3.21,3.37,5.38,4.68,5.81,6.2,6.22,6.52,]
    fs = 15
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(len(RMSE_p_11))],RMSE_p_11,'-o',color='#1A5276')
    ax.set_title(f'Whole network reconstruction error with p={r} sensors',fontsize=fs)
    ax.set_xlabel('Number of LCSs',fontsize=fs)
    ax.set_ylabel('RMSE ($\mu$g/$m^3$)',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.legend(loc='best')
    ax.grid()

    return fig

#%% 
def main():
    pass

if __name__ == '__main__':
    pass
