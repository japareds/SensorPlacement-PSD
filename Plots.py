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
import warnings
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

def plot_singularvalues(S):
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

#%% Experiments results
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


def plot_ConvexOpt_Locations(results,lambda_val,p1,p2,Psi):
    print(f'Results plot for some value of lambda in {[i for i in results.keys()]}')
    n = Psi.shape[0]
    fs = 10
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(1,len(results[lambda_val][0])+1),results[lambda_val][0],label='LCSs',color='#1a5276')
    ax.bar(np.arange(1,len(results[lambda_val][1])+1),-1*results[lambda_val][1],label='Ref.St.',color='#ca6f1e')
    ax.bar(np.argpartition(np.diag(Psi@Psi.T),-p2)[-p2:]+1,-1*np.ones(Psi.shape[0])[np.argpartition(np.diag(Psi@Psi.T),-p2)[-p2:]],label=f'{p2} highest values of Tr$(\Psi \Psi^T)$',color='#117a65',alpha=0.5)
    xrange = np.arange(0,n+5,5)
    ax.set_xticks(xrange[1:])
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    yrange = np.arange(-1,1.4,0.4)
    ax.set_yticks(yrange)
    ax.set_yticklabels(np.abs(np.round(yrange,decimals=1)),fontsize=fs)
    ax.set_xlabel('i-th entry',fontsize=fs)
    ax.set_ylabel('$h_i$',fontsize=fs)
    ax.legend(fontsize=fs-3,loc='upper right')
    ax.set_title(f'Convex optimization results $\lambda$ = {lambda_val}\n {p1} LCSs and {p2} reference stations',fontsize=fs)
    ax.grid(axis='y')
    fig.tight_layout()
        
    return fig


def plot_ConvexOpt_objFunc_vs_lambdas(D_optimal_function,objective_function,lambdas,p2,objective_function_2=[],p2_=-1):
    """
    Plot optimization objective function results for different regularization values
    """
    fs=10
    figx,figy = 3.5,2.5
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    #ax.plot(lambdas,D_optimal_metric/np.abs(D_optimal_function).max(),marker='o',color='#ca6f1e',label='D-optimal objective function')
    ax.plot(lambdas,objective_function,marker='o',color='#1a5276',label=f'{p2} Ref.St.')
    if p2_>=0:
        ax.plot(lambdas,objective_function_2,marker='o',color='#ca6f1e',label=f'{p2_} Ref.St.')
    ax.set_xticks(lambdas)
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    ax.set_xscale('log')
    ax.set_ylabel('Objective function',fontsize=fs)
#    yrange = np.arange(-1,0.2,0.2)
    #ax.set_yticks(yrange)
    ax.set_yticklabels(np.round(ax.get_yticks(),decimals=3),fontsize=fs)    
    ax.set_xlabel('Regularization parameter $\lambda$',fontsize=fs)
    ax.set_title('Solution of regularized convex relaxation',fontsize=fs)
    ax.grid()
    ax.legend(loc='lower left',fontsize=fs)
    fig.tight_layout()
    
    return fig

def plot_ConvexOpt_objFunc_vs_nRefSt(objective_function,n_refst,lambda_reg,objective_function_2 = [],lambda_reg2=-1):
    """
    Plot optimization objective function results for different number of reference stations
    """
    fs = 10
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot(111)
    ax.plot(n_refst,objective_function,marker='o',color='#ca6f1e',label=f'$\lambda$={lambda_reg}')
    if lambda_reg2>=0:
        ax.plot(n_refst,objective_function_2,marker='o',color='#1a5276',label=f'$\lambda$={lambda_reg2}')
    ax.set_xticks(n_refst)
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    ax.set_xlabel('Number of reference stations')
    ax.set_yticklabels(np.round(ax.get_yticks(),decimals=1),fontsize=fs)
    ax.set_ylabel('Regularized objetctive function',fontsize=fs)
    ax.set_title('Solution of regularized convex relaxation',fontsize=fs)
    ax.legend(loc='lower left',fontsize=fs)
    ax.grid()
    fig.tight_layout()
    
    return fig


def plot_ConvexOpt_limit(results,sigma2,p1,p2,Psi):
    n = Psi.shape[0]
    fs = 15
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(1,len(results[sigma2][0])+1),results[sigma2][0],label='LCSs',color='#1a5276')
    ax.bar(np.arange(1,len(results[sigma2][1])+1),-1*results[sigma2][1],label='Ref.St.',color='#ca6f1e')
    xrange = np.arange(0,n+5,5)
    ax.set_xticks(xrange[1:])
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    yrange = np.arange(-1,1.2,0.2)
    ax.set_yticks(yrange)
    ax.set_yticklabels(np.abs(np.round(yrange,decimals=1)),fontsize=fs)
    ax.set_xlabel('i-th entry',fontsize=fs)
    ax.set_ylabel('$h_i$',fontsize=fs)
    ax.legend(fontsize=fs,loc='upper right')
    exponent = '-'+'{:.0e}'.format(sigma2)[-1]
    ax.set_title(f'Convex optimization results $\sigma_2^2 = 10^{{{exponent}}}\sigma_1^2$\n {p1} LCSs and {p2} reference stations',fontsize=fs)
    ax.grid(axis='y')
    
    return fig
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
class Plots_Projection():
    """
    Collection of all plots used in file Plot_projection.py
    """
    def __init__(self,n,p,r,Psi):
        self.n = n
        self.p = p
        self.r = r
        self.Psi = Psi
        
    def plot_count_non_PSD(self,LMI,x_PSD_t):
        """
        Plot evolution of non-PSD points for different optimization t-values
        
        Parameters
        ----------
        LMI: list of LMI sum for different x values
        x_PSD_t: list of points that are PSD/non-PSD for different objective t-values
        """
        n_points = LMI.shape[0]
        
        fs = 14
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.bar(x=[str(t) for t in x_PSD_t.keys()],height=[100*len(x[0])/n_points for x in x_PSD_t.values()],color='#1a5276',label='PSD')
        ax.bar(x=[str(t) for t in x_PSD_t.keys()],height=[100*len(x[1])/n_points for x in x_PSD_t.values()],bottom=[100*len(x[0])/n_points for x in x_PSD_t.values()],color='orange',label='non-PSD')
        #ax.set_xticks([t for t in x_PSD_t.keys()])
        ax.set_xticklabels([np.round(i,3) for i in x_PSD_t.keys()],fontsize=fs,rotation=45)
        ax.set_yticks(np.arange(0.,120,20))
        ax.set_yticklabels([np.int(i) for i in ax.get_yticks()],fontsize=fs)
        ax.set_xlabel('Optimization t value',fontsize=fs)
        ax.set_ylabel('Percentage of elemenets',fontsize=fs)
        ax.legend(fontsize=fs)
        ax.set_title(f'Number of PSD and non-PSD elements\nr/p = {self.r/self.p}')
        fig.tight_layout()
        
        return fig

    def get_projections(self,LMI,LMI_q,projection,n,p):
        if projection == 'x-linear': #plot C entries for sensor-p
            points = np.array([[M[3,0],M[3,1],M[3,2]] for M in LMI])
            
            # entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{12}$'
            entry3 = '$x_{13}$'
        
        elif projection == 'x1-squared':# plot entry 1 of C (squared) vs the others
            points = np.array([[M[3,1],M[3,2],MS[3,0]] for M,MS in zip(LMI,LMI_q)])
            
            # entry names for plot
            entry1 = '$x_{12}$'
            entry2 = '$x_{13}$'
            entry3 = 'x_{11}^2'
            
        elif projection == 'x2-squared':#plot entry 2 of C (squared) vs the others
            points = np.array([[M[3,0],M[3,2],MS[3,1]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{13}$'
            entry3 = '$x_{12}^2$'
            
        elif projection == 'x3-squared':#plot entry 3 of C(squared) vs the others
            points = np.array([[M[3,0],M[3,1],MS[3,2]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{12}$'
            entry3 = '$x_{13}^2$'

        elif projection == 'x-squared':# plot entries of C squared
            points = np.array([[M[3,0],M[3,1],M[3,2]] for M in LMI_q])
            
            #entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{12}$'
            entry3 = '$x_{13}$'
        
        elif projection == 'x1-linear':#plot entry 1 of C linear and the others squared
            points = np.array([[MS[3,1],MS[3,2],M[3,0]] for MS,M in zip(LMI_q,LMI)])
            
            #entry names for plot
            entry1 = '$x_{12}^2$'
            entry2 = '$x_{13}^2$'
            entry3 = '$x_{11}$'
        
        elif projection == 'x2-linear':#plot entry2 of C linear and the others squared
            points =  np.array([[MS[3,0],MS[3,2],M[3,1]] for MS,M in zip(LMI_q,LMI)])
            
            # entry names for plot
            entry1 = '$x_{11}^2$'
            entry2 = '$x_{13}^2$'
            entry3 = '$x_{12}$'
        
        elif projection == 'x3-linear':#plot entry3 of C linear and the others squared
            points = np.array([[MS[3,0],MS[3,1],M[3,2]] for MS,M in zip(LMI_q,LMI)])
            
            #entry names for plot
            entry1 = '$x_{11}^2$'
            entry2 = '$x_{12}^2$'
            entry3 = '$x_{13}$'
            
        elif projection == 'diag':# plot diagonal elements
            points = np.array([np.diag(LMI[i])[0:n] for i in range(LMI.shape[0])])
            
            # points_q = np.array([np.diag(LMI_q[i])[0:n] for i in range(LMI.shape[0])])
            # points_opt = np.array([np.diag(H) for H in H_t.values()])
            # points_non_PSD = np.array([C.sum(axis=0) for C in x_non_PSD])
            # points_non_PSD_q = np.array([(C**2).sum(axis=0) for C in x_non_PSD])
            
            # entry names for plot
            entry1 = r'$\Sigma$ $x_{1i}$'
            entry2 = r'$\Sigma$ $x_{2i}$'
            entry3 = r'$\Sigma$ $x_{3i}$'
        
        elif projection == 'diag-squared':#plot diagonal elements squared
            points = np.array([np.diag(M)[:n] for M in LMI_q])
            
            # entry names for plot
            entry1 = '$\Sigma$$x_{1i}^2$'
            entry2 = '$\Sigma$$x_{2i}^2$ '
            entry3 = '$\Sigma$$x_{3i}^2$'
            
        elif projection == 'diag1-squared':#plot first diagonal entry squred vs other entries linear
            points = np.array([[M[1,1],M[2,2],MS[0,0]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$h_{22}$'
            entry2 = '$h_{33}$'
            entry3 = '$h_{11} squared$'
            
        elif projection == 'diag2-squared':#plot second diagonal entry squared vs other entries linear
            points = np.array([[M[0,0],M[2,2],MS[1,1]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$h_{11}$'
            entry2 = '$h_{33}$'
            entry3 = '$h_{22}$ squared'
        
        elif projection == 'diag3-squared':#plot third diagonal entry squared vs other entries linear
            points = np.array([[M[0,0],M[1,1],MS[2,2]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$h_{11}$'
            entry2 = '$h_{22}$'
            entry3 = '$h_{33} squared$'
        
        elif projection == 'diag1-mixed':#plot diagonal entry 1 vs other C values
            points = np.array([ [M[3,1],M[3,2],M[0,0]] for M in LMI])
            
            # entry names for plot
            entry1 = '$x_{12}$'
            entry2 = '$x_{13}$'
            entry3 = '$\Sigma$$x_{1i}$'
        
        elif projection == 'diag2-mixed':#plot diagonal entry2 vs other C values
            points = np.array([[M[3,0],M[3,2],M[1,1]]for M in LMI])
            
            # entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{13}$'
            entry3 = '$\Sigma$$x_{2i}$'
            
        elif projection == 'diag3-mixed':#plot diagonal entry 3 vs other C values
            points = np.array([[M[3,0],M[3,1],M[2,2]]for M in LMI])
            
            # entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{12}$'
            entry3 = '$\Sigma$$x_{3i}$'
            
        elif projection == 'diag1-mixed-squared':#plot diagonal 1 squared vs other C values linear
            points = np.array([[M[3,1],M[3,2],MS[0,0]] for M,MS in zip(LMI,LMI_q)])
            
            # entry names for plot
            entry1 = '$x_{12}$'
            entry2 = '$x_{13}$'
            entry3 = '$\Sigma$$x_{1i}^2$'
            
        elif projection == 'diag2-mixed-squared':#plot diagonal 2 squared vs other C values linear
            points = np.array([[M[3,0],M[3,2],MS[1,1]] for M,MS in zip(LMI,LMI_q)])
            
            #entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{13}$'
            entry3 = '$\Sigma$$x_{2i}^2$'
            
        elif projection == 'diag3-mixed-squared':#plot diagonal 3 squared vs other C values linear
            points = np.array([[M[3,0],M[3,1],MS[2,2]] for M,MS in zip(LMI,LMI_q)])
            
            # entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{12}$'
            entry3 = '$\Sigma$$x_{3i}^2$'
        
        elif projection == 'diag12-mixed':#plot diagonal 1 and 2 vs other C values linear
            points = np.array([[M[0,0],M[1,1],M[3,2]]for M in LMI])
            #entry names for plot
            entry1 = r'$\Sigma$$x_{1i}$'
            entry2 = r'$\Sigma$$x_{2i}$'
            entry3 = r'$x_{13}$'
            
        elif projection == 'diag13-mixed':#plot diagonal 1 and 3 vs other C values linear
            points = np.array([[M[0,0],M[2,2],M[3,1]]for M in LMI])
            
            #entry names for plot
            entry1 = r'$\Sigma$$x_{1i}$'
            entry2 = r'$\Sigma$$x_{3i}$'
            entry3 = '$x_{12}$'
            
        elif projection == 'diag23-mixed':#plot diagonal 2 and 3 vs other C values linear
            points = np.array([[M[1,1],M[2,2],M[3,0]] for M in LMI])
            
            #entry names for plot
            entry1 = r'$\Sigma$$x_{2i}$'
            entry2 = r'$\Sigma$$x_{3i}$'
            entry3 = '$x_{11}$'
            
        elif projection == 'diag12-mixed-squared':#plot diagonal 1 and 2 squared vs other C values linear
            points = np.array([[MS[0,0],MS[1,1],M[3,2]] for MS,M in zip(LMI_q,LMI)])
            
            # entry names for plot
            entry1 = '$\Sigma$$x_{1i}^2$'
            entry2 = '$\Sigma$$x_{2i}^2$'
            entry3 = '$x_{13}$'
            
        elif projection == 'diag13-mixed-squared':#plot diagonal 1 and 3 squared vs other C values linear
            points = np.array([[MS[0,0],MS[2,2],M[3,1]] for MS,M in zip(LMI_q,LMI)])
            
            #entry names for plot
            entry1 = '$\Sigma$$x_{1i}^2$'
            entry2 = '$\Sigma$$x_{3i}^2$'
            entry3 = '$x_{12}$'
            
        elif projection == 'diag23-mixed-squared':#plot diagonal 2 and 3 squared vs other C values linear
            points = np.array([[MS[1,1],MS[2,2],M[3,0]] for MS,M in zip(LMI_q,LMI)])
            
            # entry names for plot
            entry1 = '$\Sigma$$x_{2i}^2$'
            entry2 = '$\Sigma$$x_{3i}^2$'
            entry3 = '$x_{11}$'
            
        elif projection == 'n1':# get points at location 1
            points = np.array([np.concatenate([LMI[i][-p:,0],[LMI[i][0,0]]]) for i in range(LMI.shape[0])])
            # points_q = np.array([np.concatenate([LMI[i][-p:,0],[LMI_q[i][0,0]]]) for i in range(LMI.shape[0])])
            # points_opt = np.array([[C[0,0],C[1,0],np.diag(H)[0]] for H,C in zip(H_t.values(),C_t.values())])
            # points_non_PSD = np.array([[C[0,0],C[1,0],C.sum(axis=0)[0]]for C in x_non_PSD])
            # points_PSD = np.array([[C[0,0],C[1,0],C.sum(axis=0)[0]]for C in x_PSD])
            
            # points_non_PSD_q = np.array([[C[0,0],C[1,0],C_q.sum(axis=0)[0]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
            
            
            # entry names for plot
            entry1 = '$x_{11}$'
            entry2 = '$x_{21}$'
            entry3 = '$h_{11}$'
            
            
        elif projection == 'n2':# get points at location 2
            points = np.array([np.concatenate([LMI[i][-p:,1],[LMI[i][1,1]]]) for i in range(LMI.shape[0])])
            # points_q = np.array([np.concatenate([LMI[i][-p:,1],[LMI_q[i][1,1]]]) for i in range(LMI.shape[0])])
            # points_opt = np.array([[C[0,1],C[1,1],np.diag(H)[1]] for H,C in zip(H_t.values(),C_t.values())])
            # points_non_PSD = np.array([[C[0,1],C[1,1],C.sum(axis=0)[1]]for C in x_non_PSD])
            # points_non_PSD_q = np.array([[C[0,1],C[1,1],C_q.sum(axis=0)[1]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
            # # entry names for plot
            entry1 = '$x_{12}$'
            entry2 = '$x_{22}$'
            entry3 = '$h_{22}$'
        
        elif projection=='n3':# get points at location 3
            points = np.array([np.concatenate([LMI[i][-p:,2],[LMI[i][2,2]]]) for i in range(LMI.shape[0])])
            # points_q = np.array([np.concatenate([LMI[i][-p:,2],[LMI_q[i][2,2]]]) for i in range(LMI.shape[0])])
            # points_opt = np.array([[C[0,2],C[1,2],np.diag(H)[2]] for H,C in zip(H_t.values(),C_t.values())])
            # points_non_PSD = np.array([[C[0,2],C[1,2],C.sum(axis=0)[2]]for C in x_non_PSD])
            # points_non_PSD_q = np.array([[C[0,2],C[1,2],C_q.sum(axis=0)[2]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
            # # entry names for plot
            entry1 = '$x_{13}$'
            entry2 = '$x_{23}$'
            entry3 = '$h_{33}$'
        
        return points,entry1,entry2,entry3
    
    def get_optimization_path(self,x_PSD_t,C_t,H_t,projection):
        """
        get coordinates of solutions at different optimization t values
        """
        # linear projections
        if projection == 'x-linear':
            points_opt = np.array([C[0] for C in C_t.values()])
        elif projection == 'diag':
            points_opt = np.array([np.diag(H) for H in H_t.values()])
        elif projection == 'diag1-mixed':
            points_opt = np.array([[C[0,1],C[0,2],H[0,0]] for C,H in zip(C_t.values(),H_t.values())])
        elif projection == 'diag2-mixed':
            points_opt = np.array([[C[0,0],C[0,2],H[1,1]] for C,H in zip(C_t.values(),H_t.values())])
        elif projection == 'diag3-mixed':
            points_opt = np.array([[C[0,0],C[0,1],H[2,2]] for C,H in zip(C_t.values(),H_t.values())])
        elif projection == 'diag12-mixed':
            points_opt = np.array([[H[0,0],H[1,1],C[0,2]] for C,H in zip(C_t.values(),H_t.values())])
        elif projection == 'diag13-mixed':
            points_opt = np.array([[H[0,0],H[2,2],C[0,1]] for C,H in zip(C_t.values(),H_t.values())])
        elif projection == 'diag23-mixed':
            points_opt = np.array([[H[1,1],H[2,2],C[0,0]] for C,H in zip(C_t.values(),H_t.values())])
        # squared projections
        elif projection == 'diag-squared':
            points_opt = np.array([(C**2).sum(axis=0) for C in C_t.values()])
        elif projection == 'diag1-mixed-squared':
            points_opt = np.array([[C[0,1],C[0,2],(C**2).sum(axis=0)[0]] for C in C_t.values()])
        elif projection == 'diag2-mixed-squared':
            points_opt = np.array([[C[0,0],C[0,2],(C**2).sum(axis=0)[1]] for C in C_t.values()])
        elif projection == 'diag3-mixed-squared':
            points_opt = np.array([[C[0,0],C[0,1],(C**2).sum(axis=0)[2]] for C in C_t.values()])
        elif projection == 'diag12-mixed-squared':
            points_opt = np.array([[(C**2).sum(axis=0)[0],(C**2).sum(axis=0)[1],C[0,2]] for C in C_t.values()])
        elif projection == 'diag13-mixed-squared':
            points_opt = np.array([[(C**2).sum(axis=0)[0],(C**2).sum(axis=0)[2],C[0,1]] for C in C_t.values()])
        elif projection == 'diag23-mixed-squared':
            points_opt = np.array([[(C**2).sum(axis=0)[1],(C**2).sum(axis=0)[2],C[0,0]] for C in C_t.values()])
            
        else:
            warnings.warn(f'No optimization path for projection 0 {projection}')
            points_opt = []
        return points_opt
    
    def get_non_PSD_points(self,x_PSD_t,t,projection,n,p):
        """
        return non-PSD points from the list
        """
        x_non_PSD = np.array(x_PSD_t[t][1])
        
        if x_non_PSD.shape[0] == 0:
            return x_non_PSD
        else:
            x_non_PSD = np.reshape(x_non_PSD,(x_non_PSD.shape[0],p,n))
        
        
        
        if projection == 'x-linear':
            points_non_PSD = np.array([C[0,:] for C in x_non_PSD])
            
        elif projection == 'diag':
            points_non_PSD = np.array([C.sum(axis=0) for C in x_non_PSD])
            
        elif projection == 'diag1-mixed':
            points_non_PSD = np.array([[C[0,1],C[0,2],C.sum(axis=0)[0]] for C in x_non_PSD])
        
        elif projection == 'diag2-mixed':
            points_non_PSD = np.array([[C[0,0],C[0,2],C.sum(axis=0)[1]] for C in x_non_PSD])
        
        elif projection == 'diag3-mixed':
            points_non_PSD = np.array([[C[0,0],C[0,1],C.sum(axis=0)[2]] for C in x_non_PSD])
            
        elif projection == 'diag12-mixed':
            points_non_PSD = np.array([[C.sum(axis=0)[0],C.sum(axis=0)[1],C[0,2]] for C in x_non_PSD])
            
        elif projection == 'diag13-mixed':
            points_non_PSD = np.array([[C.sum(axis=0)[0],C.sum(axis=0)[2],C[0,1]] for C in x_non_PSD])
            
        elif projection == 'diag23-mixed':
            points_non_PSD = np.array([[C.sum(axis=0)[1],C.sum(axis=0)[2],C[0,0]] for C in x_non_PSD])
        # squared projections
        elif projection == 'diag-squared':
            points_non_PSD = np.array([(C**2).sum(axis=0) for C in x_non_PSD])
        
        elif projection == 'diag1-mixed-squared':
            points_non_PSD = np.array([[C[0,1],C[0,2],(C**2).sum(axis=0)[0]] for C in x_non_PSD])
            
        elif projection == 'diag2-mixed-squared':
            points_non_PSD = np.array([[C[0,0],C[0,2],(C**2).sum(axis=0)[1]] for C in x_non_PSD])
            
        elif projection == 'diag3-mixed-squared':
            points_non_PSD = np.array([[C[0,0],C[0,1],(C**2).sum(axis=0)[2]] for C in x_non_PSD])
        
        elif projection == 'diag12-mixed-squared':
            points_non_PSD = np.array([[(C**2).sum(axis=0)[0],(C**2).sum(axis=0)[1],C[0,2]] for C in x_non_PSD])
            
        elif projection == 'diag13-mixed-squared':
            points_non_PSD = np.array([[(C**2).sum(axis=0)[0],(C**2).sum(axis=0)[2],C[0,1]] for C in x_non_PSD])
            
        elif projection == 'diag23-mixed-squared':
            points_non_PSD = np.array([[(C**2).sum(axis=0)[1],(C**2).sum(axis=0)[2],C[0,0]] for C in x_non_PSD])
            
        return points_non_PSD
    
    def plot_projection(self,LMI,LMI_q,H_t,C_t,x_PSD_t,t,projection='diag'):
        """
        Plot projection figure
        Includes:
            - surface
            - optimization path with final result
            - non-PSD points in the mesh
        """
        p = self.p
        n = self.n
        
        if p!= 2 or n!=3:
            warnings.warn(f'The projections are reliable for p=2 and n=3\nFound p={p} and n={n}')
        
        # non-PSD region
        x_non_PSD = np.array(x_PSD_t[t][1])# ==1 for non-PSD & 0 for PSD
        x_non_PSD = x_non_PSD.reshape((x_non_PSD.shape[0],p,n))
        x_PSD = np.array(x_PSD_t[t][0])# ==1 for non-PSD & 0 for PSD
        x_PSD = x_PSD.reshape((x_PSD.shape[0],p,n))
        print(f'At t={t} there are {x_non_PSD.shape[0]} non PSD elements out of {LMI.shape[0]}')
        
        # get projection points
        points,entry1,entry2,entry3 = self.get_projections(LMI,LMI_q, projection,n,p)
        points_opt = self.get_optimization_path(x_PSD_t, C_t, H_t, projection)
        points_opt = points_opt[np.argwhere([i for i in C_t.keys()] == np.tile(t,len(C_t)))[0,0]:,:]
        points_non_PSD = self.get_non_PSD_points(x_PSD_t,t,projection,n,p)
        
        fs=14
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111,projection='3d')
        # plot mesh
        ax.plot_trisurf(points[:,0],points[:,1],points[:,2],color='orange',linewidth=100,alpha=0.5)
        ax.scatter3D(points[:,0],points[:,1],points[:,2],color='orange',marker='o',s=30,label='mesh')
        try:
            # plot optimization path and result at final step
            ax.plot3D(points_opt[:,0],points_opt[:,1],points_opt[:,2],color='#b03a2e',marker='o',markersize=10,label='optimization path')
            ax.scatter3D(points_opt[0,0],points_opt[0,1],points_opt[0,2],color='#1abc9c',marker='*',s=30,label=f'optimization result t={t}')
        except:
            print('No optimization path plotted')
        
        try:
            ax.scatter3D(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k',marker='x',s=30,label='non-PSD points')
            ax.plot_trisurf(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k',linewidth=100,alpha=0.8)
        except:
            print('No non-PSD points detected')
        #optimization results
        #ax.plot3D(points_opt[:,0],points_opt[:,1],points_opt[:,2],color='#b03a2e',marker='o',markersize=10,label='optimization path')
        #ax.scatter3D(points_opt[0,0],points_opt[0,1],points_opt[0,2],color='#1abc9c',marker='*',s=30,label='optimization result')
        #surfaces
        #ax.plot_trisurf(points_q[:,0],points_q[:,1],points_q[:,2],color='#1a5276',linewidth=10)
        
        #ax.scatter3D(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k',s=20,label='non-PSD region')
        
        ax.set_xlabel(entry1,fontsize=fs)
        ax.set_ylabel(entry2,fontsize=fs)
        ax.set_zlabel(entry3,fontsize=fs)
        
        ax.set_xticks([np.round(i,1) for i in np.arange(0.,1.1,0.1)])
        ax.set_yticks([np.round(i,1) for i in np.arange(0.,1.1,0.1)])
        ax.set_zticks([np.round(i,1) for i in np.arange(0.,1.1,0.1)])
        ax.set_title(f'projection: {projection}')
        ax.legend(fontsize=fs-5)
        # try:#non-PSD points
        #     ax.plot_trisurf(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k')
        #     ax.plot_trisurf(points_non_PSD_q[:,0],points_non_PSD_q[:,1],points_non_PSD_q[:,2],color='k')
        #     ax.plot_trisurf(points_PSD[:,0],points_PSD[:,1],points_PSD[:,2],color='w')
        #     fig.tight_layout()
        # except:
        fig.tight_layout()
        
      
        # # second figure: scatter 2D
        # fig1 = plt.figure(figsize=(3.5,2.5))
        # ax = fig1.add_subplot(111)
        # ax.scatter(points_PSD[:,0],points_PSD[:,1],label='PSD points',color='#1a5276',marker='o')
        # ax.scatter(points_non_PSD[:,0],points_non_PSD[:,1],label='non-PSD points',color='orange',marker='.')
        # ax.set_xticks([np.round(i,1) for i in np.arange(0.,1.2,0.2)])
        # ax.set_yticks([np.round(i,1) for i in np.arange(0.,1.2,0.2)])
        # ax.set_xlabel(entry1,fontsize=fs)
        # ax.set_ylabel(entry2,fontsize=fs)
        # ax.set_title('2D projection')
        # ax.legend(loc='upper right')
        
        
        return fig
            
    def plot_projection_Schur(self,x_PSD_t,B_t,C_t,H_t,t):
        """
        Plot projection of the mapping to the PSD matrix cone which is the schur complement used 
        for evaluating if the points are PSD or not
        """
        p,n,r = self.p,self.n,self.r
        Psi = self.Psi
        
        x_PSD = np.array(x_PSD_t[t][0])
        x_non_PSD = np.array(x_PSD_t[t][1])
        x_PSD = x_PSD.reshape(x_PSD.shape[0],p,n)
        x_non_PSD = x_non_PSD.reshape(x_non_PSD.shape[0],p,n)
        
        # create LMIs
        LMI_sparse_PSD = [np.block([[np.diag(C.sum(axis=0)),C.T],[C,np.zeros((p,p))]]) for C in x_PSD]
        LMI_sparse_non_PSD = [np.block([[np.diag(C.sum(axis=0)),C.T],[C,np.zeros((p,p))]]) for C in x_non_PSD]
        LMI_sparse_solution = np.block([[H_t[t],C_t[t].T],[C_t[t],np.zeros((p,p))]])
        
        Ip = np.identity(p)
        R_p = [np.block([[Psi,np.zeros((n,1))],[np.zeros((p,r)),Ip[:,j][:,None]]]) for j in range(p)]
        # t-valued
        B = B_t[t]
        # mapping
        points_PSD = []
        points_non_PSD = []
        points_solution = []
        for R in R_p:
            # compute the p Psi-mapped LMIs
            # PSD mesh
            # solution points
            LMI_mapped = R.T@LMI_sparse_solution@R + B 
            points_solution_p = np.array([LMI_mapped[0,0],LMI_mapped[r,0],LMI_mapped[0,r]])
            #points_solution_p = points_solution_p.reshape(1,points_solution_p.shape[1])
            points_solution.append(points_solution_p)
            
            LMIs = [R.T@M@R + B for M in LMI_sparse_PSD]
            if len(LMIs) == 0:
                pass
            else:
                points_PSD_p = np.array([[M[0,0],M[r,0],M[0,r]] for M in LMIs])
                #points_PSD_p = points_PSD_p.reshape(points_PSD_p.shape[0],points_PSD_p[0].shape[0])
                points_PSD.append(points_PSD_p)
            
            # non-PSD mesh
            LMIs = [R.T@M@R + B for M in LMI_sparse_non_PSD]
            if len(LMIs) ==0:
                pass
            else:
                points_non_PSD_p = np.array([[M[0,0],M[r,0],M[0,r]] for M in LMIs])
                #points_non_PSD_p = points_non_PSD_p.reshape(points_non_PSD_p.shape[0],points_non_PSD_p[0].shape[0])
                points_non_PSD.append(points_non_PSD_p)
            
         
            
            
        points_PSD = np.array(points_PSD)
        points_non_PSD = np.array(points_non_PSD)
        points_solution = np.array(points_solution)
        
        
        # plot mapping
        fs=10
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111,projection='3d')
        marker_psd = {0:'o',1:'x'}
        marker_non_psd = {0:'o',1:'x'}
        
        for j in range(p):
            ax.scatter3D(points_solution[j,0],points_solution[j,1],points_solution[j,2],label='Optimization result',color='#b03a2e',marker=marker_psd[j])
            if points_PSD.shape[0] != 0:
                ax.scatter3D(points_PSD[j,:,0],points_PSD[j,:,1],points_PSD[j,:,2],label=f'PSD points LMI-{j}',color='orange',marker=marker_psd[j])
                
            else:
                print(f'No PSD points for t={t}')
            if points_non_PSD.shape[0] != 0:
                ax.scatter3D(points_non_PSD[j,:,0],points_non_PSD[j,:,1],points_non_PSD[j,:,2],label=f'non-PSD points LMI-{j}',color='k',marker=marker_non_psd[j])
            else:
                print(f'All points are PSD for t={t}')
                    
            
        ax.legend(fontsize=fs-3)
        ax.set_xlabel(r'$\Sigma_{\beta_{11}}$',fontsize=fs)
        ax.set_ylabel(r'C$\Psi$',fontsize=fs)
        ax.set_zlabel(r'$C\Psi^T$',fontsize=fs)
        ax.set_title(f'Mapped mesh points at t={t}',fontsize=fs)
        
        ax.grid()
        fig.tight_layout()
        
        
        
        return fig
    
    
    

#%% plots Experiment OLS LMI
class Plots_Experiment_OLS_LMI():
    """
    Collection of all plots used in file Experiment_OLS_LMI.py
    
    """
    
    def __init__(self,residuals_cov,residuals_cov_random,optimal_locations,random_locations,n,p,r):
            self.residuals_cov = residuals_cov
            self.residuals_cov_random = residuals_cov_random
            self.optimal_locations = optimal_locations
            self.random_locations = random_locations
            self.p  = p
            self.n = n
            self.r = r
            
    def plot_covariances_trace(residuals_cov,residuals_cov_random,optimal_locations,random_locations):
        """
        Plot residuals covariance matrix obtained via convex relaxation SDP
        and compares them with min/max random placement
        """
        # min/max traces of random placement
        traces_random= []
        for cov_mat in residuals_cov_random:
            traces_random.append(np.trace(cov_mat))
        diag_min = [np.diag(residuals_cov_random[i]).min() for i in range(len(residuals_cov_random))]
        diag_max = [np.diag(residuals_cov_random[i]).max() for i in range(len(residuals_cov_random))]
        
        idx_min = np.argmin(traces_random)#np.argmin(traces_random)
        idx_max = np.argmax(traces_random)#np.argmax(traces_random)
        
        trace_random_min = traces_random[idx_min]
        trace_random_max = traces_random[idx_max]
        trace_optimal = np.trace(residuals_cov)
        
        
        
        # plot minimum and maximum covariances
        
        figx,figy = 3.5,2.5
        fs = 10
        
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        #ax.imshow(np.ma.masked_where(residuals_cov==np.diag(residuals_cov),residuals_cov),vmin = residuals_cov.min(), vmax = residuals_cov.max(),cmap='Oranges',alpha=1)
        im = ax.imshow(residuals_cov,vmin = residuals_cov.min(), vmax = residuals_cov.max(),cmap='Oranges',alpha=1.)    
        ax.set_title(f'Residuals covariance matrix: SDP\n Tr = {trace_optimal:.2f}, max = {np.diag(residuals_cov).max():.2f}, min = {np.diag(residuals_cov).min():.2f}',fontsize=fs)
        loc = np.sort(np.concatenate([optimal_locations[0],optimal_locations[1]]))
        ax.set_xticks(loc)
        ax.set_yticks(loc)
        ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
        ax.set_yticklabels(ax.get_xticks(),fontsize=fs)
        # identify LCSs
        for t in ax.get_xticks():
            if t in optimal_locations[0]:# change only LCSs locations
                idx = np.argwhere(t == ax.get_xticks())[0,0]    
                ax.get_xticklabels()[idx].set_color('red')
                ax.get_yticklabels()[idx].set_color('red')
        # color bar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        
        fig.tight_layout()
        
        fig1 = plt.figure(figsize=(figx,figy))
        
        residuals_cov_random_min = residuals_cov_random[idx_min]
        random_locations_min = random_locations[idx_min]
        
        ax1 = fig1.add_subplot(111)
        im1 = ax1.imshow(residuals_cov_random_min,vmin = residuals_cov_random_min.min(), vmax = residuals_cov_random_min.max(),cmap='Oranges')
        ax1.set_title(f'Residuals covariance matrix: Best random\n Tr = {trace_random_min:.2f}, max = {np.diag(residuals_cov_random_min).max():.2f}, min = {np.diag(residuals_cov_random_min).min():.2f}',fontsize=fs)
        loc = np.sort(np.concatenate([optimal_locations[0],optimal_locations[1]]))
        ax1.set_xticks(loc)
        ax1.set_yticks(loc)
        ax1.set_xticklabels(ax1.get_xticks(),fontsize=fs)
        ax1.set_yticklabels(ax1.get_xticks(),fontsize=fs)
        
        for t in ax1.get_xticks():
            if t in random_locations_min[0]:
                idx = np.argwhere(t == ax1.get_xticks())[0,0]
                ax1.get_xticklabels()[idx].set_color('red')
                ax1.get_yticklabels()[idx].set_color('red')
        # color bar
        fig1.subplots_adjust(right=0.8)
        cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig1.colorbar(im1, cax=cbar_ax)
        
        fig1.tight_layout()
        
        fig2 = plt.figure(figsize=(figx,figy))
        
        residuals_cov_random_max = residuals_cov_random[np.argmax(traces_random)]
        random_locations_max = random_locations[idx_max]
        
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(residuals_cov_random_max,vmin = residuals_cov_random_max.min(), vmax = residuals_cov_random_max.max(),cmap='Oranges')
        ax2.set_title(f'Residuals covariance matrix: Worst random\n Tr = {trace_random_max:.2f}, max = {np.diag(residuals_cov_random_max).max():.2f}, min = {np.diag(residuals_cov_random_max).min():.2f}',fontsize=fs)
        loc = np.sort(np.concatenate([optimal_locations[0],optimal_locations[1]]))
        ax2.set_xticks(loc)
        ax2.set_yticks(loc)
        ax2.set_xticklabels(ax2.get_xticks(),fontsize=fs)
        ax2.set_yticklabels(ax2.get_xticks(),fontsize=fs)
        
        
        for t in ax2.get_xticks():
            if t in random_locations_max[0]:
                idx = np.argwhere(t == ax2.get_xticks())[0,0]
                ax2.get_xticklabels()[idx].set_color('red')
                ax2.get_yticklabels()[idx].set_color('red')
                
        # color bar
        fig2.subplots_adjust(right=0.8)
        cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig2.colorbar(im2, cax=cbar_ax)
        
        fig2.tight_layout()
        
        # report dict
        results_dict = {}
        ## SDP placement
        results_dict['SDP_placement_trace'] = trace_optimal
        results_dict['SDP_placement_max_diag'] = np.diag(residuals_cov).max()
        results_dict['SDP_placement_min_diag'] = np.diag(residuals_cov).min()
        results_dict['SDP_placement_sensor_max'] = np.diag(residuals_cov)[optimal_locations[0]].max()
        results_dict['SDP_placement_sensor_min'] = np.diag(residuals_cov)[optimal_locations[0]].min()
        results_dict['SDP_placement_reconstruction_max'] = np.diag(residuals_cov)[optimal_locations[1]].max()
        results_dict['SDP_placement_reconstruction_min'] = np.diag(residuals_cov)[optimal_locations[1]].min()
        ## random placement
        results_dict['Random_placement_min_trace'] = trace_random_min
        results_dict['Random_placement_min_max_diag'] = np.diag(residuals_cov_random_min).max()
        results_dict['Random_placement_min_min_diag'] = np.diag(residuals_cov_random_min).min()
        results_dict['Random_placement_min_sensor_max'] = np.diag(residuals_cov_random_min)[random_locations_min[0]].max()
        results_dict['Random_placement_min_sensor_min'] = np.diag(residuals_cov_random_min)[random_locations_min[0]].min()
        results_dict['Random_placement_min_reconstruction_max'] = np.diag(residuals_cov_random_min)[random_locations_min[1]].max()
        results_dict['Random_placement_min_reconstruction_min'] = np.diag(residuals_cov_random_min)[random_locations_min[1]].min()
        
        results_dict['Random_placement_max_trace'] = trace_random_max
        results_dict['Random_placement_max_max_diag'] = np.diag(residuals_cov_random_max).max()
        results_dict['Random_placement_max_min_diag'] = np.diag(residuals_cov_random_max).min()
        results_dict['Random_placement_max_sensor_max'] = np.diag(residuals_cov_random_max)[random_locations_max[0]].max()
        results_dict['Random_placement_max_sensor_min'] = np.diag(residuals_cov_random_max)[random_locations_max[0]].min()
        results_dict['Random_placement_max_reconstruction_max'] = np.diag(residuals_cov_random_max)[random_locations_max[1]].max()
        results_dict['Random_placement_max_reconstruction_min'] = np.diag(residuals_cov_random_max)[random_locations_max[1]].min()
        
        
        
        
        
            
        return (fig,fig1,fig2), results_dict
    
    def plot_covariances_min(self):
        """
        Plot covariance matrices: SDP and random placements
        highlightning the minimum value of reconstruction
        """
        residuals_cov = self.residuals_cov
        residuals_cov_random=self.residuals_cov_random
        optimal_locations = self.optimal_locations
        random_locations = self.random_locations
        p = self.p
        r = self.r
        
        # obtain min/max covariance at measurement locations
        random_covariance = {el:0 for el in random_locations.keys()}
        for cov_mat,loc in zip(residuals_cov_random,random_locations.keys()):
            loc_sensors = random_locations[loc][0]
            sensors_cov = np.diag(cov_mat)
            random_covariance[loc] = [sensors_cov.min(),sensors_cov.max()]
        
        random_covariance_max = np.array([random_covariance[i][1] for i in random_covariance.keys()])
        
        loc_sensors = optimal_locations[0]
        sdp_covariance = np.diag(residuals_cov)
            
        # plot
        figx,figy = 3.5,2.5
        fs = 10
            
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        ax.plot(random_covariance_max,color='#1a5276',label='Max covariance random',alpha=0.9)
        ax.hlines(y=sdp_covariance.max(),xmin=-1,xmax=len(random_covariance_max)+1,linestyle='--',label='SDP solution',color='k',linewidth=2.)
        ax.hlines(y=r/p,xmin=-1,xmax=len(random_covariance_max)+1,linestyle='--',label='r/p',color='orange',linewidth=2.)
        ax.set_xticks(np.arange(len(random_covariance_max),10)) 
        ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
        ax.set_yticks(np.arange(0.6,1.2,0.2))
        ax.set_yticklabels(np.round(ax.get_yticks(),2),fontsize=fs)
        ax.set_xlabel('Random iteration',fontsize=fs)
        ax.set_ylabel('Max covariance',fontsize=fs)
        ax.legend(loc='lower right')
        fig.tight_layout()
        
        return fig
    
    def plot_locations_optimal_vs_random(self):
        """
        Plot comparison of PSD optimal locations vs random locations.
        How many stations differ between both approaches
        """
        optimal_locations = self.optimal_locations
        random_locations = self.random_locations
        p = self.p
        n = self.n
        number_shared_locations = [np.sum(np.in1d(loc[0],optimal_locations[0]))/p for loc in random_locations.values()]
        
        fs=10
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.plot(number_shared_locations,color='#1a5276')
        ax.set_xticks(np.arange(0,len(number_shared_locations),len(number_shared_locations)/10))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()],fontsize=fs)
        ax.set_yticks(np.arange(0,1.2,0.2))
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()],fontsize=fs)
        ax.set_ylabel('Fraction of shared locations',fontsize=fs)
        ax.set_xlabel('Random iteration',fontsize=fs)
        ax.set_title(f'Shared locations between PSD location and random\n {p} sensors in {n} locations',fontsize=fs)
        fig.tight_layout()
        
    def plot_swaps(self,obj_swap,r,p,residuals_cov_random,random_locations):
        random_covariance = {el:0 for el in random_locations.keys()}
        for cov_mat,loc in zip(residuals_cov_random,random_locations.keys()):
            loc_sensors = random_locations[loc][0]
            sensors_cov = np.diag(cov_mat)
            random_covariance[loc] = [sensors_cov.min(),sensors_cov.max()]
        
        random_covariance_max = np.array([random_covariance[i][1] for i in random_covariance.keys()])
        
        
        fs = 10
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        ax.plot(obj_swap,color='#1a5276',label='Swap')
        ax.hlines(y=r/p,xmin=-1,xmax=len(obj_swap)+1,color='orange',label='r/p',linestyles='--')
        ax.hlines(y=random_covariance_max.min(),xmin=-1,xmax=len(obj_swap)+1,color='k',label='best random',linestyles='--')
        ax.set_xticks(np.arange(0,len(obj_swap),1))
        ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
        ax.set_yticks(np.arange(0,1.,0.2))
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()],fontsize=fs)
        ax.set_title('Ordered swap')
        ax.legend()
        
        fig.tight_layout()
        return fig