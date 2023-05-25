#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor placement problem for Low-cost sensors (LCSs) network.
Location based on covariance residual matrix.
Problem solved using LMIs

Created on Wed May 17 10:51:25 2023

@author: jparedes
"""
import os
import pandas as pd
import scipy
import math
import networkx as nx
import itertools
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle
import warnings

from scipy.spatial.distance import squareform, pdist

import SensorPlacementMethods as SPM

#%% Network creation
class DataSet():
    def __init__(self,n):
        self.n = n
        
    def generate_points(self):
        """
        Generate random points in space (nodes of the graph)
        """
        # genrate random points in plane and get neighbors
        rng = np.random.default_rng(seed=92)
        self.points = rng.uniform(0.0,1.0,(self.n,2))
        self.dist = squareform(pdist(self.points,metric='euclidean'))
        
    def generate_cluster_graph(self,num_clusters,plot_graph=True):
        self.num_clusters = num_clusters
        p = self.points
        
        #Cluster the points
        kmeans = KMeans(n_clusters=num_clusters).fit(p)
        labels = kmeans.labels_
        
        # create Graph object
        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i,pos=p[i],cluster=labels[i])
        for i in range(self.n):
            for j in range(i+1,self.n):
                if labels[i] == labels[j]:
                    G.add_edge(i,j,weight=self.dist[i,j])
        
        self.G = G
        
        # plot
        if plot_graph:
            fig = plt.figure(figsize=(3.5,2.5))
            ax = fig.add_subplot(111)
            pos = {i: G.nodes[i]['pos'] for i in range(self.n)}
            nx.draw(G, pos, node_color=[G.nodes[i]['cluster'] for i in range(self.n)], with_labels=True)
            ax.set_title(f'Graph with {self.num_clusters} clusters')
            
        
    def Generate_NeighborsGraph(self,a=0.0,b=1.0,n_neighbors=1,dist_based=True):
        """
        Create graph from K-nearest neighbors
        """
        k = n_neighbors + 1
        dist = self.dist
        
        # count connections
        neighbors = np.argsort(dist, axis=1)[:, 0:k]
       
        # # connect
        # coords = np.zeros((n, k, 2, 2))
        # for i in np.arange(n):
        #     for j in np.arange(k):
        #         coords[i, j, :, 0] = np.array([p[i,:][0], p[neighbors[i, j], :][0]])
        #         coords[i, j, :, 1] = np.array([p[i,:][1], p[neighbors[i, j], :][1]])
                
        # create graph object
        G = nx.Graph()
        G.add_nodes_from(range(self.n,self.n))
        for i in np.arange(neighbors.shape[0]):
            for j in np.arange(1,neighbors.shape[1]):
                G.add_edge(neighbors[i,0],neighbors[i,j])
        
        # store distance between nodes and graph
        self.dist = dist
        self.neighbors = neighbors
        self.G = G
        
    def remove_GraphConnections(self,fraction=0.1,plot_graph=True):
        """
        Randomly remove connections.
        Random nodes are sampled from the network and then the edge with the largest distance is removed
        """
        G = self.G
        dist = self.dist
        rng = np.random.default_rng(seed=0)
        nodes_remove = rng.choice(G.nodes(),int(fraction*G.number_of_nodes()),replace=False)
        # edges_remove = rng.choice(G.number_of_edges(),int(fraction*G.number_of_edges()),replace=False)
        # for i, (u, v) in enumerate(G.edges()):
        #     if i in edges_remove and G.degree()[u] > 1 and G.degree()[v]>1:# remove edges but avoid isolating nodes
        #         G.remove_edge(u, v)
        for node in nodes_remove:
            node_neighbors = list(G.neighbors(node))
            if len(node_neighbors)>1:# avoid leaving single nodes
                largest_distance = np.max([dist[node,neighbor] for neighbor in node_neighbors])
                sorted_neighbors = sorted(node_neighbors, key=lambda x: G.edges[(node, x)]['weight'])
                G.remove_edge(node,sorted_neighbors[-1])
        
        self.G = G
        
        # plot
        if plot_graph:
            fig = plt.figure(figsize=(3.5,2.5))
            ax = fig.add_subplot(111)
            pos = {i: G.nodes[i]['pos'] for i in range(self.n)}
            nx.draw(G, pos, node_color=[G.nodes[i]['cluster'] for i in range(self.n)], with_labels=True)
            ax.set_title(f'Graph with {self.num_clusters} clusters')
        
        
        
        
    def get_laplacian(self,dist_based=True,plot_laplacian=True):
        """
        Get the laplacian of the graph.
        It can be dist_based: Adjacency matrix as a function of distance
        or can count the number of neighbors connections
        """
        G = self.G
        # laplacian
        if dist_based:
            dist = self.dist
            W = np.exp((-dist**2)/(2*0.5)) # weight distance-based adjacency matrix
            
            A = (nx.adjacency_matrix(G)).toarray()
            W = W*A 
            
            D = np.diag(W.sum(axis=0))
            L = D-W
        
        else:
            #A = (nx.adjacency_matrix(G)).toarray()
            #D = np.diag([val for (node,val) in G.degree()])
            L = (nx.laplacian_matrix(G)).toarray() # same as D-A,obviously
        
        self.L = L
        
        if plot_laplacian:
            fig = plt.figure(figsize=(3.5,2.5))
            ax = fig.add_subplot(111)
            im = ax.imshow(self.L,vmin=self.L.min(),vmax=self.L.max(),cmap='Blues')
            cbar = plt.colorbar(im)
            ax.set_title('Graph laplacian')
            
    def sample_from_graph(self,num_samples=1):
        """
        get samples from graph
        If num_samples is larger than 1 then it samples multiple measurements,
        thus creating a snapshots matrix
        """
        CovMat = np.linalg.pinv(self.L)
        #np.fill_diagonal(CovMat,1)
        rng = np.random.default_rng(seed=100)
        self.x = rng.multivariate_normal(mean=np.zeros(self.n), cov=CovMat,size=(num_samples)).T
        
    def plot_network(self):
        L = self.L
        G = self.G
        # show graph
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(L,vmin=L.min(),vmax=L.max())
        cbar = plt.colorbar(im)
        ax.set_title('Laplacian matrix')
        
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_subplot(111)
        nx.draw_networkx(G)
        ax.set_title(f'Graph built from {self.knn} nearest neighbors')

def check_consistency(n,r,p_zero,p_eps,p_empty):
    if p_zero + p_eps < r:
        raise Warning(f'The assumptions on the KKT probelm (left-invertibility of [Theta_eps;Theta_zero] == linearly independent columns) impose that the number of eigenmodes has to be smaller than the number of sensors.\nReceived:\nEigenmodes: {r}\nSensors: {p_zero + p_eps}')
    if p_zero > r:
        raise Warning(f'The assumptions on the KKT problem (right-invertibility of Theta_zero == linearly independent rows) impose that the number of reference stations has to be smaller than the number of eigenmodes.\nReceived:\nRef.St: {p_zero}\nEigenmodes: {r}')
    if p_zero + p_eps + p_empty != n:
        raise Warning(f'Something seems wrong about the distribution of sensors:\n Ref.St ({p_zero}) + LCSs ({p_eps}) + Empty ({p_empty}) != number of locations (n)')
    if p_zero <= r and r <= p_zero + p_eps:
        print(f'Number of sensors pass consistency check: Ref.St ({p_zero}) <= r ({r}) <= Ref.St ({p_zero}) + LCSs ({p_eps})')
    return

        
#%% data pre-processing
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



#%%
# =============================================================================
# Sensors locations
# =============================================================================

# Convex relaxation
def locations_SDP(Psi,p_eps,p_empty,n):
    """
    Optimal locations found via A-optimal convex relaxation
    Inequality expressed as LMI and solved using SDP
    """
    # SDP results
    H_optimal,C_optimal,obj_value = SPM.ConvexOpt_LMI(Psi, p_eps)
    #H_optimal,obj_value = SPM.ConvexOpt_SDP_regressor(Psi,p)
    # optimal locations
    loc = np.sort(np.diag(H_optimal).argsort()[-p_eps:])
    #loc = np.sort(C_optimal[0,:].argsort()[-p_eps:])
    loc_empty = np.sort([i for i in np.arange(0,n) if i not in loc])
    weights = np.diag(H_optimal)
    optimal_locations = [loc,loc_empty]
    
    return weights,optimal_locations


        
# Random placement
def locations_random(p_eps,p_empty,n,num_samples=10):
    print(f'Sampling {num_samples} over {math.comb(n,p_eps)} possible combinations.')
    locations = np.arange(n)
    random_locations = {el:0 for el in np.arange(num_samples)}
    rng = np.random.default_rng(seed=92)
    for i in np.arange(num_samples):
        rng.shuffle(locations)
        loc_eps = np.sort(locations[:p_eps])
        loc_empty = np.sort(locations[-p_empty:])
        random_locations[i] = [loc_eps,loc_empty]
    
    return random_locations

#%% 
# =============================================================================
# Estimations
# =============================================================================

def OLS_estimations(optimal_locations,n,signal_lcs):
    """
    Reconstruct signal from measurements at certain points
    """
    loc = optimal_locations[0]
    In = np.identity(n)
    C_eps = In[loc]
    Theta_eps = C_eps@Psi
    
    # measurements
    y_eps = C_eps@signal_lcs
    
    # estimations
    beta_hat = np.linalg.inv(Theta_eps.T@Theta_eps)@Theta_eps.T@y_eps # OLS beta
    y_hat = Psi@beta_hat
    
    return y_hat

def OLS_cov(optimal_locations,sigma_eps,n,Psi):
    """
    Residual covariance matrix
    """
    loc = optimal_locations[0]
    In = np.identity(n)
    C_eps = In[loc]
    Theta_eps = C_eps@Psi
    
    beta_cov = sigma_eps*np.linalg.pinv(Theta_eps.T@Theta_eps)
    residuals_cov = Psi@beta_cov@Psi.T
    
    return beta_cov,residuals_cov

#%%
# =============================================================================
# Plots
# =============================================================================

def plot_covariances(residuals_cov,residuals_cov_random,optimal_locations,random_locations):
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
    random_locations_min = random_locations[idx_max]
    
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
    random_locations_max = random_locations[np.argmax(traces_random)]
    
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
#%%

if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # =============================================================================
    #   Generate Data set
    # =============================================================================
    
    n,m = 20,1000# num_stations,num_snapshots
    # create network
    ds = DataSet(n)
    ds.generate_points()
    ds.generate_cluster_graph(num_clusters=2,plot_graph=False)
    ds.remove_GraphConnections(fraction=0.5,plot_graph=True)
    ds.get_laplacian(dist_based=True,plot_laplacian=True)
    ds.sample_from_graph(num_samples=m)
    X = ds.x # snapshots matrix
    # X = createData(np.zeros(shape=(n)), n, m,plot_graph=True)
    
    # =============================================================================
    #   Get reduced basis
    # =============================================================================
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    U,S,Vt = np.linalg.svd(X_scaled,full_matrices=False)
    beta = np.diag(S)@Vt
    
    #sparsity
    r = 15
    Psi = U[:,:r]
    beta_sparse = beta.copy()
    beta_sparse[r:,:] = np.zeros(shape=(beta_sparse[r:,:].shape))
    X_sparse = U@beta_sparse # = Psi@beta[:r,:]
    
    # =============================================================================
    #   Sensors parameters
    # =============================================================================
    
    # Refst and LCS
    var_eps = 1
    var_zero = 1e-3
    var_ratio = var_zero/var_eps #should be <1
    
    X_eps = Perturbate_data(X_sparse,noise=var_eps)
    X_zero = Perturbate_data(X_sparse,noise=var_zero)
    
    # number of sensors
    p = r #total
    p_eps = p #LCSs
    p_zero = 0# Ref.St. ==  p-p_eps
    p_empty = n-(p_zero+p_eps) #no-sensor
    
    print(f'Sensor placement parameters\n{n} locations\n{r} eigenmodes\n{p_eps+p_zero} sensors in total\n{p_eps} LCSs - variance = {var_eps:.2e}\n{p_zero} Ref.St. - variance = {var_zero:.2e}\n{p_empty} Empty locations')
    check_consistency(n,r,p_zero,p_eps,p_empty)
    input('Press Enter to continue...')
    
    # =============================================================================
    #     Convex relaxation
    # =============================================================================
    print('Solving Convex Optimization problem')
    weights,optimal_locations = locations_SDP(Psi, p_eps, p_empty, n)
    
    # reconstruction
    print('Reconstructing signal')
    y_hat = OLS_estimations(optimal_locations,n,X_eps)
    # covariances
    beta_cov, residuals_cov = OLS_cov(optimal_locations,var_eps,n,Psi)
    
    # =============================================================================
    #     Random placement: repeat reconstruction
    # =============================================================================
    
    print('Random placement')
    random_locations = locations_random(p_eps,p_empty,n,num_samples=10)
    
    y_hat_random = []
    beta_cov_random = []
    residuals_cov_random = []
    for i in range(len(random_locations)):
        y_hat_random.append(OLS_estimations(random_locations[i], n, X_eps))
        b,r = OLS_cov(random_locations[i],var_eps,n,Psi)
        beta_cov_random.append(b)
        residuals_cov_random.append(r)
        
    
        
    # print('empty_loc\t RMSE')
    # for k,val in zip(reconstruction_error_empty_random.keys(),reconstruction_error_empty_random.values()):
    #     print(f'{k}\t {val}')
        
    # fname = f'ReconstructionErrorEmpty_Randomplacement_RefSt{p_zero}_LCS{p_eps}_Empty{p_empty}_r{r}_varRatio{var_ratio}.pkl'
    # with open(results_path+fname, 'wb') as handle:
    #     pickle.dump(reconstruction_error_empty, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # =============================================================================
    #     Graphics
    # =============================================================================
    
    # plot
    print('Plotting results')
    # fig_weights = plot_locations_weights(weights,Trace_residuals,n)
    # fig_trace = plot_trace(Trace_residuals,Trace_residuals_random,p_eps,p_zero,p_empty,r)
    fig_covariances, dict_results = plot_covariances(residuals_cov,residuals_cov_random,optimal_locations,random_locations)
    
    
    print('------------\nAll Finished\n------------')
    
