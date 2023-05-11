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
        G.add_nodes_from(range(n,n))
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
        self.x = rng.multivariate_normal(mean=np.zeros(n), cov=CovMat,size=(num_samples)).T
        
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
        
        
        
#%% data pre-processing
def Generate_graph(n,k,a=0.0,b=1.0,dist_based=True):
    
    # genrate random points in plane and get neighbors
    rng = np.random.default_rng(seed=92)
    p = rng.uniform(0.0,1.0,(n,2))
    dist = squareform(pdist(p,metric='euclidean'))
    
    # count connections
    neighbors = np.argsort(dist, axis=1)[:, 0:k]
    # connect
    coords = np.zeros((n, k, 2, 2))
    for i in np.arange(n):
        for j in np.arange(k):
            coords[i, j, :, 0] = np.array([p[i,:][0], p[neighbors[i, j], :][0]])
            coords[i, j, :, 1] = np.array([p[i,:][1], p[neighbors[i, j], :][1]])
            
    G = nx.Graph()
    G.add_nodes_from(range(n,n))
    for i in np.arange(neighbors.shape[0]):
        for j in np.arange(1,neighbors.shape[1]):
            G.add_edge(neighbors[i,0],neighbors[i,j])
            
    # laplacian
    if dist_based:
        A = (nx.adjacency_matrix(G)).toarray()
        W = np.exp((-dist**2)/(2*0.5)) # weight distance-based adjacency matrix
        #W = W*A #?
        D = np.diag(W.sum(axis=0))
        L = D-W
    
    else:
        #A = (nx.adjacency_matrix(G)).toarray()
        #D = np.diag([val for (node,val) in G.degree()])
        L = (nx.laplacian_matrix(G)).toarray() # same as D-A,obviously
    
    # show graph
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(L,vmin=L.min(),vmax=L.max())
    cbar = plt.colorbar(im)
    ax.set_title('Laplacian matrix')
    
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot(111)
    nx.draw_networkx(G)
    ax.set_title(f'Graph built from {k-1} nearest neighbors')
    
    return L

def createData(mean,n,m,plot_graph=False):
    G=nx.fast_gnp_random_graph(n,0.5,seed=92)
    adj_matrix = nx.adjacency_matrix(G)
    M = adj_matrix.toarray()
    M_ = M.copy()
    np.fill_diagonal(M, 1)
    
    rng = np.random.default_rng(seed=92)
    sigmas = rng.random(size=(n,n))
    sigmas = (sigmas + sigmas.T)/2
    CovMat = np.multiply(sigmas,M)
    np.fill_diagonal(CovMat,np.ones(CovMat.shape[0]))
    X = rng.multivariate_normal(mean, CovMat, m).T
    
    if plot_graph:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(M_,cmap='gray')
        ax.set_title('Graph adjacency matrix')
        
        ax1 = fig.add_subplot(122)
        im1 = ax1.imshow(CovMat)
        fig.colorbar(im1)
        ax1.set_title('Graph covariance matrix')
        
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        nx.draw_networkx(G)
        plt.show(block=False)
        plt.pause(0.001)

    
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
        raise Warning(f'The assumptions on the KKT probelm (left-invertibility of [Theta_eps;Theta_zero] == linearly independent columns) impose that the number of eigenmodes has to be smaller than the number of sensors.\nReceived:\nEigenmodes: {r}\nSensors: {p_zero + p_eps}')
    if p_zero > r:
        raise Warning(f'The assumptions on the KKT problem (right-invertibility of Theta_zero == linearly independent rows) impose that the number of reference stations has to be smaller than the number of eigenmodes.\nReceived:\nRef.St: {p_zero}\nEigenmodes: {r}')
    if p_zero + p_eps + p_empty != n:
        raise Warning(f'Something seems wrong about the distribution of sensors:\n Ref.St ({p_zero}) + LCSs ({p_eps}) + Empty ({p_empty}) != number of locations (n)')
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
#%% random placement functions
def locations_random(p_eps,p_zero,p_empty,n,num_samples=10):
    locations = np.arange(n)
    random_locations = {el:0 for el in np.arange(num_samples)}
    rng = np.random.default_rng(seed=92)
    for i in np.arange(num_samples):
        rng.shuffle(locations)
        loc_eps = np.sort(locations[:p_eps])
        loc_zero = np.sort(locations[p_eps:-p_empty])
        loc_empty = np.sort(locations[-p_empty:])
        random_locations[i] = [loc_eps,loc_zero,loc_empty]
    
    return random_locations

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
    
    random_locations = {}
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
            random_locations[i] = [loc_eps,loc_zero,loc_empty]
            
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

#%% Optimal procedure
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
    Covariance_beta = {el:0 for el in reg_param}
    Covariance_res = {el:0 for el in reg_param}
    Trace_res = {el:0 for el in reg_param}
    
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
            
            Covariance_beta[l] = Sigma_beta
            Covariance_res[l] = Sigma_residuals
            Trace_res[l] = np.trace(Sigma_residuals)
            
        elif p_eps == 0: # reconstruct using only Ref.St.
            
            # Exact OLS
            
            Sigma_beta = np.zeros(shape=(Psi.shape[0],Psi.shape[0])) 
            Sigma_residuals = Psi@Sigma_beta@Psi.T
            
            Covariance_beta[l] = Sigma_beta
            Covariance_res[l] = Sigma_residuals
            Trace_res[l] = np.trace(Sigma_residuals)
            
            
    return Covariance_beta, Covariance_res, Trace_res
#%% plots
def plot_locations_weights(weights,Trace_optimal,n):
    """
    Plot weights distribution for best (and worst?) trace results
    after convex optimization relaxation experiment
    """
    lambdas = [i for i in weights.keys()]
    idx_opt = np.argmin([i for i in Trace_optimal.values()])
    weights_lcs, weights_zero = weights[lambdas[idx_opt]][0], weights[lambdas[idx_opt]][1]
    
    figx,figy = 3.5,2.5
    fs = 10
    
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(n),weights_lcs,label='LCSs',color='#ca6f1e')
    ax.bar(np.arange(n),-1*weights_zero,label='Ref. St.',color='#1f618d')
    
    ax.set_yticks(np.arange(-1,1.25,0.25))
    ax.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=fs)
    ax.set_ylabel('Weight',fontsize=fs)
    
    ax.set_xticks(np.arange(0,n,5))
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    ax.set_xlabel('Location index',fontsize=fs)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    
    return fig

def plot_trace(Trace_optimal,Trace_random,p_eps,p_zero,p_empty,r,save=False):
    """
    Plot Trace of residuals covariance matrix for different values of regularization parameter
    Compares the results with random sampling
    """
    lambdas = Trace_optimal.keys()
    random_vals = [ i for i in Trace_random.values()]
    
    figx,figy = 3.5,2.5
    fs = 10
    
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0,len(lambdas)),np.round([i for i in Trace_residuals.values()],4),marker='o',color='#1f618d',label='Trace of convex opt. results')
    ax.fill_between(x=np.arange(0,len(lambdas)),y1=np.min(random_vals),y2 = np.max(random_vals),color='#ca6f1e',alpha=0.5,label=f'{len(Trace_random)} random placements')
    
    ax.legend(loc='upper right',fontsize=fs)
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels(labels=lambdas,fontsize=fs)
    ax.set_xlabel('Regularization parameter $\lambda$',fontsize=fs)
    
    ax.set_yticks(np.linspace(start=0.0, stop=np.round(np.max(random_vals)),num=5))
    ax.set_yticklabels(np.round(ax.get_yticks(),2),fontsize=fs)
    ax.set_ylabel('Tr $\Sigma_{\hat{e}}$',fontsize=fs)
    ax.set_title(f'Convex optimization results vs random placements\n {p_eps+p_zero+p_empty} locations, {r} eigenmodes, {p_eps} LCSs, {p_zero} Ref.st. {p_empty} Empty locations',fontsize=fs)
    
    ax.grid()
    fig.tight_layout()
    if save:
        fname = f'{results_path}Plot_trace_vs_lambda_RandomLocations_{p_eps}LCS_{p_zero}RefSt_{p_empty}Empty_{r}eigenmodes.png'
        plt.savefig(fname,dpi=600,format='png')
    
    
    return fig


def plot_covariances(Covariance_optimal,Covariance_random,Trace_optimal,Trace_random,n,locations_optimal,locations_random):
    """
    Plot residuals covariance matrices for different regularization parameters and compare them
    with random placements
    """
    lambdas = [i for i in Covariance_optimal.keys()]
    Traces_optimal = [i for i in Trace_optimal.values()]
    Traces_random = [i for i in Trace_random.values()]
    
    # min/max occurrence ofr different lambda & randomness
    Trace_optimal_max = np.argmax(Traces_optimal)# only keeps first occurrence
    Trace_optimal_min = np.argmin(Traces_optimal)
    Trace_random_max = np.argmax(Traces_random)
    Trace_random_min = np.argmin(Traces_random)
    
    # plot minimum and maximum covariances
    Covariance_optimal_min = Covariance_optimal[lambdas[Trace_optimal_min]]
    Covariance_optimal_max = Covariance_optimal[lambdas[Trace_optimal_max]]
    Covariance_random_min = Covariance_random[Trace_random_min]
    Covariance_random_max = Covariance_random[Trace_random_max]
    global_min = np.min([Covariance_optimal_min.min(),Covariance_optimal_max.min(),Covariance_random_min.min(),Covariance_random_max.min()])
    global_max = np.max([Covariance_optimal_min.max(),Covariance_optimal_max.max(),Covariance_random_min.max(),Covariance_random_max.max()])
    
    figx,figy = 3.5,2.5
    fs = 10
    
    fig = plt.figure(figsize=(figx,figy))

    ax = fig.add_subplot(221)
    im = ax.imshow(Covariance_optimal_max,vmin = global_min, vmax = global_max,cmap='Oranges')
    ax.set_title(f'Max Tr $\Sigma_e$ ($\lambda$= {lambdas[Trace_optimal_max]}) = {Traces_optimal[Trace_optimal_max]:.2f}',fontsize=fs)
    loc = np.sort(np.concatenate([locations_optimal[lambdas[Trace_optimal_max]][0],locations_optimal[lambdas[Trace_optimal_max]][-1]]))
    ax.set_xticks(loc)
    ax.set_yticks(loc)
    ax.set_xticklabels(ax.get_xticks(),fontsize=fs)
    ax.set_yticklabels(ax.get_xticks(),fontsize=fs)
    # identify LCSs
    for t in ax.get_xticks():
        if t in locations_optimal[lambdas[Trace_optimal_max]][0]:# change only LCSs locations
            idx = np.argwhere(t == ax.get_xticks())[0,0]    
            ax.get_xticklabels()[idx].set_color('red')
            ax.get_yticklabels()[idx].set_color('red')
            
    
    ax1 = fig.add_subplot(222)
    im1 = ax1.imshow(Covariance_optimal_min,vmin = global_min, vmax = global_max,cmap='Oranges')
    ax1.set_title(f'Min Tr $\Sigma_e$ ($\lambda$ = {lambdas[Trace_optimal_min]}) = {Traces_optimal[Trace_optimal_min]:.2f}',fontsize=fs)
    loc = np.sort(np.concatenate([locations_optimal[lambdas[Trace_optimal_min]][0],locations_optimal[lambdas[Trace_optimal_min]][-1]]))
    ax1.set_xticks(loc)
    ax1.set_yticks(loc)
    ax1.set_xticklabels(ax1.get_xticks(),fontsize=fs)
    ax1.set_yticklabels(ax1.get_xticks(),fontsize=fs)
    
    for t in ax1.get_xticks():
        if t in locations_optimal[lambdas[Trace_optimal_min]][0]:
            idx = np.argwhere(t == ax1.get_xticks())[0,0]
            ax1.get_xticklabels()[idx].set_color('red')
            ax1.get_yticklabels()[idx].set_color('red')
    
    
    ax2 = fig.add_subplot(223)
    im2 = ax2.imshow(Covariance_random_max,vmin = global_min, vmax = global_max,cmap='Oranges')
    ax2.set_title(f'Max Tr $\Sigma_e$ random = {Traces_random[Trace_random_max]:.2f}',fontsize=fs)
    loc = np.sort(np.concatenate([locations_random[Trace_random_max][0],locations_random[Trace_random_max][-1]]))
    ax2.set_xticks(loc)
    ax2.set_yticks(loc)
    ax2.set_xticklabels(ax2.get_xticks(),fontsize=fs)
    ax2.set_yticklabels(ax2.get_xticks(),fontsize=fs)
    
    
    for t in ax2.get_xticks():
        if t in locations_random[Trace_random_max][0]:
            idx = np.argwhere(t == ax2.get_xticks())[0,0]
            ax2.get_xticklabels()[idx].set_color('red')
            ax2.get_yticklabels()[idx].set_color('red')
            
    
    ax3 = fig.add_subplot(224)
    im3 = ax3.imshow(Covariance_random_min,vmin = global_min, vmax = global_max,cmap='Oranges')
    ax3.set_title(f'Min Tr $\Sigma_e$ random = {Traces_random[Trace_random_min]:.2f}',fontsize=fs)
    loc = np.sort(np.concatenate([locations_random[Trace_random_min][0],locations_random[Trace_random_min][-1]]))
    ax3.set_xticks(loc)
    ax3.set_yticks(loc)
    ax3.set_xticklabels(ax3.get_xticks(),fontsize=fs)
    ax3.set_yticklabels(ax3.get_xticks(),fontsize=fs)
    
    
    for t in ax3.get_xticks():
        if t in locations_random[Trace_random_min][0]:
            idx = np.argwhere(t == ax3.get_xticks())[0,0]
            ax3.get_xticklabels()[idx].set_color('red')
            ax3.get_yticklabels()[idx].set_color('red')
    
    plt.suptitle(f'Convex optimization results vs random placements\n {p_eps+p_zero+p_empty} locations, {r} eigenmodes, {p_eps} LCSs, {p_zero} Ref.st. {p_empty} Empty locations',fontsize=fs)

    fig.tight_layout()
    # color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    #cbar.set_ticks([mn,md,mx])
    #cbar.set_ticklabels([mn,md,mx])
    
    
        
    return fig
#%%

if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # =============================================================================
    #   Generate Data set
    # =============================================================================
    
    n,m = 500,1000# num_stations,num_snapshots
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
    r = 290
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
    p = n-300 #total
    p_eps = 200 #LCSs
    p_zero = 100#p-p_eps # Ref.St.
    p_empty = n-(p_zero+p_eps) #no-sensor
    
    print(f'Sensor placement parameters\n{n} locations\n{r} eigenmodes\n{p_eps+p_zero} sensors in total\n{p_eps} LCSs - variance = {var_eps:.2e}\n{p_zero} Ref.St. - variance = {var_zero:.2e}\n{p_empty} Empty locations')
    check_consistency(n,r,p_zero,p_eps,p_empty)
    input('Press Enter to continue...')
    
    # =============================================================================
    #     Convex relaxation
    # =============================================================================
    print('Solving Convex Optimization problem')
    weights, optimal_locations, obj_function = locations_vs_lambdas(Psi,
                                                                   p_eps,
                                                                   p_zero,
                                                                   p_empty,
                                                                   var_eps,
                                                                   var_zero,
                                                                   X_eps,
                                                                   X_sparse)
    
    # reconstruction
    reconstruction_error_full, reconstruction_error_zero, reconstruction_error_eps, reconstruction_error_empty = KKT_estimations(optimal_locations,p_eps,p_zero,n,var_eps,X_sparse,X_eps)
   
    Sigma_beta, Sigma_residuals, Trace_residuals = KKT_cov(optimal_locations,Psi,var_eps)
    

    print('Objective function\nLambda\t val')
    for lambda_reg,val in zip(obj_function.keys(),obj_function.values()):
        print(f'{lambda_reg}\t{val}')
       
    print(f'\n{p_empty} Reconstruction errors\nLambda\t RMSE')
    for lambda_reg,val in zip(reconstruction_error_full.keys(),reconstruction_error_full.values()):
        print(f'{lambda_reg}\t {val}')

    #save_results(objective_func,reconstruction_error_empty,locations,p_eps,p_zero,p_empty,r,var_ratio,results_path)
    
    # =============================================================================
    #     Random placement
    # =============================================================================
    
    print('Random placement')
    random_locations = locations_random(p_eps,p_zero,p_empty,n,num_samples=100)
    reconstruction_error_full_random, reconstruction_error_zero_random, reconstruction_error_eps_random, reconstruction_error_empty_random = KKT_estimations(random_locations,p_eps,p_zero,n,var_eps,X_sparse,X_eps)
    Sigma_beta_random, Sigma_residuals_random, Trace_residuals_random = KKT_cov(random_locations,Psi,var_eps)
    
        
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
    fig_weights = plot_locations_weights(weights,Trace_residuals,n)
    fig_trace = plot_trace(Trace_residuals,Trace_residuals_random,p_eps,p_zero,p_empty,r)
    fig_covariances = plot_covariances(Sigma_residuals,Sigma_residuals_random,Trace_residuals,Trace_residuals_random,n,optimal_locations,random_locations)
    
    
    print('------------\nAll Finished\n------------')
    
    
