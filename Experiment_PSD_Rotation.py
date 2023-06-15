#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:32:39 2023

@author: jparedes
"""
import os
import numpy as np
from scipy import linalg
import itertools
from matplotlib import pyplot as plt
import warnings
import decimal
import scipy.linalg
import math
import time

import LoadDataSet as LDS
import LowRankDecomposition as LRD
import SensorPlacementMethods as SPM
import Plots
#%%
def get_lmi_sparse(S,n,p,complexity='np'):
    tol = 5e-2
    x_range = np.arange(0.,1+tol,tol)
    # vector constraints
    R1 = np.zeros((p,n*p))
    for i in range(p):
        R1[i,i*n:(i*n+n)] = np.ones(n)
        
    R2 = np.tile(np.identity(n),p)
    
    if complexity == 'np':
        
        x_valid = np.array([np.round(x,2) for x in itertools.product(x_range,repeat=n*p) if np.sum(x)==p])
        x_valid = [x for x in x_valid if (R1@x == np.ones((p,1))).all()]
        x_valid = [x for x in x_valid if (R2@x <= np.ones((n,1))).all()]
        x_valid = [x for x in x_valid if (R2@x>=np.zeros((n,1))).all()]
        x_valid = np.array(x_valid)
        
        
        LMI = np.array([np.array([x[i]*S[i] for i in range(n*p)]).sum(axis=0)for x in x_valid])
        LMI_q = np.array([np.array([(x[i]**2)*S[i] for i in range(n*p)]).sum(axis=0) for x in x_valid])
        
        # apply sensors symmetry
        LMI = np.array([M for M in LMI if M[3,0]==M[4,0] and M[3,1]==M[4,1] and M[3,2] == M[4,2]])
        LMI_q = np.array([M for M in LMI_q if M[3,0]==M[4,0] and M[3,1]==M[4,1] and M[3,2] == M[4,2]])
        x_valid = [np.tile(M[n,:p+1],p) for M in LMI]
        x_valid = np.reshape(x_valid,(len(x_valid),n*p))
    
    elif complexity == 'n':
        x_valid = np.array([np.tile(np.round(x,2),p) for x in itertools.product(x_range,repeat=n) if np.sum(x)==1])
        x_valid = [x for x in x_valid if (R1@x == np.ones((p,1))).all()]
        x_valid = [x for x in x_valid if (R2@x <= np.ones((n,1))).all()]
        x_valid = [x for x in x_valid if (R2@x>=np.zeros((n,1))).all()]
        x_valid = np.array(x_valid)
        
        LMI = np.array([np.array([x[i]*S[i] for i in range(n*p)]).sum(axis=0)for x in x_valid])
        LMI_q = np.array([np.array([(x[i]**2)*S[i] for i in range(n*p)]).sum(axis=0) for x in x_valid])
        
    
    return LMI,LMI_q,x_valid

def get_lmi_non_sparse(LMI_sparse,x_valid,Psi,n,p,r,B):
    """
    Obtain p-LMIs in non-sparse representation:
    R.T@LMI_sparse@R + B(t) >= 0
    
    return PSD/non-PSD points
    """
    Ip = np.identity(p)
    R = [np.block([[Psi,np.zeros((n,1))],[np.zeros((p,r)),Ip[:,j][:,None]]]) for j in range(p)]
    x_non_PSD = []
    x_PSD = []
    for j in range(len(LMI_sparse)):
        LMIs = [R[i].T@LMI_sparse[j]@R[i] + B for i in range(p)]# p LMIs per configuration
        # check PSD: LMI>=0
        if any(np.concatenate([np.linalg.eig(LMIs[k])[0] for k in range(p)])<0):
            x_non_PSD.append(x_valid[j])
        if all(np.concatenate([np.linalg.eig(LMIs[k])[0] for k in range(p)])>=0):
            x_PSD.append(x_valid[j])
            
    return  [x_PSD,x_non_PSD]
    
#%%
class Basis():
    """
    Basis of eigenmodes
    """
    def __init__(self,n,r,basis):
        self.n = n
        self.r = r
        self.basis = basis
        
    def create_psi_basis(self,random_seed=40):
        """
        Create a basis of eigenmodes
        """
        
        if self.basis=='identity':
            print('Identity basis')
            U = np.identity(self.n)
            
        elif self.basis=='random':
            rng = np.random.default_rng(seed=random_seed)#good basis = [40,50], bad basis = [1,92]
            U = linalg.orth(rng.random((self.n,self.n)))
            
        
        elif self.basis == 'stations':
            # load data set for specific stations
            if self.n == 4:
                RefStations = ['Palau-Reial','Eixample','Gracia','Ciutadella']
            elif self.n== 3:
                RefStations = ['Palau-Reial','Eixample','Gracia']
            pollutant = 'O3'
            start_date = '2011-01-01'
            end_date = '2022-12-31'
            dataset = LDS.dataSet(pollutant,start_date,end_date, RefStations)
            dataset.load_dataSet(file_path)
            dataset.cleanMissingvalues(strategy='remove')
            # get eigenbasis
            X_train = LRD.Snapshots_matrix(dataset.ds)
            U,S,V = LRD.low_rank_decomposition(X_train,normalize=True)
            
        
        elif self.basis=='rotation_x':
            if self.n == 3:
                rotation = np.block([[1,0,0],[0,np.cos(np.pi/4),-np.sin(np.pi/4)],[0,np.sin(np.pi/4),np.cos(np.pi/4)]])
            elif self.n==2:
                rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
            U = rotation@np.identity(self.n)
            
        elif self.basis=='rotation_z':
            if self.n==3:
                rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4),0],[np.sin(np.pi/4),np.cos(np.pi/4),0],[0,0,1]])
                
            elif n==2:
                rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
            U = rotation@np.identity(self.n)
        
        self.Psi = U[:,:self.r]
        return

    def rotation_matrix(self,u,v,theta):
        """
        Rotates a vector in the R^(n-2) plane by an angle theta
        given two orthogonal vectors in R^n
        """
        L = v@u.T - u@v.T
        rotation = scipy.linalg.expm(theta*L)
        return rotation
    
    def rotate_eigenmodes(self,u,v,theta):
        """
        rotate the eigenmodes according to certain points and angle
        """
        R = self.rotation_matrix(u,v,theta)
        return R@self.Psi
    
    def perturbate_eigenmodes(self,mode='rotation',num_rotations = 5,theta_max=np.pi/20,rotation_vector=0):
        """
        Perturbate eigenmodes to new values.
        """
        if mode == 'rotation_random':
            
            possible_vectors = np.array([np.array(i).T for i in itertools.combinations(np.identity(self.n),2)])
            rng = np.random.default_rng(seed=0)
            rotations_vectors = rng.choice(possible_vectors,size=num_rotations,replace=True)
            rotations_angles = rng.uniform(low=0.0,high=theta_max,size=num_rotations)
            Psi = self.Psi    
            
            for u,theta in zip(rotations_vectors,rotations_angles):
                Psi = self.rotation_matrix(u[:,0][:,None],u[:,1][:,None],theta)@Psi
        
        elif mode=='rotation_single':
            possible_vectors = np.array([np.array(i).T for i in itertools.combinations(np.identity(self.n),2)])
            rotation_vector = possible_vectors[rotation_vector]
            Psi = self.rotation_matrix(rotation_vector[:,0][:,None],rotation_vector[:,1][:,None],theta_max)@self.Psi
            
        self.Psi = Psi
                
# =============================================================================
# Mesh
# =============================================================================
def Mesh_LMI(t_range,p,n,Psi):
    """
    LMI evolution at different t-thresholds
    """
    # sparse matrices
    S = []
    for i in range(p):
        for j in range(n):
            C = np.zeros((p,n))
            C[i,j] = 1
            S_k = np.block([[C.T@C,C.T],[C,np.zeros((p,p))]])
            S.append(S_k)
    
    LMI = []
    LMI,LMI_q,x_valid = get_lmi_sparse(S,n,p,complexity='n')
    
    
    B_t = {np.round(el,exponent):0 for el in t_range}
    for t in t_range:
        t = np.round(t,exponent)
        B_j = np.zeros((r+1,r+1))
        B_j[-1,-1] = t
        B_t[t] = B_j
    
    
    # evaluate PSD at different t-values
    print(f'Getting positive semi-definite points for different t values:\n{[np.round(i,2) for i in t_range]}\nPoint of interest at r/p = {r/p}')
    x_PSD_t = {np.round(el,exponent):0 for el in t_range}
    for t in t_range:
        t = np.round(t,exponent)
        B = B_t[t]
        x_PSD_t[t] = get_lmi_non_sparse(LMI,x_valid,Psi,n,p,r,B)
    
    return x_PSD_t

def Convex_LMI(Psi,n,p,r,t):
    convex_opt = SPM.ConvexOpt(Psi,n,p,r)
    convex_opt.ConvexOpt_LMI(threshold_t=t,var_beta=1.0)
    C,H,obj = convex_opt.C, convex_opt.H,convex_opt.obj
    return C,H,obj
#%%
def discretization(C,H,p,n):
    """
    converts C,H from linear solution with values in [0,1] into discrete {0,1}
    """
    locations = np.argsort(np.diag(H))[-p:]
    In = np.identity(n)
    C_discrete = In[np.sort(locations)]
    H_discrete = C_discrete.T@C_discrete
    return C_discrete, H_discrete

def compare_solution(C,H,Psi,obj,r,n,p,t):
    """
    Given C,H,obj: solutions of the convex optimization problem
    Compute and compare covariance matrix
    """
    # real solutions
    In = np.identity(n)
    C_sparse = np.array([x for x in itertools.combinations(In,p)])
    covariances = [C@Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T@C.T for C in C_sparse]
    # print('Different covariance matrices')
    # for i in range(C_sparse.shape[0]):
    #     print(f'C=\n{C_sparse[i]}\nSigma=\n{covariances[i]}')
    print(f'Worst case scenario covariance entry:\nCombination of maximum:\n{C_sparse[np.argmax(([np.diag(cov).max() for cov in covariances]))]}\nValue:{np.max([np.diag(cov).max() for cov in covariances])}\nCombination of minimum:\n{C_sparse[np.argmin(([np.diag(cov).max() for cov in covariances]))]}\nValue: {np.min([np.diag(cov).max() for cov in covariances])}')
    
    # SDP solution at step t
    cov = C@Psi@np.linalg.inv(Psi.T@H@Psi)@Psi.T@C.T
    print(f'\nLinear solution at threshold t>={t}\nobjective={obj}\nC=\n{C}\nH=\n{H}\nCovariance=\n{cov}')
    print(f'Maximum diagonal value of covariance matrix: {np.diag(cov).max()}')
    
    # SDP solution discretization
    C_discrete, H_discrete = discretization(C,H,p,n)
    cov_real = C_discrete@Psi@np.linalg.inv(Psi.T@H_discrete@Psi)@Psi.T@C_discrete.T
    print(f'Discretization of SDP results:\nLocations:\n{C_discrete}\ncovariance:\n{cov_real}\nMax diagonal value: {np.max(np.diag(cov_real))}')
    
    
    # # vertex solutions
    # In = (1/p)*np.identity(n)
    # x_vertex = np.array([np.tile(i,p).reshape((p,n)) for i in itertools.combinations(In,1)])
    # x_vertex = np.array([np.sum(i,axis=0) for i in itertools.combinations(x_vertex,2)])
    # cov_discrete = [C@Psi@np.linalg.inv(Psi.T@np.diag(C.sum(axis=0))@Psi)@Psi.T@C.T for C in x_vertex]
    
    # print(f'\nVertex solutions:')
    # for i in range(x_vertex.shape[0]):
    #     print(f'C=\n{x_vertex[i]}\nH=\n{np.diag(x_vertex[i].sum(axis=0))}\nSigma=\n{cov_discrete[i]}')
        
    return
#%%
if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    # =============================================================================
    #     Parameters
    # =============================================================================
    n,p,r = 30,12,7
    basis = Basis(n,r,basis='random')
    basis.create_psi_basis(random_seed=92)
    Psi_orig = basis.Psi.copy()
    
    
    t_tol = 1e-3
    t_range = np.arange(0.,r/p+t_tol,t_tol)
    exponent = -1*decimal.Decimal(str(t_tol)).as_tuple().exponent
    
    # =============================================================================
    #     Mesh cone
    # =============================================================================
    
    create_mesh = False
    if create_mesh:
        x_PSD_t = Mesh_LMI(t_range,p,n,basis.Psi)
        t_last_point = r/p
        for t in np.arange(r/p,0.0,-t_tol):
            t = np.round(t,exponent)
            if len(x_PSD_t[t][0]) == 0:
                break
            else:
                t_last_point = t
    else:
        t_last_point = 0.0

    # =============================================================================
    # Find solution point via SDP
    # =============================================================================

    C_init,H_init,obj_init = Convex_LMI(Psi_orig,n,p,r,0.0)
    
    
    proximity_tol = 4e-2
    time_init = time.time()
    n_iter = 500
    C_iter = {el:0 for el in range(n_iter)}
    H_iter = {el:0 for el in range(n_iter)}
    for i in range(n_iter):
        rng = np.random.default_rng(seed=i)
        count = 0
        basis.Psi = Psi_orig
        vertex_proximity = np.sum(np.diag(H_init)**2)
        while p-vertex_proximity > proximity_tol:  # still far from vertex
            count+=1
            theta,plane = rng.normal(loc=0.0,scale=1.0,size=1)[0],rng.integers(low=0.0,high=math.comb(n,2) -1)
            basis.perturbate_eigenmodes(mode='rotation_single',theta_max=np.deg2rad(theta),rotation_vector=plane)
            
            C,H,obj = Convex_LMI(basis.Psi,n,p,r,0.0)
            vertex_proximity_iter = np.sum(np.diag(H)**2) # tends to p as the point is close to a vertex
            
            if vertex_proximity_iter - vertex_proximity > 1e-4:# keep basis and update proximity
                vertex_proximity = vertex_proximity_iter
            else:# keep old proximity and return rotation
                basis.perturbate_eigenmodes(mode='rotation_single',theta_max=np.deg2rad(-theta),rotation_vector=plane)
        C_iter[i] = C
        H_iter[i] = H
        
        
    time_end = time.time()
    print(f'Finished in: {(time_end-time_init):.2f} seconds after {count} rotations')
    
    # Psi_final = basis.Psi
    # print('\nOriginal basis results')
    compare_solution(C_init,H_init,Psi_orig,obj_init,r,n,p,t_last_point)
    # print('\nFinal basis results')
    # compare_solution(C,H,Psi_final,obj,r,n,p,t_last_point)
    # print('\nOriginal basis with final locations')
    # compare_solution(C,H,Psi_orig,obj,r,n,p,t_last_point)
    
    
    
    
    