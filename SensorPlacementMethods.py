#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor placement algorithms

Created on Mon Feb 27 10:05:11 2023

@author: jparedes
"""
import math
import itertools
import numpy as np
from scipy import linalg
import cvxpy as cp
#%%
# def get_combinations(n, m):
#     arr = []
#     for matrix in itertools.product([1, 0], repeat=n*m):
#         if sum(matrix)==n:
#             m_ = np.reshape(matrix,(n,m))
#             if (m_.sum(axis=1)<=1).all() and (m_.sum(axis=0)<=1).all():
#                 arr.append(m_)
#         # if m_.sum() == n and (m_.sum(axis=1)<=1).all():
#         #     arr.append(m_)
#         #     #yield m_
#     return arr



#%% D-optimal subset selection
"""
g* = argmax det (C Psi_r).T (C Psi_r)
     g, |g| = p
combinatorial search (n) determinant evaluations
                     (p)
"""
def possible_C(n,m):
    I = np.identity(m)
    placements = itertools.combinations(I,n)
    arr = []
    for m in placements:
        arr.append(np.array(m))
    return arr

def D_optimal(Psi,p):
    """
    n = space dimension
    r = number of basis vectors
    p = number of point measurements
    """
    n,r = Psi.shape 
    print(f'D-optimal subset selection algorithm\nspace dimension: {n}\nnumber of basis vectors:{r}\nnumber of points measurements: {p}\n{n}C{p} = {math.comb(n,p)} determinant evaluations')
    C = np.zeros((p,n))
    # initialize placement
    C = possible_C(p,n)
    results = []
    # iterate over possibilities
    for c in C:
        Theta = c @ Psi
        M = Theta.T @ Theta
        f = np.log10(np.linalg.det(M))
        results.append(f)
    # compute optimal placement
    loc = np.argmax(results)
    C_optimal = C[loc]
    locations = np.argwhere(C_optimal==1)[:,1]
    return C_optimal,locations
#%% QR pivoting
"""
Psi.T C.T = QR
C[pivots] = 1
"""
def QR_pivoting(Psi,p):
    """
    n = space dimension
    r = number of basis vectors
    p = number of point measurements
    """
    
    n,r = Psi.shape
    C = np.zeros((p,n))
    Q,R,P = linalg.qr(Psi.T,pivoting=True,mode='economic')
    positions = np.sort(P[:p])
    for i in range(p):
        C[i,positions[i]] = 1
    return C, positions
#%% Convex optimization
"""
Theta_i = C_i Psi
argmax log det sum_i^n beta_i Theta_i.T Theta_i
select p largest betas
"""
def function_i(Theta):
    f = []
    for i in range(Theta.shape[0]):
        f_i = Theta[i,:].reshape((Theta.shape[1],1)) @ Theta[i,:].reshape((1,Theta.shape[1])) # Theta_i.T @ Theta_i
        f.append(f_i)
    return f

def ConvexOpt_homoscedasticity(Psi,p):
    """
    n = space dimension
    r = number of basis vectors
    p = number of point measurements
    """
    n,r = Psi.shape
    C = np.identity(n) # all canonical vectors in R^n
    Theta = C @ Psi # useless - it's here only to clarify
    #beta_domain = np.arange(0,1+1e-2,1e-2)
    f = function_i(Theta)
    
    # problem data
    beta = cp.Variable((n,),pos=True,name='beta')
    F = cp.sum([beta[i]*f[i] for i in range(n)],axis=0)
    objective = cp.Minimize(-1*cp.log_det(F)) 
    constraints = [
        beta >= np.zeros((n,)), # beta_i >= 0
        beta <= np.ones(n), # beta_i <= 1
        cp.sum(beta) == p, # sum betas == num_sensors
        ]
    problem = cp.Problem(objective, constraints)
    
    problem.is_dgp()
    problem.solve(gp=False)
    
    # largest p values
    locations = np.sort(np.argpartition(beta.value,-p)[-p:])
    beta_vals = [beta.value[i] for i in locations]
    print(f'{p} optimal sensor locations found:\nlocations: {locations}\nbeta values: {beta_vals}')
    
    C = np.zeros((p,n))
    for i in range(C.shape[0]):
        C[i,locations[i]] = 1
    
    
    return C,locations,beta.value

def ConvexOpt(Psi,p,k=[],sensors = []):
    """
    Variant medthod for heteroscedasticity in the measurements: different type of sensors.
    
    sensors = number of sensors of each class such that classes = len(sensors)
    n = space dimension
    r = number of basis vectors
    p = number of point measurements
    k = ratio between sigma_1/sigma_i <<1
    """
    classes = len(sensors)
    if sensors[0] == 0 or sensors[0] == p: # homoscedasticity: only one class of sensors
        print('Only 1 class of sensors. Solving homoscedastic problem.')
        C,locations,beta_value = ConvexOpt_homoscedasticity(Psi, p)
        if sensors[0] == 0:
            loc = []
            loc.append([])
            loc.append(locations)
            return C,loc,beta_value
        elif sensors[1]==0:
            loc = []
            loc.append(locations)
            loc.append([])
            return C,loc,beta_value
        return C,locations,beta_value
    
    else:
        print(f'{len(sensors)} types of sensors with variance ratios {k}. Solving heteroscedastic problem.')
        
        n, r = Psi.shape
        C = np.identity(n) # canonical vectors in R^n
        Theta = C @ Psi # just to enforce the idea
        f = function_i(Theta)
        
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
        
        locations = []
        #comparison = np.argost(beta.value)
        # first select places based on Refeerence sstations (entry 0), then LCSs (entry 1)
        for s in range(classes):
            arr = beta.value[:,s].copy()
            if s==0:# place reference stations
                loc = np.sort(np.argpartition(arr,-sensors[0])[-sensors[0]:])
            if s==1:# avoid placing LCSs where there is already a Reference Station
                loc_rs = locations[-1]
                arr[loc_rs] = 0.0
                loc = np.sort(np.argpartition(arr,-sensors[1])[-sensors[1]:])
            locations.append(loc)
                
                
        print(f'{p} optimal sensor placement found:\nlocations: {locations}\nbeta values:\n {beta.value}')
        C = np.zeros((p,n))
        for i in range(C.shape[0]):
                C[i,np.sort(np.concatenate(locations))[i]] = 1
    
    return C,locations,beta.value

def ConvexOpt_limit(Psi,sigma1,sigma2,p1,p2,lambda_reg = 0.0):
    """
    Convex optimization for sensor placement problem
    """
    n,r = Psi.shape
    H1 = cp.Variable(n,nonneg=True,value=np.ones(n))
    H2 = cp.Variable(n,nonneg=True,value=np.ones(n))
    Phi1 = Psi.T@cp.diag(H1)@Psi
    Phi2 = Psi.T@cp.diag(H2)@Psi
    
    if p1!=0 and p2!= 0: # LCS and RefSt
        obj = cp.log_det( (1/sigma1)*Phi1 + (1/sigma2)*Phi2 )
        constraints = [
            0<=H1,
            0<=H2,
            H1<=1,
            H2<=1,
            H1+H2 <=1,
            cp.sum(H1) == p1,
            cp.sum(H2) == p2
            ]
        
    elif p2 == 0 :
        print('0 reference stations')
        obj = cp.log_det((1/sigma1)*Phi1)
        constraints = [
            0<=H1,
            H1<=1,
            cp.sum(H1) == p1,
            ]
        
    elif p1 == 0:
        print('0 LCSs')
        obj = cp.log_det((1/sigma2)*Phi2)
        constraints = [
            0<=H2,
            H2<=1,
            cp.sum(H2) == p2
            ]
        
        
    
    
    if lambda_reg > 0.0 and p1!=0:
        print(f'Regularized optimization problem with lambda = {lambda_reg}')
        obj2 = (1/sigma2)*cp.trace(Phi2)
        problem = cp.Problem(cp.Minimize(-1*obj - lambda_reg*obj2),constraints)
    else:
        problem = cp.Problem(cp.Minimize(-1*obj),constraints)
    
    problem.solve(verbose=True,max_iters=10000000)
    H1_optimal = H1.value
    H2_optimal = H2.value
    
    return H1_optimal,H2_optimal,problem.value

def ConvexOpt_nuclearNorm(Psi,p2):
    n,r = Psi.shape
    H2 = cp.Variable(n,nonneg=True,value=np.ones(n))
    Phi2 =  Psi.T@cp.diag(H2)@Psi
    objective = cp.trace(Psi@Psi.T@cp.diag(H2))#cp.norm(Phi2,'nuc')
    constraints = [
        0<=H2,
        H2<=1,
        cp.sum(H2)==p2
        ]
    problem = cp.Problem(cp.Minimize(objective),constraints)
    problem.solve()
    H2_optimal = H2.value
    Phi2_optimal = Phi2.value
    
    return H2_optimal,Phi2_optimal

def ConvexOpt_2classes(Psi,lambda_reg,sigma1,sigma2,p1,p2):
    n,r = Psi.shape
    H1 = cp.Variable(n,nonneg=True,value=np.ones(n))
    H2 = cp.Variable(n,nonneg=True,value=np.ones(n))
    Phi1 = Psi.T@cp.diag(H1)@Psi
    Phi2 = Psi.T@cp.diag(H2)@Psi
    obj1 = (1/sigma1)*cp.log_det(Phi1)
    obj2 = (1/sigma2)*cp.trace(Phi2)
    constraints = [
        0<=H1,
        0<=H2,
        H1<=1,
        H2<=1,
        H1+H2 <=1,
        cp.sum(H1) == p1,
        cp.sum(H2) == p2
        ]
    problem = cp.Problem(cp.Minimize(-1*obj1 - lambda_reg*obj2),constraints)
    problem.solve()
    H1_optimal = H1.value
    H2_optimal = H2.value
    Phi1_optimal = Phi1.value
    Phi2_optimal = Phi2.value
    
    return H1_optimal,H2_optimal,Phi1_optimal,Phi2_optimal
#%%
def main():
    pass

if __name__ == '__main__':
    pass
