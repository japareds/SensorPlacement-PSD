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
from scipy.special import logsumexp
import scipy.linalg
import cvxpy as cp
import mosek
import warnings
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

def ConvexOpt_heteroscedasticity(Psi,p,k=[],sensors = []):
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
#%% Optimization
class ConvexOpt():
    """
    Convex optimization alternatives
    """
    def __init__(self,Psi,n,p,r):
        self.Psi = Psi
        self.n = n
        self.r = r
        self.p = p

    def ConvexOpt_limit(Psi,sigma1,sigma2,p1,p2,lambda_reg = 0.0):
        """
        Convex optimization for sensor placement problem
        """
        n,r = Psi.shape
        H1 = cp.Variable(n,nonneg=True,value=np.zeros(n))
        H2 = cp.Variable(n,nonneg=True,value=np.zeros(n))
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
        
        problem.solve(verbose=True,max_iters=100000)#10000000 iters for corvengence lambda 1000
        H1_optimal = H1.value
        H2_optimal = H2.value
        
        return H1_optimal,H2_optimal,problem.value

    def ConvexOpt_LMI(self,threshold_t=0.0,var_beta=1.0,rotate=False,theta=0.0,plane=0):
        """
        Semidefinite program solution to covariance minimizing
        using sparse LMIs
        t is bounded: t>sigma**2*r/p
        """
        p = self.p
        n,r = self.Psi.shape
        # variables
        t = cp.Variable(1,nonneg=True,value=10*np.ones(1))
        x = cp.Variable((n*p,1),nonneg=True,value=np.ones((n*p,1)))
        # matrices
        Bj = cp.bmat([[np.zeros((r,r)),np.zeros((r,1))],[np.zeros((1,r)),t[:,None]]])
        Ip = np.identity(p)
        R = [np.block([[self.Psi,np.zeros((n,1))],[np.zeros((p,r)),Ip[:,j][:,None]]]) for j in range(p) ]
        # central sparse matrix
        S = []
        for i in range(p):
            for j in range(n):
                C = np.zeros((p,n))
                C[i,j] = 1
                S_k = np.block([[C.T@C,C.T],[C,np.zeros((p,p))]])
                S.append(S_k)
        A_j = [R[j].T@S@R[j] for j in range(p)]
        # construct single LMI
        A_p = []
        for j in range(p):
            A_p.append(cp.sum([x[i]*A_j[j][i] for i in range(x.shape[0])]))
        
        LMI = []
        for j in range(p):
            LMI.append(cp.sum([A_p[j],Bj]))
            
        # add rotation to LMI
        if rotate:
            dim = (self.r+1)**2
            possible_vectors = np.array([np.array(i).T for i in itertools.combinations(np.identity(dim),2)])
            for j in range(p):
                LMI_vectorized = cp.reshape(LMI[j],shape=((r+1)**2,1))
                LMI_rotated = self.include_rotation(theta,plane,possible_vectors,LMI_vectorized)
                LMI[j] = cp.reshape(LMI_rotated,shape=(r+1,r+1))
                
            
    
        # single LMI 
        #B = cp.kron(np.eye(p),Bj)
        
        # linear constraints
        R1 = np.zeros((p,n*p))
        for i in range(p):
            #R1[(i-1),(i-1)*n:i*n] = np.ones(n)
            R1[i,i*n:(i*n+n)] = np.ones(n)
            
        R2 = np.tile(np.identity(n),p)
        # constraints
        constraints = []
        ## LMIs
        constraints += [LMI[i] >> 0 for i in range(p)]
        ## weights
        constraints += [np.zeros((n*p,1))<= x,
                        x<= np.ones(((n*p),1)),
                        cp.sum(x)==p,# sum of weights equal to number of sensors
                        R1@x == np.ones((p,1)),#sum of sensor in space is 1:== [cp.sum(x[(i-1)*n:i*n]) ==1 for i in range(1,n)]
                        R2@x >= np.zeros((n,1)),
                        R2@x <= np.ones((n,1))# sum of sensors weights at single location between 0 and 1
            ]
        if threshold_t>0:
            print(f'Adding convergence threshold: t>={threshold_t}')
            constraints += [t>=threshold_t]
        
        obj = cp.Minimize(t/var_beta)
        prob = cp.Problem(obj,constraints)
        if not prob.is_dcp():
            warnings.warn('Problem is not dcp')
        else:
            prob.solve(verbose=True)
        
        C = np.reshape(x.value,(p,n))
        H = np.zeros((n,n))
        np.fill_diagonal(H, (R2@x).value)
        

        self.H = H 
        self.C = C
        self.obj = prob.value
        
    def rotation_matrix(self,u,v,theta):
        """
        Rotates a vector in the R^(n-2) plane by an angle theta
        given two orthogonal vectors in R^n
        """
        L = v@u.T - u@v.T
        rotation = scipy.linalg.expm(theta*L)
        return rotation
     
    def include_rotation(self,theta,plane,possible_vectors,M):
        """
        Include rotation on the r+1 PSD polytope
        """
        
        rotation_vectors = possible_vectors[plane]
        return self.rotation_matrix(rotation_vectors[:,0][:,None],rotation_vectors[:,1][:,None],theta)@M
        
    def k_swap(max_swaps):
        """
        Swap the k-th chosen location with the n-p remaining locations 
        until no swap increases the objective function
        """
        pass
        return

def diag_block_mat(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out




def ConvexOpt_SDP(Psi,p,threshold_t=0,var_beta = 1.0):
    """
    Semidefinite program solution for the sensor placement problem
    Only one class of sensors (homoscedastic network)
    """
    n,r = Psi.shape
    t = cp.Variable(1,nonneg=True)
    C = cp.Variable((p,n),nonneg=True,value=np.zeros((p,n)))
    H = cp.Variable((n,n),nonneg=True,value=np.zeros((n,n)))#C.T@C
    # Ip = cp.Constant(np.identity(p))
    # M = cp.bmat([[H,C.T],[C,Ip]])
    M = []
    M1 = Psi.T@H@Psi
    M2 = Psi.T@C.T
    
    for i in range(p):
        M.append(cp.bmat([[M1,M2[:,i][:,None]],[(M2[:,i][:,None]).T,t[None]]]))
      
    constraints = []
    constraints += [M[i]>>0 for i in range(p)]
    # weights constraints
    constraints += [cp.diag(cp.vec(C))>=np.zeros((p*n,p*n)),
                    cp.diag(cp.vec(C))<=np.identity(p*n),
                    cp.diag(H)>=np.zeros(n),
                    cp.diag(H)<=np.ones(n),
                    cp.sum(cp.diag(H))==p,
                    cp.sum(C,axis=1)==1,
                    cp.sum(C,axis=0)<=1,
                    cp.diag(H) == cp.sum(C,axis=0),
                    H - cp.diag(cp.diag(H)) == 0
                    ]
    if threshold_t>0:
        constraints += [t>=threshold_t]
    
    problem = cp.Problem(cp.Minimize(t/var_beta),constraints)
    problem.solve(verbose=True)
    
    return H.value,C.value,problem.value


def ConvexOpt_SDP_regressor(Psi,p):
    """
    Semidefinite program solution for the sensor placement problem
    Only one class of sensors (homoscedastic netowrk)
    """
    n,r = Psi.shape
    t = cp.Variable(1,nonneg=True)
    H = cp.Variable((n),nonneg=True,value=np.zeros((n)))#C.T@C
    # Ip = cp.Constant(np.identity(p))
    # M = cp.bmat([[H,C.T],[C,Ip]])
    M = []
    M1 = Psi.T@cp.diag(H)@Psi
    M2 = np.identity(r)
    
    for i in range(r):
        M.append(cp.bmat([[M1,M2[:,i][:,None]],[(M2[:,i][:,None]).T,t[None]]]))
      
    constraints = []
    constraints += [M[i]>>0 for i in range(r)]
    # weights constraints
    constraints += [cp.diag(H)>=0,
                    cp.diag(H)<=1,
                    cp.sum(cp.diag(H))==p
                    ]
    
    problem = cp.Problem(cp.Minimize(t),constraints)
    problem.solve(verbose=True)
    
    return np.diag(H.value),problem.value

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
