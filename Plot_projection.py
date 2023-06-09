#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:32:39 2023

@author: jparedes
"""
import numpy as np
from scipy import linalg
import itertools
from matplotlib import pyplot as plt
import warnings
import decimal
import SensorPlacementMethods as SPM
import Plots
#%%
def get_lmi_sparse(S,n,p):
    tol = 5e-2
    x_range = np.arange(0.,1+tol,tol)
    # vector constraints
    R1 = np.zeros((p,n*p))
    for i in range(p):
        R1[i,i*n:(i*n+n)] = np.ones(n)
        
    R2 = np.tile(np.identity(n),p)
    
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
    # check PSD of LMIs
    # if not any([(np.linalg.eig(LMI[i])[0]>=0).all() for i in range(LMI.shape[0])]):
    #    warnings.warn('Non-PSD LMI') 
    
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
def create_psi_basis(n,r,basis='identity'):
    """
    Create a basis of eigenmodes
    """
    if basis=='identity':
        print('Identity basis')
        Psi = np.identity(n)[:,:r]
    elif basis=='random':
        rng = np.random.default_rng(seed=92)
        Psi = linalg.orth(rng.random((n,n)))[:,:r]
    elif basis=='rotation_x':
        if n == 3:
            rotation = np.block([[1,0,0],[0,np.cos(np.pi/4),-np.sin(np.pi/4)],[0,np.sin(np.pi/4),np.cos(np.pi/4)]])
        elif n==2:
            rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
        Psi = (rotation @ np.identity(n))[:,:r]
    elif basis=='rotation_z':
        if n==3:
            rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4),0],[np.sin(np.pi/4),np.cos(np.pi/4),0],[0,0,1]])
            
        elif n==2:
            rotation = np.block([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
        Psi = (rotation @ np.identity(n))[:,:r]
        
    return Psi
#%%
def check_solution(C,H,Psi,t,S,r,n,p):
    """
    Given C,H,t solutions of the convex optimization problem.
    Compute the covariance matrix and check PSD
    """
    cov = C@Psi@np.linalg.inv(Psi.T@H@Psi)@Psi.T@C.T
    print(f'Maximum value of diagonal entries of covariance matrix: {np.diag(cov).max()}')
    print(f'objective value for t={t}')
    
    B = np.zeros((r+1,r+1))
    B[-1,-1] = t

    x = np.tile(C_t[float(str(t))][0,:],p)
    LMI_sparse_solution = np.array([x[i]*S[i] for i in range(n*p)]).sum(axis=0)
    Ip = np.identity(p)
    R = [np.block([[Psi,np.zeros((n,1))],[np.zeros((p,r)),Ip[:,j][:,None]]]) for j in range(p)]
    LMI_solution = [R[i].T@LMI_sparse_solution@R[i] + B for i in range(p)]# p LMIs per configuration
    if all (np.concatenate([np.linalg.eig(LMI)[0] for LMI in LMI_solution])>=0):
        print(f'PSD matrix\neigenvalues: {np.concatenate([np.linalg.eig(LMI)[0] for LMI in LMI_solution])}')
    elif any(np.concatenate([np.linalg.eig(LMI)[0] for LMI in LMI_solution])<0):
        print(f'non-PSD matrix\neigenvalues: {np.concatenate([np.linalg.eig(LMI)[0] for LMI in LMI_solution])}')
    return 
    


#%%
# =============================================================================
# Parameters
# =============================================================================
n,p,r = 3,2,1
Psi = create_psi_basis(n,r,basis='random')

x = np.ones(p*n)


# =============================================================================
# Get LMIs
# =============================================================================

# sparse matrices
S = []
for i in range(p):
    for j in range(n):
        C = np.zeros((p,n))
        C[i,j] = 1
        S_k = np.block([[C.T@C,C.T],[C,np.zeros((p,p))]])
        S.append(S_k)

LMI = []
LMI,LMI_q,x_valid = get_lmi_sparse(S,n,p)

t_tol = 1e-3
exponent = -1*decimal.Decimal(str(t_tol)).as_tuple().exponent
t_range = np.concatenate([np.arange(0.,r/p+t_tol,t_tol),np.arange(r/p + t_tol,2*r/p+t_tol,t_tol)])
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
    print(t)
    B = B_t[t]
    x_PSD_t[t] = get_lmi_non_sparse(LMI,x_valid,Psi,n,p,r,B)
    
# =============================================================================
# Solve ConvexOpt problem
# =============================================================================
print(f'Solving convex optimization problem at different t values:\n{[np.round(i,2) for i in t_range]}\nPoint of interest at r/p = {r/p}')
H_t={np.round(el,exponent):0 for el in t_range}
C_t={np.round(el,exponent):0 for el in t_range}
obj_t = {np.round(el,exponent):0 for el in t_range}

convex_opt = SPM.ConvexOpt(Psi,n,p,r)
for t in t_range:
    t = np.round(t,exponent)
    convex_opt.ConvexOpt_LMI(threshold_t=t,var_beta=1.0)
    C,H,obj = convex_opt.C, convex_opt.H,convex_opt.obj
    H_t[t] = H
    C_t[t] = C
    obj_t[t] = obj

t_check= 0.476
check_solution(C_t[t_check],H_t[t_check],Psi,obj_t[t_check],S,r,n,p)


#%%
# =============================================================================
# Plot projections
# =============================================================================

# projection in space
t_plot = 0.476
check_solution(C_t[t_plot],H_t[t_plot],Psi,obj_t[t_plot],S,r,n,p)
figures = Plots.Plots_Projection(n, p, r, Psi)
figures.plot_count_non_PSD(LMI,x_PSD_t)
# linear plots
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='x-linear')

figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag1-mixed')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag2-mixed')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag3-mixed')

figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag12-mixed')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag13-mixed')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag23-mixed')

# quadratic plots
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag-squared')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag1-mixed-squared')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag2-mixed-squared')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag3-mixed-squared')

figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag12-mixed-squared')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag13-mixed-squared')
figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t_plot,projection='diag23-mixed-squared')
#%% projection in time: both in sparse space and after mapping
for t in [0.5,0.493,0.488,0.484,0.481,0.480,0.478,0.477,0.476,0.475]:# different levels of PSD-ness
    print('\n')
    t = np.round(t,exponent)
    print(f'{t}')
    #check_solution(C_t[t],H_t[t],Psi,obj_t[t],S,r,n,p)
    figures.plot_projection(LMI, LMI_q, H_t, C_t, x_PSD_t, t,projection='x-linear')
    figures.plot_projection_Schur(x_PSD_t,B_t,C_t,H_t,t)

# plot projections
# fig,fig1,fig2,fig3 = plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.5,projection='diag'),\
# plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n1'),\
#     plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n2'),\
#         plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n3')
#fig,fig1 = plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.5,projection='n1')

