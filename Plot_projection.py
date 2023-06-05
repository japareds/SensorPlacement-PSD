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
#%%
def get_lmi_sparse(S,n,p):
    tol = 1e-1
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
    # check PSD of LMIs
    # if not any([(np.linalg.eig(LMI[i])[0]>=0).all() for i in range(LMI.shape[0])]):
    #    warnings.warn('Non-PSD LMI') 
    
    return LMI,LMI_q,x_valid

def get_lmi_non_sparse(LMI_sparse,x_valid,Psi,n,p,r,B):
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
def check_solution(C,Psi):
    
    #cov_H = C@Psi@np.linalg.inv(Psi.T@H@Psi)@Psi.T@C.T
    #np.linalg.eig(cov_H)[0]
    return C@Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T@C.T
    
#%% plots

def plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t,projection='diag'):
    if p!= 2 or n!=3:
        warnings.warn(f'The results are reliable for p=2 and n=3\nFound p={p} and n={n}')
    
    # non-PSD region
    x_non_PSD = np.array(x_PSD_t[t][1])# ==1 for non-PSD & 0 for PSD
    x_non_PSD = x_non_PSD.reshape((x_non_PSD.shape[0],p,n))
    x_PSD = np.array(x_PSD_t[t][0])# ==1 for non-PSD & 0 for PSD
    x_PSD = x_PSD.reshape((x_PSD.shape[0],p,n))
    print(f'At t={t} there are {x_non_PSD.shape[0]} non PSD elements out of {LMI.shape[0]}')
    
    # get projection points
    
    if projection == 'diag':# plot diagonal elements
        points = np.array([np.diag(LMI[i])[0:n] for i in range(LMI.shape[0])])
        points_q = np.array([np.diag(LMI_q[i])[0:n] for i in range(LMI.shape[0])])
        points_opt = np.array([np.diag(H) for H,_ in zip(H_t.values(),H_t.keys()) if _>=t])
        points_non_PSD = np.array([C.sum(axis=0) for C in x_non_PSD])
        points_non_PSD_q = np.array([(C**2).sum(axis=0) for C in x_non_PSD])
        
        # entry names for plot
        entry1 = '$h_{11}$'
        entry2 = '$h_{12}$'
        entry3 = '$h_{13}$'
        
    elif projection == 'n1':# get points at location 1
        points = np.array([np.concatenate([LMI[i][-p:,0],[LMI[i][0,0]]]) for i in range(LMI.shape[0])])
        points_q = np.array([np.concatenate([LMI[i][-p:,0],[LMI_q[i][0,0]]]) for i in range(LMI.shape[0])])
        points_opt = np.array([[C[0,0],C[1,0],np.diag(H)[0]] for H,C,_ in zip(H_t.values(),C_t.values(),C_t.keys()) if _>=t])
        points_non_PSD = np.array([[C[0,0],C[1,0],C.sum(axis=0)[0]]for C in x_non_PSD])
        points_PSD = np.array([[C[0,0],C[1,0],C.sum(axis=0)[0]]for C in x_PSD])
        
        
        points_non_PSD_q = np.array([[C[0,0],C[1,0],C_q.sum(axis=0)[0]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
        
        
        # entry names for plot
        entry1 = '$x_{11}$'
        entry2 = '$x_{21}$'
        entry3 = '$h_{11}$'
        
        
    elif projection == 'n2':# get points at location 2
        points = np.array([np.concatenate([LMI[i][-p:,1],[LMI[i][1,1]]]) for i in range(LMI.shape[0])])
        points_q = np.array([np.concatenate([LMI[i][-p:,1],[LMI_q[i][1,1]]]) for i in range(LMI.shape[0])])
        points_opt = np.array([[C[0,1],C[1,1],np.diag(H)[1]] for H,C,_ in zip(H_t.values(),C_t.values(),C_t.keys()) if _>=t])
        points_non_PSD = np.array([[C[0,1],C[1,1],C.sum(axis=0)[1]]for C in x_non_PSD])
        points_non_PSD_q = np.array([[C[0,1],C[1,1],C_q.sum(axis=0)[1]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
        # entry names for plot
        entry1 = '$x_{12}$'
        entry2 = '$x_{22}$'
        entry3 = '$h_{22}$'
    
    elif projection=='n3':# get points at location 3
        points = np.array([np.concatenate([LMI[i][-p:,2],[LMI[i][2,2]]]) for i in range(LMI.shape[0])])
        points_q = np.array([np.concatenate([LMI[i][-p:,2],[LMI_q[i][2,2]]]) for i in range(LMI.shape[0])])
        points_opt = np.array([[C[0,2],C[1,2],np.diag(H)[2]] for H,C,_ in zip(H_t.values(),C_t.values(),C_t.keys()) if _>=t])
        points_non_PSD = np.array([[C[0,2],C[1,2],C.sum(axis=0)[2]]for C in x_non_PSD])
        points_non_PSD_q = np.array([[C[0,2],C[1,2],C_q.sum(axis=0)[2]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
        # entry names for plot
        entry1 = '$x_{13}$'
        entry2 = '$x_{23}$'
        entry3 = '$h_{33}$'
    
        
    elif projection == 'n3_mixed':
        points = np.array([np.concatenate([[LMI[i][n,0]],[LMI[i][n+1,1]],[LMI[i][2,2]]]) for i in range(LMI.shape[0])])
        points_q = np.array([np.concatenate([[LMI_q[i][n,0]],[LMI_q[i][n+1,1]],[LMI_q[i][2,2]]]) for i in range(LMI.shape[0])])
        points_opt = np.array([[C[0,0],C[1,1],np.diag(H)[2]] for H,C,_ in zip(H_t.values(),C_t.values(),C_t.keys()) if _>=t])
        points_non_PSD = np.array([[C[0,0],C[1,1],C.sum(axis=0)[2]]for C in x_non_PSD])
        points_non_PSD_q = np.array([[C[0,0],C[1,1],C_q.sum(axis=0)[2]]for C,C_q in zip(x_non_PSD,x_non_PSD**2)])
        # entry names for plot
        entry1 = '$x_{11}$'
        entry2 = '$x_{22}$'
        entry3 = '$h_{33}$'
    
    
    fs=14
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot(111,projection='3d')
    #optimization results
    ax.plot3D(points_opt[:,0],points_opt[:,1],points_opt[:,2],color='#b03a2e',marker='o',markersize=10,label='optimization path')
    ax.scatter3D(points_opt[0,0],points_opt[0,1],points_opt[0,2],color='#1abc9c',marker='*',s=30,label='optimization result')
    #surfaces
    ax.plot_trisurf(points_q[:,0],points_q[:,1],points_q[:,2],color='#1a5276',linewidth=10)
    ax.plot_trisurf(points[:,0],points[:,1],points[:,2],color='orange',linewidth=100)
    
    #ax.scatter3D(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k',s=20,label='non-PSD region')
    
    ax.set_xlabel(entry1,fontsize=fs)
    ax.set_ylabel(entry2,fontsize=fs)
    ax.set_zlabel(entry3,fontsize=fs)
    
    ax.set_xticks([np.round(i,1) for i in np.arange(0.,1.2,0.2)])
    ax.set_yticks([np.round(i,1) for i in np.arange(0.,1.2,0.2)])
    ax.set_zticks([np.round(i,1) for i in np.arange(0.,1.2,0.2)])
    ax.set_title(f'{p} sensors\n projection: {projection}')
    ax.legend()
    try:#PSD and non-PSD points
        ax.scatter3D(points_non_PSD[:,0],points_non_PSD[:,1],points_non_PSD[:,2],color='k')
        ax.plot_trisurf(points_non_PSD_q[:,0],points_non_PSD_q[:,1],points_non_PSD_q[:,2],color='k')
        ax.plot_trisurf(points_PSD[:,0],points_PSD[:,1],points_PSD[:,2],color='w')
        fig.tight_layout()
    except:
        fig.tight_layout()
    
    
    
    return fig
#%%
# parameters
n,p,r = 3,2,1

x = np.ones(p*n)

# sparse matrices
S = []
for i in range(p):
    for j in range(n):
        C = np.zeros((p,n))
        C[i,j] = 1
        S_k = np.block([[C.T@C,C.T],[C,np.zeros((p,p))]])
        S.append(S_k)
# t matrix

#LMIs

# solutions
Psi = create_psi_basis(n,r,basis='random')
t_tol = 1e-1
round_entry = -decimal.Decimal(str(t_tol)).as_tuple().exponent
t_range = np.arange(0.,2+t_tol,t_tol)

B_t = {np.round(el,2):0 for el in t_range}
for t in t_range:
    t = np.round(t,2)
    B_j = np.zeros((r+1,r+1))
    B_j[-1,-1] = t
    B_t[t] = B_j

LMI = []
LMI,LMI_q,x_valid = get_lmi_sparse(S,n,p)

x_PSD_t = {np.round(el,2):0 for el in t_range}
for t in t_range:
    t = np.round(t,2)
    print(t)
    B = B_t[t]
    x_PSD_t[t] = get_lmi_non_sparse(LMI,x_valid,Psi,n,p,r,B)
    

H_t={np.round(el,2):0 for el in t_range}
C_t={np.round(el,2):0 for el in t_range}
obj_t = {np.round(el,2):0 for el in t_range}
for t in t_range:
    t = np.round(t,2)
    H,C,obj = SPM.ConvexOpt_LMI(Psi, p,threshold_t=t,var_beta=1.0)
    H_t[t] = H
    C_t[t] = C
    obj_t[t] = obj

# plot projections
fig,fig1,fig2,fig3 = plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.5,projection='diag'),\
plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n1'),\
    plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n2'),\
        plot_projection(LMI,LMI_q,p,n,H_t,C_t,x_PSD_t,t=0.0,projection='n3')


