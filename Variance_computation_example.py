#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:46:55 2023

@author: jparedes
"""

import numpy as np

r = 5
p1 = 3
p2 = 1
p = p1+p2

s1 = 0.5
s2 = 0.2

Sigma1 = s1*np.identity(p1)
Sigma2 = s2*np.identity(p2)

Sigma = np.zeros(np.add(Sigma1.shape,Sigma2.shape))
Sigma[:Sigma1.shape[0],:Sigma1.shape[1]] = Sigma1
Sigma[Sigma1.shape[0]:,Sigma1.shape[1]:] = Sigma2

Theta = np.random.rand(p,r)
Theta1 = np.zeros((p1,r))
Theta1 = Theta[:Theta1.shape[0],:]
Theta2 = np.zeros((p2,r))
Theta2 = Theta[Theta1.shape[0]:,:]

Sigma_a_full = Theta.T @ Sigma @ Theta
Sigma_a_reduced = (Theta1.T @ Sigma1 @ Theta1) + (Theta2.T @ Sigma2 @ Theta2)
print(f'Number of sensors of type 1: {p1}\nNumber of sensors of type 2: {p2}\nTotal number of sensors: {p}')
print(f'Number of eigenvectors kept: {r}')
print(f'Shape of matrices:\nTheta: {Theta.shape}\nSigma: {Sigma.shape}')
print(f'Shape of reduced matrices:\nTheta1 {Theta1.shape}\t Sigma1: {Sigma1.shape}\nTheta2: {Theta2.shape}\t Sigma2: {Sigma2.shape}')
print(f'Shape of First term in the reduced sum: {(Theta1.T @ Sigma1 @ Theta1).shape}\nShape of Second term in the reduced sum: {(Theta2.T @ Sigma2 @ Theta2).shape}')

print(f'Variance of estimator:\nFull:\n {Sigma_a_full}\nReduced:\n {Sigma_a_reduced}')
if np.allclose(Sigma_a_full,Sigma_a_reduced):
    print('Variances are equivalent in both methods')
