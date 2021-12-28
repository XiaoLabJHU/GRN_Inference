#!/usr/bin/env python3.8
"""
This module contains functions to calculate entropies and mutual information based on k-nearest neighbor.
Results are in [nats] = natural log using np.log()
Written by Lior Shachaf 2020-03-30
Updates:
2020-11-11: replaced MI & TC algo by version x2 faster
2021-05-13: Improving MI & TC algo speed by replacing the loop over query_ball_point with an array of points and later looping to remove the edge points
"""

import numpy as np
from scipy import spatial
from scipy import special

def Two_way_info_from_entropy(Ex,Ey,Exy):
    return (Ex + Ey - Exy)

def Two_way_info_norm_from_entropy(Ex,Ey,Exy):
    return (Ex + Ey - Exy)/max(Ex,Ey)

def Conditional_mutual_info_from_entropy(Exz,Eyz,Exyz,Ez): #I(X;Y|Z)
    return (Exz + Eyz - Exyz - Ez)

def Three_way_info_from_entropy(Ez,Exy,Exyz): #I(X,Y;Z)
    return (Ez + Exy - Exyz)

def Total_Corr_from_entropy(Ex,Ey,Ez,Exyz):
    return (Ex + Ey + Ez - Exyz)

def Inter_Info_from_entropy(Ex,Ey,Ez,Exy,Exz,Eyz,Exyz):
    return -(Ex + Ey + Ez - (Exy + Exz + Eyz) + Exyz)

## Algo 1&2 from: Kraskov, Stogbauer and Grassberger - Estimating mutual information
def Entropy_KNN_eq20(k,Ntot,distance_i_array,dimension):
    """Entropy calc for vector with d dimensions. distance_i_array = Chebyshev distance vector from u_i to k-th neighbor""" 
    c_d = 1 # for Chebyshev distance metric not Euclidean
    E_v1 = - special.digamma(k) + special.digamma(Ntot) + np.log(c_d) + dimension * np.average(np.log(2 * distance_i_array))
    return E_v1

def Entropy_KNN_eq22(n_i_array,Ntot,distance_i_array,dimension): 
    c_d = 1 # for Chebyshev distance metric not Euclidean
    E_v2 = - np.average(special.digamma(n_i_array + 1)) + special.digamma(Ntot) + np.log(c_d) + dimension * np.average(np.log(2 * distance_i_array))
    return E_v2

def MI_KNN_calc1(k,Ntot,n_x,n_y):
    I = special.digamma(k) - np.average(special.digamma(n_x + 1) + special.digamma(n_y + 1)) + special.digamma(Ntot) 
    return I

def TC_KNN_calc1(k,Ntot,n_x,n_y,n_z):
    I3 = special.digamma(k) - np.average(special.digamma(n_x + 1) + special.digamma(n_y + 1) + special.digamma(n_z + 1)) + 2 * special.digamma(Ntot) 
    return I3

def MI_KNN_calc2(k,Ntot,n_x,n_y):
    I = special.digamma(k) - 1/k - np.average(special.digamma(n_x) + special.digamma(n_y)) + special.digamma(Ntot) 
    return I


def Entropy_KNN_KDtree_algo(vec1,k_max):
    dimension = 1 
    Entropy_k = np.zeros(k_max)
    Ntot = len(vec1)

    tree_x = spatial.cKDTree(np.c_[vec1])
    distance_array = tree_x.query(np.c_[vec1], k_max+2, p=np.inf)[0]
    
    for k in range(1,k_max+1):
        Entropy_k[k-1] = Entropy_KNN_eq20(k,Ntot,distance_array[:,k],dimension)
        #Entropy_k[k-1] = Entropy_KNN_eq22(k*np.ones(Ntot),Ntot,distance_array[:,k+1],dimension)
        
    return Entropy_k


def Entropy2D_KNN_KDtree_algo(vec1,vec2,k_max):
    dimension = 2 
    Entropy2D_k = np.zeros(k_max)
    Ntot = len(vec1)

    pts = np.c_[vec1.ravel(), vec2.ravel()]
    tree = spatial.cKDTree(pts) 
    distance_array = tree.query(pts, k_max+2, p=np.inf)[0]
    for k in range(1,k_max+1):
        Entropy2D_k[k-1] = Entropy_KNN_eq20(k,Ntot,distance_array[:,k],dimension)
        #Entropy2D_k[k-1] = Entropy_KNN_eq22(k*np.ones(Ntot),Ntot,distance_array[:,k+1],dimension)

    return Entropy2D_k


def Entropy3D_KNN_KDtree_algo(vec1,vec2,vec3,k_max):
    dimension = 3 
    Entropy3D_k = np.zeros(k_max)
    Ntot = len(vec1)

    pts = np.c_[vec1.ravel(), vec2.ravel(), vec3.ravel()]
    tree = spatial.cKDTree(pts) 
    distance_array = tree.query(pts, k_max+2, p=np.inf)[0]
    for k in range(1,k_max+1):
        Entropy3D_k[k-1] = Entropy_KNN_eq20(k,Ntot,distance_array[:,k],dimension)
        #Entropy3D_k[k-1] = Entropy_KNN_eq22(k*np.ones(Ntot),Ntot,distance_array[:,k+1],dimension)

    return Entropy3D_k

def MI_KNN_KDtree_algo(vec1,vec2,k_max):

    Ntot = len(vec1) # Assuming all vectors has the same length
    MI_k = np.zeros(k_max)

    pts = np.c_[vec1.ravel(), vec2.ravel()]
    tree = spatial.cKDTree(pts)
    tree_x = spatial.cKDTree(np.c_[vec1])
    tree_y = spatial.cKDTree(np.c_[vec2])
    distance_array, pts_locations = tree.query(pts, k_max+1, p=np.inf)

    for k in range(1,k_max+1):
        n_x = np.zeros(Ntot, dtype = int); n_y = np.zeros(Ntot, dtype = int);
        n_x = tree_x.query_ball_point(pts[:,0].reshape(-1, 1), distance_array[:,k], p=1, return_sorted=False, return_length=True) - 1 #Removing self                     
        n_y = tree_y.query_ball_point(pts[:,1].reshape(-1, 1), distance_array[:,k], p=1, return_sorted=False, return_length=True) - 1 #Removing self
                
        for counter, point in enumerate(pts):
                edge_point_axis = np.argmax([abs(point[0]-pts[pts_locations[counter,k]][0]), abs(point[1]-pts[pts_locations[counter,k]][1])])
                if edge_point_axis == 0: #'x'
                    n_x[counter] -= 1 #Removing edge points                    
                else: #'y'
                    n_y[counter] -= 1 #Removing edge points
        
        MI_k[k-1] = MI_KNN_calc1(k,Ntot,n_x,n_y)

    return MI_k

def TC_KNN_KDtree_algo(vec1,vec2,vec3,k_max):

    TC_k = np.zeros(k_max)

    pts = np.c_[vec1.ravel(), vec2.ravel(), vec3.ravel()]
    tree = spatial.cKDTree(pts)
    tree_x = spatial.cKDTree(np.c_[vec1])
    tree_y = spatial.cKDTree(np.c_[vec2])
    tree_z = spatial.cKDTree(np.c_[vec3])
    distance_array, pts_locations = tree.query(pts, k_max+1, p=np.inf)

    Ntot = len(vec1) # Assuming all vectors has the same length

    for k in range(1,k_max+1):
        n_x = np.zeros(Ntot, dtype = int); n_y = np.zeros(Ntot, dtype = int); n_z = np.zeros(Ntot, dtype = int);
        n_x = tree_x.query_ball_point(pts[:,0].reshape(-1, 1), distance_array[:,k], p=1, return_sorted=False, return_length=True) - 1 #Removing self                     
        n_y = tree_y.query_ball_point(pts[:,1].reshape(-1, 1), distance_array[:,k], p=1, return_sorted=False, return_length=True) - 1 #Removing self
        n_z = tree_z.query_ball_point(pts[:,2].reshape(-1, 1), distance_array[:,k], p=1, return_sorted=False, return_length=True) - 1 #Removing self

        for counter, point in enumerate(pts):
                edge_point_axis = np.argmax([abs(point[0]-pts[pts_locations[counter,k]][0]), abs(point[1]-pts[pts_locations[counter,k]][1]), abs(point[2]-pts[pts_locations[counter,k]][2])])
                if edge_point_axis == 0: #'x'
                    n_x[counter] -= 1 #Removing edge points                    
                elif edge_point_axis == 1: #'y'
                    n_y[counter] -= 1 #Removing edge points
                else: #'z'
                    n_z[counter] -= 1 #Removing edge points

        TC_k[k-1] = TC_KNN_calc1(k,Ntot,n_x,n_y,n_z)

    return TC_k
