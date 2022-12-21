#!/usr/bin/env python3
"""
This module contains functions to calculate entropies and mutual information based on fixed binning (discritizing continuous data).
Mutual Information estimators include: Shannon, Miller-Madow
Default results are in [nats] = natural log using np.log()
Written by Lior Shachaf 2021-02-09
"""

import numpy as np

def Entropy(X,bins_num,mi_est): 
    """ This func. calculates 1D entropy by discretizing (binning) the input data and approximating the probability func. by calculating the frequency.
    The entropy is then calculated by Shannon's formula or Miller-Madow correction (Entropy + ({non empty bins_num}-1)/(2*N)).
    X input is 1D array of floats.
    bins_num = number of bins, and can accept any value allowed by the histogram func.
    mi_est = MI estimator of choice. Can be Shannon's (a.k.a naive, empirical) or Miller-Madaw
    """
    hist1dvar, bin_edges1d = np.histogram(X, bins=bins_num, density=False)
    product = hist1dvar/hist1dvar.sum()*np.log(hist1dvar/hist1dvar.sum())
    product[np.isnan(product)] = 0
    if mi_est == "Shannon":    
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1)/(2*len(X))


def Entropy2var(X,Y,bins_num,mi_est): # each arg is array of floats
    hist2d, Xedges, Yedges = np.histogram2d(X, Y, bins=bins_num, density=False)
    product = hist2d/hist2d.sum()*np.log(hist2d/hist2d.sum())
    product[np.isnan(product)] = 0
    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1)/(2*len(X))    


def Entropy3var(X,Y,Z,bins_num,mi_est): # each arg is array of floats
    XYZ = [X, Y, Z]
    hist3d, edges = np.histogramdd(XYZ, bins = (bins_num, bins_num, bins_num), density=False)
    product = hist3d/hist3d.sum()*np.log(hist3d/hist3d.sum())
    product[np.isnan(product)] = 0    
    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1)/(2*len(X))


def Two_way_info(X,Y,bins_num,mi_est):
    return (Entropy(X,bins_num,mi_est) + Entropy(Y,bins_num,mi_est) - Entropy2var(X,Y,bins_num,mi_est))


def Two_way_info_norm(X,Y,bins_num,mi_est):
    return (Entropy(X,bins_num,mi_est) + Entropy(Y,bins_num,mi_est) - Entropy2var(X,Y,bins_num,mi_est))/max(Entropy(X,bins_num,mi_est),Entropy(Y,bins_num,mi_est))


def Conditional_mutual_info(X,Y,Z,bins_num,mi_est): #I(X;Y|Z)
    return (Entropy2var(Y,Z,bins_num,mi_est) + Entropy2var(Z,X,bins_num,mi_est) - Entropy3var(X,Y,Z,bins_num,mi_est) - Entropy(Z,bins_num,mi_est))


def Three_way_info(X,Y,Z,bins_num,mi_est): #I(X,Y;Z)
    return (Entropy(Z,bins_num,mi_est) + Entropy2var(X,Y,bins_num,mi_est) - Entropy3var(X,Y,Z,bins_num,mi_est))


def Total_Corr(X,Y,Z,bins_num,mi_est):
    return (Entropy(X,bins_num,mi_est) + Entropy(Y,bins_num,mi_est) + Entropy(Z,bins_num,mi_est) - Entropy3var(X,Y,Z,bins_num,mi_est))


def Inter_Info(X,Y,Z,bins_num,mi_est):
    return -(Entropy(X,bins_num,mi_est) + Entropy(Y,bins_num,mi_est) + Entropy(Z,bins_num,mi_est) - (Entropy2var(X,Y,bins_num,mi_est) + Entropy2var(Y,Z,bins_num,mi_est) + Entropy2var(Z,X,bins_num,mi_est)) + Entropy3var(X,Y,Z,bins_num,mi_est))

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
