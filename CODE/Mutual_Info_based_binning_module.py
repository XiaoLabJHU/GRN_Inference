"""
This module contains functions to calculate entropies and mutual information
based on fixed binning (discretizing continuous data).
Mutual Information estimators include: Shannon, Miller-Madow.
Default results are in [nats] = natural log using np.log().

Written by Lior Shachaf 2021-02-09.
"""

import numpy as np


def entropy(X, bins_num, mi_est):
    """
    Calculate 1D entropy by discretizing (binning) the input data
    and approximating the probability function by calculating the frequency.
    The entropy is then calculated by Shannon's formula or Miller-Madow correction
    (Entropy + ({non empty bins_num}-1)/(2*N)).

    Parameters:
    X (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated entropy.
    """
    hist1dvar, bin_edges1d = np.histogram(X, bins=bins_num, density=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        product = hist1dvar / hist1dvar.sum() * np.log(hist1dvar / hist1dvar.sum())
        product[np.isnan(product)] = 0

    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1) / (2 * len(X))


def entropy2var(X, Y, bins_num, mi_est):
    """
    Calculate 2D entropy by discretizing (binning) the input data and
    approximating the probability function by calculating the frequency.

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated entropy.
    """
    hist2d, xedges, yedges = np.histogram2d(X, Y, bins=bins_num, density=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        product = hist2d / hist2d.sum() * np.log(hist2d / hist2d.sum())
        product[np.isnan(product)] = 0

    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1) / (2 * len(X))


def entropy3var(X, Y, Z, bins_num, mi_est):
    """
    Calculate 3D entropy by discretizing (binning) the input data
    and approximating the probability function by calculating the frequency.

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    Z (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated entropy.
    """
    hist3d, edges = np.histogramdd([X, Y, Z], bins=(bins_num, bins_num, bins_num), density=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        product = hist3d / hist3d.sum() * np.log(hist3d / hist3d.sum())
        product[np.isnan(product)] = 0

    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1) / (2 * len(X))


def two_way_info(X, Y, bins_num, mi_est, normalize=False):
    """
    Calculate two-way mutual information.

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.
    normalize (bool): If True, return normalized mutual information.

    Returns:
    float: Calculated two-way mutual information.
    """
    h_x = entropy(X, bins_num, mi_est)
    h_y = entropy(Y, bins_num, mi_est)
    h_xy = entropy2var(X, Y, bins_num, mi_est)
    if normalize:
        return (h_x + h_y - h_xy) / max(h_x, h_y)
    else:
        return h_x + h_y - h_xy


def conditional_mutual_info(X, Y, Z, bins_num, mi_est):
    """
    Calculate conditional mutual information I(X;Y|Z).

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    Z (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated conditional mutual information.
    """
    h_z = entropy(Z, bins_num, mi_est)
    h_zx = entropy2var(Z, X, bins_num, mi_est)
    h_yz = entropy2var(Y, Z, bins_num, mi_est)
    h_xyz = entropy3var(X, Y, Z, bins_num, mi_est)
    return h_zx + h_yz - h_xyz - h_z


def three_way_info(X, Y, Z, bins_num, mi_est):
    """
    Calculate three-way mutual information I(X,Y;Z).

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    Z (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated three-way mutual information.
    """
    h_z = entropy(Z, bins_num, mi_est)
    h_xy = entropy2var(X, Y, bins_num, mi_est)
    h_xyz = entropy3var(X, Y, Z, bins_num, mi_est)
    return h_z + h_xy - h_xyz


def total_corr(X, Y, Z, bins_num, mi_est):
    """
    Calculate total correlation.

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    Z (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated total correlation.
    """
    h_x = entropy(X, bins_num, mi_est)
    h_y = entropy(Y, bins_num, mi_est)
    h_z = entropy(Z, bins_num, mi_est)
    h_xyz = entropy3var(X, Y, Z, bins_num, mi_est)
    return h_x + h_y + h_z - h_xyz


def inter_info(X, Y, Z, bins_num, mi_est):
    """
    Calculate interaction information.

    Parameters:
    X (array-like): 1D array of floats.
    Y (array-like): 1D array of floats.
    Z (array-like): 1D array of floats.
    bins_num (int): Number of bins, can accept any value allowed by the histogram function.
    mi_est (str): MI estimator of choice. Can be Shannon (a.k.a naive, empirical) or Miller-Madow.

    Returns:
    float: Calculated interaction information.
    """
    h_x = entropy(X, bins_num, mi_est)
    h_y = entropy(Y, bins_num, mi_est)
    h_z = entropy(Z, bins_num, mi_est)
    h_xy = entropy2var(X, Y, bins_num, mi_est)
    h_xz = entropy2var(X, Z, bins_num, mi_est)
    h_yz = entropy2var(Y, Z, bins_num, mi_est)
    h_xyz = entropy3var(X, Y, Z, bins_num, mi_est)
    return -(h_x + h_y + h_z - (h_xy + h_xz + h_yz) + h_xyz)


def two_way_info_from_entropy(Ex, Ey, Exy, normalize=False):
    """
    Calculate two-way mutual information from entropies.

    Parameters:
    Ex (float): Entropy of X.
    Ey (float): Entropy of Y.
    Exy (float): Joint entropy of X and Y.
    normalize (bool): If True, return normalized mutual information.

    Returns:
    float: Calculated two-way mutual information.
    """
    if normalize:
        return (Ex + Ey - Exy) / max(Ex, Ey)
    else:
        return Ex + Ey - Exy


def conditional_mutual_info_from_entropy(Exz, Eyz, Exyz, Ez):
    """
    Calculate conditional mutual information I(X;Y|Z) from entropies.

    Parameters:
    Exz (float): Joint entropy of X and Z.
    Eyz (float): Joint entropy of Y and Z.
    Exyz (float): Joint entropy of X, Y, and Z.
    Ez (float): Entropy of Z.

    Returns:
    float: Calculated conditional mutual information.
    """
    return Exz + Eyz - Exyz - Ez


def three_way_info_from_entropy(Ez, Exy, Exyz):
    """
    Calculate three-way mutual information I(X,Y;Z) from entropies.

    Parameters:
    Ez (float): Entropy of Z.
    Exy (float): Joint entropy of X and Y.
    Exyz (float): Joint entropy of X, Y, and Z.

    Returns:
    float: Calculated three-way mutual information.
    """
    return Ez + Exy - Exyz


def total_corr_from_entropy(Ex, Ey, Ez, Exyz):
    """
    Calculate total correlation from entropies.

    Parameters:
    Ex (float): Entropy of X.
    Ey (float): Entropy of Y.
    Ez (float): Entropy of Z.
    Exyz (float): Joint entropy of X, Y, and Z.

    Returns:
    float: Calculated total correlation.
    """
    return Ex + Ey + Ez - Exyz


def inter_info_from_entropy(Ex, Ey, Ez, Exy, Exz, Eyz, Exyz):
    """
    Calculate interaction information from entropies.

    Parameters:
    Ex (float): Entropy of X.
    Ey (float): Entropy of Y.
    Ez (float): Entropy of Z.
    Exy (float): Joint entropy of X and Y.
    Exz (float): Joint entropy of X and Z.
    Eyz (float): Joint entropy of Y and Z.
    Exyz (float): Joint entropy of X, Y, and Z.

    Returns:
    float: Calculated interaction information.
    """
    return -(Ex + Ey + Ez - (Exy + Exz + Eyz) + Exyz)
