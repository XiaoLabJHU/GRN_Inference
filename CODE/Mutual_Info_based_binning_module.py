"""
This module contains functions to calculate entropies and mutual information
based on fixed binning (discretizing continuous data).
Mutual Information estimators include: Shannon, Miller-Madow.
Default results are in [nats] = natural log using np.log().

Written by Lior Shachaf 2021-02-09.
"""

import numpy as np


def entropy1var(X, bins_num, mi_est):
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
    h_x = entropy(X, bins_num=bins_num, mi_est=mi_est)
    h_y = entropy(Y, bins_num=bins_num, mi_est=mi_est)
    h_xy = entropy(X, Y, bins_num=bins_num, mi_est=mi_est)
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
    h_z = entropy(Z, bins_num=bins_num, mi_est=mi_est)
    h_zx = entropy(Z, X, bins_num=bins_num, mi_est=mi_est)
    h_yz = entropy(Y, Z, bins_num=bins_num, mi_est=mi_est)
    h_xyz = entropy(X, Y, Z, bins_num=bins_num, mi_est=mi_est)
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
    h_z = entropy(Z, bins_num=bins_num, mi_est=mi_est)
    h_xy = entropy(X, Y, bins_num=bins_num, mi_est=mi_est)
    h_xyz = entropy(X, Y, Z, bins_num=bins_num, mi_est=mi_est)
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
    h_x = entropy(X, bins_num=bins_num, mi_est=mi_est)
    h_y = entropy(Y, bins_num=bins_num, mi_est=mi_est)
    h_z = entropy(Z, bins_num=bins_num, mi_est=mi_est)
    h_xyz = entropy(X, Y, Z, bins_num=bins_num, mi_est=mi_est)
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
    h_x = entropy(X, bins_num=bins_num, mi_est=mi_est)
    h_y = entropy(Y, bins_num=bins_num, mi_est=mi_est)
    h_z = entropy(Z, bins_num=bins_num, mi_est=mi_est)
    h_xy = entropy(X, Y, bins_num=bins_num, mi_est=mi_est)
    h_xz = entropy(X, Z, bins_num=bins_num, mi_est=mi_est)
    h_yz = entropy(Y, Z, bins_num=bins_num, mi_est=mi_est)
    h_xyz = entropy(X, Y, Z, bins_num=bins_num, mi_est=mi_est)
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


def entropy(*arrays, bins_num=10, bins_or_neighbors=None, mi_est="Shannon"):
    """
    Calculate entropy by discretizing (binning) the input data and
    approximating the probability function by calculating the frequency.
    The entropy is then calculated by Shannon's formula or Miller-Madow correction.

    Parameters:
    *arrays (array-like): Variable number of 1D arrays (X, Y, Z).
    bins_num (int): Number of bins to use for discretizing the data. Default is 10.
    bins_or_neighbors (int): Number of bins, can accept any value allowed by the histogram function. Default is None.
    mi_est (str): MI estimator of choice. Can be "Shannon" (a.k.a naive, empirical) or "Miller-Madow". Default is "Shannon".

    Returns:
    float: Calculated entropy.

    Examples:
    # 1D entropy
    entropy_1d = entropy([1, 2, 3, 4, 5], bins_num=5, mi_est="Shannon")

    # 2D entropy
    entropy_2d = entropy([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], bins_or_neighbors=5, mi_est="Shannon")

    # 3D entropy
    entropy_3d = entropy([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 3, 4, 5, 6], bins_or_neighbors=5, mi_est="Shannon")
    """
    if bins_or_neighbors is not None:
        bins_num = bins_or_neighbors

    dimension = len(arrays)
    if dimension == 1:
        hist, bin_edges = np.histogram(arrays[0], bins=bins_num, density=False)
    elif dimension == 2:
        hist, xedges, yedges = np.histogram2d(arrays[0], arrays[1], bins=bins_num, density=False)
    elif dimension == 3:
        hist, edges = np.histogramdd(arrays, bins=(bins_num, bins_num, bins_num), density=False)
    else:
        raise ValueError("Only 1D, 2D, and 3D entropy calculations are supported.")

    with np.errstate(divide='ignore', invalid='ignore'):
        product = hist / hist.sum() * np.log(hist / hist.sum())
        product[np.isnan(product)] = 0

    if mi_est == "Shannon":
        return -np.sum(product)
    elif mi_est == "Miller-Madow":
        return -np.sum(product) + (np.count_nonzero(product) - 1) / (2 * len(arrays[0]))
    else:
        raise ValueError(f"Invalid mutual information estimator: {mi_est}")
