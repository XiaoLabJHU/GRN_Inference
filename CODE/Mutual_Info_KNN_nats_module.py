"""
This module contains functions to calculate entropies and mutual information based on k-nearest neighbor.
Results are in [nats] = natural log using np.log()
Written by Lior Shachaf 2020-03-30
Updates:
2020-11-11: replaced MI & TC algo by version x2 faster.
2021-05-13: Improving MI & TC algo speed by replacing the loop over query_ball_point()
            with an array of points and later looping to remove the edge points.
"""

import numpy as np
from scipy import spatial
from scipy import special


def entropy_knn_eq20(k, n_tot, distance_i_array, dimension):
    """
    Entropy calculation for vector with d dimensions using k-NN.
    Algo 1 from: Kraskov, Stogbauer and Grassberger - Estimating mutual information

    Parameters:
    k (int): Number of nearest neighbors.
    n_tot (int): Total number of points.
    distance_i_array (array-like): Chebyshev distance vector from u_i to k-th neighbor.
    dimension (int): Dimension of the data.

    Returns:
    float: Calculated entropy.
    """
    c_d = 1  # for Chebyshev distance metric, not Euclidean
    e_v1 = (-special.digamma(k)
            + special.digamma(n_tot)
            + np.log(c_d)
            + dimension * np.average(np.log(2 * distance_i_array)))
    return e_v1


def entropy_knn_eq22(n_i_array, n_tot, distance_i_array, dimension):
    """
    Entropy calculation for vector with d dimensions using k-NN.
    Algo 2 from: Kraskov, Stogbauer and Grassberger - Estimating mutual information

    Parameters:
    n_i_array (array-like): Array of neighbor counts.
    n_tot (int): Total number of points.
    distance_i_array (array-like): Chebyshev distance vector from u_i to k-th neighbor.
    dimension (int): Dimension of the data.

    Returns:
    float: Calculated entropy.
    """
    c_d = 1  # for Chebyshev distance metric, not Euclidean
    e_v2 = (-np.average(special.digamma(n_i_array + 1))
            + special.digamma(n_tot)
            + np.log(c_d)
            + dimension * np.average(np.log(2 * distance_i_array)))
    return e_v2


def mi_knn(k, n_tot, n_x, n_y, method="calc1"):
    """
    Mutual information calculation using k-NN.

    Parameters:
    k (int): Number of nearest neighbors.
    n_tot (int): Total number of points.
    n_x (array-like): Array of neighbor counts for X.
    n_y (array-like): Array of neighbor counts for Y.
    method (str): Calculation method, either "calc1" or "calc2".

    Returns:
    float: Calculated mutual information.
    """
    if method == "calc1":
        mi = (special.digamma(k)
              - np.average(special.digamma(n_x + 1) + special.digamma(n_y + 1))
              + special.digamma(n_tot))
    elif method == "calc2":
        mi = (special.digamma(k)
              - 1 / k
              - np.average(special.digamma(n_x) + special.digamma(n_y))
              + special.digamma(n_tot))
    return mi


def tc_knn(k, n_tot, n_x, n_y, n_z, method="calc1"):
    """
    Total correlation calculation using k-NN.

    Parameters:
    k (int): Number of nearest neighbors.
    n_tot (int): Total number of points.
    n_x (array-like): Array of neighbor counts for X.
    n_y (array-like): Array of neighbor counts for Y.
    n_z (array-like): Array of neighbor counts for Z.
    method (str): Calculation method, either "calc1" or "calc2".

    Returns:
    float: Calculated total correlation.
    """
    if method == "calc1":
        tc = (special.digamma(k)
              - np.average(special.digamma(n_x + 1) + special.digamma(n_y + 1) + special.digamma(n_z + 1))
              + 2 * special.digamma(n_tot))
    elif method == "calc2":
        # Placeholder for another calculation method if needed
        pass
    return tc


def entropy_knn_kdtree_algo(*vecs, k_max, method="eq20"):
    """
    Generalized function to calculate entropy using k-NN KDTree algorithm for 1D, 2D, and 3D data.

    Parameters:
    *vecs (array-like): Variable number of 1D arrays (vec1, vec2, vec3).
    k_max (int): Maximum number of nearest neighbors.
    method (str): Method to use for entropy calculation. Can be "eq20" or "eq22".

    Returns:
    np.ndarray: Calculated entropy for each k value.

    Example:
    # 1D example
    vec1 = np.random.rand(100)
    k_max = 5
    entropy_1d = entropy_knn_kdtree_algo(vec1, k_max=k_max)

    # 3D example
    vec1 = np.random.rand(100)
    vec2 = np.random.rand(100)
    vec3 = np.random.rand(100)
    entropy_3d = entropy_knn_kdtree_algo(vec1, vec2, vec3, k_max=k_max)
    """
    dimension = len(vecs)
    entropy_k = np.zeros(k_max)
    n_tot = len(vecs[0])

    pts = np.c_[tuple(vec.ravel() for vec in vecs)]
    tree = spatial.cKDTree(pts)
    distance_array = tree.query(pts, k_max + 2, p=np.inf)[0]

    for k in range(1, k_max + 1):
        if method == "eq20":
            entropy_k[k - 1] = entropy_knn_eq20(k, n_tot, distance_array[:, k], dimension)
        elif method == "eq22":
            entropy_k[k - 1] = entropy_knn_eq22(k * np.ones(n_tot), n_tot, distance_array[:, k + 1], dimension)

    return entropy_k


def mi_knn_kdtree_algo(vec1, vec2, k_max):

    n_tot = len(vec1)  # Assuming all vectors has the same length
    mi_k = np.zeros(k_max)
    pts = np.c_[vec1.ravel(), vec2.ravel()]
    tree = spatial.cKDTree(pts)
    tree_x = spatial.cKDTree(np.c_[vec1])
    tree_y = spatial.cKDTree(np.c_[vec2])
    distance_array, pts_locations = tree.query(pts, k_max+1, p=np.inf)

    for k in range(1, k_max+1):
        n_x = np.zeros(n_tot, dtype=int)
        n_y = np.zeros(n_tot, dtype=int)

        n_x = tree_x.query_ball_point(pts[:, 0].reshape(-1, 1), distance_array[:, k], p=1,
                                      return_sorted=False, return_length=True) - 1  # Removing self
        n_y = tree_y.query_ball_point(pts[:, 1].reshape(-1, 1), distance_array[:, k], p=1,
                                      return_sorted=False, return_length=True) - 1  # Removing self

        for counter, point in enumerate(pts):
            edge_point_axis = np.argmax([abs(point[0]-pts[pts_locations[counter, k]][0]),
                                         abs(point[1]-pts[pts_locations[counter, k]][1])])
            if edge_point_axis == 0:  # 'x'
                n_x[counter] -= 1  # Removing edge points
            else:  # 'y'
                n_y[counter] -= 1  # Removing edge points

        mi_k[k-1] = mi_knn(k, n_tot, n_x, n_y, method="calc1")

    return mi_k


def tc_knn_kdtree_algo(vec1, vec2, vec3, k_max):

    n_tot = len(vec1)  # Assuming all vectors has the same length
    tc_k = np.zeros(k_max)
    pts = np.c_[vec1.ravel(), vec2.ravel(), vec3.ravel()]
    tree = spatial.cKDTree(pts)
    tree_x = spatial.cKDTree(np.c_[vec1])
    tree_y = spatial.cKDTree(np.c_[vec2])
    tree_z = spatial.cKDTree(np.c_[vec3])
    distance_array, pts_locations = tree.query(pts, k_max+1, p=np.inf)

    for k in range(1, k_max+1):
        n_x = np.zeros(n_tot, dtype=int)
        n_y = np.zeros(n_tot, dtype=int)
        n_z = np.zeros(n_tot, dtype=int)

        n_x = tree_x.query_ball_point(pts[:, 0].reshape(-1, 1), distance_array[:, k], p=1,
                                      return_sorted=False, return_length=True) - 1  # Removing self
        n_y = tree_y.query_ball_point(pts[:, 1].reshape(-1, 1), distance_array[:, k], p=1,
                                      return_sorted=False, return_length=True) - 1  # Removing self
        n_z = tree_z.query_ball_point(pts[:, 2].reshape(-1, 1), distance_array[:, k], p=1,
                                      return_sorted=False, return_length=True) - 1  # Removing self

        for counter, point in enumerate(pts):
            edge_point_axis = np.argmax([abs(point[0]-pts[pts_locations[counter, k]][0]),
                                         abs(point[1]-pts[pts_locations[counter, k]][1]),
                                         abs(point[2]-pts[pts_locations[counter, k]][2])])
            if edge_point_axis == 0:  # 'x'
                n_x[counter] -= 1  # Removing edge points
            elif edge_point_axis == 1:  # 'y'
                n_y[counter] -= 1  # Removing edge points
            else:  # 'z'
                n_z[counter] -= 1  # Removing edge points

        tc_k[k-1] = tc_knn(k, n_tot, n_x, n_y, n_z, method="calc1")

    return tc_k
