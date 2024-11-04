"""
Mutual Information Estimators Comparison for Gaussian Distribution

This script calculates the following MI quantities for four different estimators
(Shannon, Miller-Madow, KL, KSG) for random variables drawn from a Gaussian
distribution, with different correlations and different data sizes:
1) Two-way MI, MI2
2) Total Correlation, TC
3) Three-way MI, MI3
4) Interaction Information, II
5) Conditional MI, CMI

(*) This is repeated 100 times (replicates)
"""

import os
import time
import numpy as np
import scipy.io

# Information calculation with units of [nats]
import Mutual_Info_based_binning_module as MI_FB_mod
import Mutual_Info_KNN_nats_module as MI_KNN_mod


def run_MI_FB_2d(iterations, corr_list, nTot_list, max_num_of_bins, rand_seed=13, mi_est="Shannon"):
    """
    Calculate two-way mutual information (MI) for 2D data using fixed binning.

    This function generates random variables drawn from a Gaussian distribution with specified correlations
    and data sizes, and calculates the two-way MI for each combination of correlation, data size, and number of bins.
    The MI is calculated using the specified estimator (Shannon or Miller-Madow).

    Parameters:
    iterations (int): Number of iterations to run the calculation.
    corr_list (list of float): List of correlation values to use for generating the Gaussian distribution.
    nTot_list (list of int): List of total data sizes to use for generating the Gaussian distribution.
    max_num_of_bins (int): Maximum number of bins to use for discretizing the data.
    rand_seed (int, optional): Random seed for reproducibility. Default is 13.
    mi_est (str, optional): Mutual information estimator to use. Can be "Shannon" or "Miller-Madow". Default is "Shannon".

    Returns:
    np.ndarray: A 4D array containing the calculated MI values. The shape of the array is
                (len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations).
    """
    # Local constants initialization
    np.random.seed(rand_seed)
    mi2_bins_4d_array = np.zeros((len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations), dtype=float)

    for iteration in range(iterations):
        for c, correlation in enumerate(corr_list):
            covs = [[1, correlation], [correlation, 1]]
            m = np.random.multivariate_normal([0, 0], covs, max(nTot_list)).T

            for n, nTot in enumerate(nTot_list):
                for num_of_bins in range(2, max_num_of_bins):
                    mi2_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.two_way_info(
                        m[0][:nTot], m[1][:nTot], num_of_bins, mi_est)

    return mi2_bins_4d_array


def run_MI_FB_3d(iterations, corr_list, nTot_list, max_num_of_bins, rand_seed=13, mi_est="Shannon"):
    """
    Calculate various mutual information (MI) quantities for 3D data using fixed binning.

    This function generates random variables drawn from a Gaussian distribution with specified correlations
    and data sizes, and calculates the following MI quantities for each combination of correlation, data size,
    and number of bins:
    1) Total Correlation (TC)
    2) Interaction Information (II)
    3) Conditional Mutual Information (CMI)
    4) Three-way Mutual Information (MI3)

    The MI is calculated using the specified estimator (Shannon or Miller-Madow).

    Parameters:
    iterations (int): Number of iterations to run the calculation.
    corr_list (list of float): List of correlation values to use for generating the Gaussian distribution.
    nTot_list (list of int): List of total data sizes to use for generating the Gaussian distribution.
    max_num_of_bins (int): Maximum number of bins to use for discretizing the data.
    rand_seed (int, optional): Random seed for reproducibility. Default is 13.
    mi_est (str, optional): Mutual information estimator to use. Can be "Shannon" or "Miller-Madow". Default is "Shannon".

    Returns:
    tuple: A tuple containing four 4D arrays with the calculated MI values:
        - TC_bins_4d_array: Total Correlation values.
        - II_bins_4d_array: Interaction Information values.
        - CMI_bins_4d_array: Conditional Mutual Information values.
        - MI3_bins_4d_array: Three-way Mutual Information values.
    """
    # Local constants initialization
    np.random.seed(rand_seed)
    TC_bins_4d_array = np.zeros((len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations), dtype=float)
    II_bins_4d_array = np.zeros((len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations), dtype=float)
    CMI_bins_4d_array = np.zeros((len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations), dtype=float)
    MI3_bins_4d_array = np.zeros((len(corr_list), len(nTot_list), max_num_of_bins - 2, iterations), dtype=float)

    for iteration in range(iterations):
        for c, correlation in enumerate(corr_list):
            covs = [[1, correlation, correlation], [correlation, 1, correlation], [correlation, correlation, 1]]
            m = np.random.multivariate_normal([0, 0, 0], covs, max(nTot_list)).T

            for n, nTot in enumerate(nTot_list):
                for num_of_bins in range(2, max_num_of_bins):
                    # Calculating MI quantities - slow if using to calculate more than one quantity
                    # TC_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.total_corr(
                    #     m[0][:nTot], m[1][:nTot], m[2][:nTot], num_of_bins)
                    # II_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.inter_info(
                    #     m[0][:nTot], m[1][:nTot], m[2][:nTot], num_of_bins)
                    # CMI_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.conditional_mutual_info(
                    #     m[0][:nTot], m[1][:nTot], m[2][:nTot], num_of_bins)
                    # MI3_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.three_way_info(
                    #     m[0][:nTot], m[1][:nTot], m[2][:nTot], num_of_bins)

                    # Calculating from entropies - this is faster when calculating all 3D MI quantities
                    Ex = MI_FB_mod.entropy(m[0][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Ey = MI_FB_mod.entropy(m[1][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Ez = MI_FB_mod.entropy(m[2][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Exy = MI_FB_mod.entropy(m[0][:nTot], m[1][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Exz = MI_FB_mod.entropy(m[0][:nTot], m[2][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Eyz = MI_FB_mod.entropy(m[1][:nTot], m[2][:nTot], bins_num=num_of_bins, mi_est=mi_est)
                    Exyz = MI_FB_mod.entropy(m[0][:nTot], m[1][:nTot], m[2][:nTot], bins_num=num_of_bins, mi_est=mi_est)

                    TC_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.total_corr_from_entropy(Ex, Ey, Ez, Exyz)
                    II_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.inter_info_from_entropy(Ex, Ey, Ez, Exy,
                                                                                                           Exz, Eyz, Exyz)
                    CMI_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.conditional_mutual_info_from_entropy(
                                                                                                        Exz, Eyz, Exyz, Ez)
                    MI3_bins_4d_array[c][n][num_of_bins - 2][iteration] = MI_FB_mod.three_way_info_from_entropy(Ez, Exy, Exyz)

    return TC_bins_4d_array, II_bins_4d_array, CMI_bins_4d_array, MI3_bins_4d_array


def run_MI_KNN_2d(iterations, corr_list, nTot_list, k_max=3, rand_seed=13, mi_est="KSG"):
    """
    Calculate two-way mutual information (MI) for 2D data using k-nearest neighbors (k-NN).

    This function generates random variables drawn from a Gaussian distribution with specified correlations
    and data sizes, and calculates the two-way MI for each combination of correlation, data size, and number of neighbors.
    The MI is calculated using the specified estimator (KSG or KL).

    Parameters:
    iterations (int): Number of iterations to run the calculation.
    corr_list (list of float): List of correlation values to use for generating the Gaussian distribution.
    nTot_list (list of int): List of total data sizes to use for generating the Gaussian distribution.
    k_max (int, optional): Maximum number of nearest neighbors. Default is 3.
    rand_seed (int, optional): Random seed for reproducibility. Default is 13.
    mi_est (str, optional): Mutual information estimator to use. Can be "KSG" or "KL". Default is "KSG".

    Returns:
    np.ndarray: A 4D array containing the calculated MI values. The shape of the array is
                (len(corr_list), len(nTot_list), k_max, iterations).
    """
    # Local constants initialization
    np.random.seed(rand_seed)
    mi2_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)

    for iteration in range(iterations):
        for c, correlation in enumerate(corr_list):
            covs = [[1, correlation], [correlation, 1]]
            m = np.random.multivariate_normal([0, 0], covs, max(nTot_list)).T

            for n, nTot in enumerate(nTot_list):
                if mi_est == "KSG":
                    mi2_knn_4d_array[c, n, :, iteration] = MI_KNN_mod.mi_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], k_max)
                elif mi_est == "KL":
                    Ex = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], k_max=k_max)
                    Ey = MI_KNN_mod.entropy_knn_kdtree_algo(m[1][:nTot], k_max=k_max)
                    Exy = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], k_max=k_max)
                    mi2_knn_4d_array[c, n, :, iteration] = MI_FB_mod.two_way_info_from_entropy(Ex, Ey, Exy)

    return mi2_knn_4d_array


# Save path to where data will be stored
# path_to_data = os.path.expanduser('~/DATA/MI_comparison_FB_vs_KNN/')
project_directory = os.getcwd().split('CODE')[0]
path_to_data = f"{project_directory}/DATA/MI_comparison_FB_vs_KNN/"

# Global constants initialization
corr_list = [0.3, 0.6, 0.9]
nTot_list = [100, 1000, 10000]
# nTot_list = [100, 250]  # debug
iterations = 100
# iterations = 3  # debug
max_num_of_bins = 101
k_max = 10  # k-NN
random_seed = 13  # np.random.seed(13)  # CHANGE ME IF NEEDED


# Calculate MI using fixed width binning (FB) in 2D
mi2_bins_4d_array = run_MI_FB_2d(iterations, corr_list, nTot_list, max_num_of_bins, mi_est="Shannon")
print(f"MI2 bins 4d array shape: {mi2_bins_4d_array.shape}")
dict_name = 'MI2_bins_4d_array'
matfile = f"{path_to_data}{dict_name}.mat"
scipy.io.savemat(matfile, mdict={dict_name: mi2_bins_4d_array})


# Calculate MI using Miller-Madow FB correction in 2D
mi2_MM_bins_4d_array = run_MI_FB_2d(iterations, corr_list, nTot_list, max_num_of_bins, mi_est="Miller-Madow")
print(f"MI2 bins 4d array shape: {mi2_MM_bins_4d_array.shape}")
dict_name = 'MI2_MM_bins_4d_array'
matfile = f"{path_to_data}{dict_name}.mat"
scipy.io.savemat(matfile, mdict={dict_name: mi2_MM_bins_4d_array})


# Calculate MI using fixed width binning (FB) in 3D with 100 iterations
t1 = time.time()
TC_bins_4d_array, II_bins_4d_array, CMI_bins_4d_array, MI3_bins_4d_array = run_MI_FB_3d(iterations, corr_list, nTot_list,
                                                                                        max_num_of_bins, mi_est="Shannon")
t2 = time.time()
print(f"Shannon FB 3D calc with {iterations} iterations: Run time = {t2 - t1:.2f} [sec]")
dict_name_list = ['TC_bins_4d_array', 'II_bins_4d_array', 'CMI_bins_4d_array', 'MI3_bins_4d_array']
data_4d_array_list = [TC_bins_4d_array, II_bins_4d_array, CMI_bins_4d_array, MI3_bins_4d_array]
for counter, dict_name in enumerate(dict_name_list):
    matfile = f"{path_to_data}{dict_name}.mat"
    scipy.io.savemat(matfile, mdict={dict_name: data_4d_array_list[counter]})


# Calculate MI using Miller-Madow FB 3D calc with 100 iterations
t1 = time.time()
TC_bins_4d_array, II_bins_4d_array, CMI_bins_4d_array, MI3_bins_4d_array = run_MI_FB_3d(iterations, corr_list, nTot_list,
                                                                                        max_num_of_bins, mi_est="Miller-Madow")
t2 = time.time()
print(f"Miller-Madow FB 3D calc with {iterations} iterations: Run time = {t2 - t1:.2f} [sec]")
dict_name_list = ['TC_MM_bins_4d_array', 'II_MM_bins_4d_array', 'CMI_MM_bins_4d_array', 'MI3_MM_bins_4d_array']
data_4d_array_list = [TC_bins_4d_array, II_bins_4d_array, CMI_bins_4d_array, MI3_bins_4d_array]
for counter, dict_name in enumerate(dict_name_list):
    matfile = f"{path_to_data}{dict_name}.mat"
    scipy.io.savemat(matfile, mdict={dict_name: data_4d_array_list[counter]})


# 2D KNN multiple iterations
t1 = time.time()
mi2_knn_KSG_4d_array = run_MI_KNN_2d(iterations, corr_list, nTot_list, k_max=k_max, rand_seed=random_seed, mi_est="KSG")
t2 = time.time()
print(f"2D KNN multiple iterations: Run time = {t2 - t1:.2f} [sec]")
dict_name = 'MI2_knn_KSG_4d_array'
matfile = f"{path_to_data}{dict_name}.mat"
scipy.io.savemat(matfile, mdict={dict_name: mi2_knn_KSG_4d_array})


# 2D KL-KNN
t1 = time.time()
mi2_knn_KL_4d_array = run_MI_KNN_2d(iterations, corr_list, nTot_list, k_max=k_max, rand_seed=random_seed, mi_est="KL")
t2 = time.time()
print(f"KL-KNN in 2D: Run time = {t2 - t1:.3f} [sec]")
dict_name = 'MI2_knn_KL_4d_array'
matfile = f"{path_to_data}{dict_name}.mat"
scipy.io.savemat(matfile, mdict={dict_name: mi2_knn_KL_4d_array})

"""
3D KNN
"""
# Local constants initialization
np.random.seed(13)  # CHANGE ME IF NEEDED
MI2xy_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI2xz_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI2yz_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
TC_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
II_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
CMI_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI3_knn_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
t1 = time.time()

for iteration in range(iterations):
    for c, correlation in enumerate(corr_list):
        covs = [[1, correlation, correlation], [correlation, 1, correlation], [correlation, correlation, 1]]
        m = np.random.multivariate_normal([0, 0, 0], covs, max(nTot_list)).T

        for n, nTot in enumerate(nTot_list):
            MI2xy_knn_4d_array[c, n, :, iteration] = MI_KNN_mod.mi_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], k_max)
            MI2xz_knn_4d_array[c, n, :, iteration] = MI_KNN_mod.mi_knn_kdtree_algo(m[0][:nTot], m[2][:nTot], k_max)
            MI2yz_knn_4d_array[c, n, :, iteration] = MI_KNN_mod.mi_knn_kdtree_algo(m[1][:nTot], m[2][:nTot], k_max)
            TC_knn_4d_array[c, n, :, iteration] = MI_KNN_mod.tc_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], m[2][:nTot], k_max)
            II_knn_4d_array[c, n, :, iteration] = (TC_knn_4d_array[c, n, :, iteration]
                                                   - MI2xy_knn_4d_array[c, n, :, iteration]
                                                   - MI2xz_knn_4d_array[c, n, :, iteration]
                                                   - MI2yz_knn_4d_array[c, n, :, iteration])
            CMI_knn_4d_array[c, n, :, iteration] = (TC_knn_4d_array[c, n, :, iteration]
                                                    - MI2xz_knn_4d_array[c, n, :, iteration]
                                                    - MI2yz_knn_4d_array[c, n, :, iteration])
            MI3_knn_4d_array[c, n, :, iteration] = (TC_knn_4d_array[c, n, :, iteration]
                                                    - MI2xy_knn_4d_array[c, n, :, iteration])

t2 = time.time()
print(f"3D KNN multiple iterations: Run time = {t2 - t1:.2f} [sec]")

dict_name_list = [
    'MI2xy_knn_KSG_4d_array',
    'MI2xz_knn_KSG_4d_array',
    'MI2yz_knn_KSG_4d_array',
    'TC_knn_KSG_4d_array',
    'II_knn_KSG_4d_array',
    'CMI_knn_KSG_4d_array',
    'MI3_knn_KSG_4d_array'
]
data_4d_array_list = [
    MI2xy_knn_4d_array,
    MI2xz_knn_4d_array,
    MI2yz_knn_4d_array,
    TC_knn_4d_array,
    II_knn_4d_array,
    CMI_knn_4d_array,
    MI3_knn_4d_array
]

for counter, dict_name in enumerate(dict_name_list):
    matfile = f"{path_to_data}{dict_name}.mat"
    scipy.io.savemat(matfile, mdict={dict_name: data_4d_array_list[counter]})


"""
3D KL-KNN
"""
# Local constants initialization
np.random.seed(13)  # CHANGE ME IF NEEDED
MI2xy_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI2xz_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI2yz_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
TC_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
II_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
CMI_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
MI3_knn_KL_4d_array = np.zeros((len(corr_list), len(nTot_list), k_max, iterations), dtype=float)
t1 = time.time()

for iteration in range(iterations):
    for c, correlation in enumerate(corr_list):
        covs = [[1, correlation, correlation], [correlation, 1, correlation], [correlation, correlation, 1]]
        m = np.random.multivariate_normal([0, 0, 0], covs, max(nTot_list)).T

        for n, nTot in enumerate(nTot_list):
            Ex = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], k_max=k_max)
            Ey = MI_KNN_mod.entropy_knn_kdtree_algo(m[1][:nTot], k_max=k_max)
            Ez = MI_KNN_mod.entropy_knn_kdtree_algo(m[2][:nTot], k_max=k_max)
            Exy = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], k_max=k_max)
            Exz = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], m[2][:nTot], k_max=k_max)
            Eyz = MI_KNN_mod.entropy_knn_kdtree_algo(m[1][:nTot], m[2][:nTot], k_max=k_max)
            Exyz = MI_KNN_mod.entropy_knn_kdtree_algo(m[0][:nTot], m[1][:nTot], m[2][:nTot], k_max=k_max)

            MI2xy_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.two_way_info_from_entropy(Ex, Ey, Exy)
            MI2xz_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.two_way_info_from_entropy(Ex, Ez, Exz)
            MI2yz_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.two_way_info_from_entropy(Ey, Ez, Eyz)
            TC_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.total_corr_from_entropy(Ex, Ey, Ez, Exyz)
            II_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.inter_info_from_entropy(Ex, Ey, Ez, Exy, Exz, Eyz, Exyz)
            CMI_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.conditional_mutual_info_from_entropy(Exz, Eyz, Exyz, Ez)
            MI3_knn_KL_4d_array[c, n, :, iteration] = MI_FB_mod.three_way_info_from_entropy(Ez, Exy, Exyz)

t2 = time.time()
print(f"KL-KNN in 3D: Run time = {t2 - t1:.3f} [sec]")

dict_name_list = ['TC_knn_KL_4d_array', 'II_knn_KL_4d_array', 'CMI_knn_KL_4d_array', 'MI3_knn_KL_4d_array']
data_4d_array_list = [TC_knn_KL_4d_array, II_knn_KL_4d_array, CMI_knn_KL_4d_array, MI3_knn_KL_4d_array]

for counter, dict_name in enumerate(dict_name_list):
    matfile = f"{path_to_data}{dict_name}.mat"
    scipy.io.savemat(matfile, mdict={dict_name: data_4d_array_list[counter]})
