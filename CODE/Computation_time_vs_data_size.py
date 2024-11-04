"""
Computation time vs. data size with an expression matrix (genes X conditions)
"""
import math
import os
import time
import numpy as np
import Building_MI_matrices as Building_MI_matrices_mod


def time_array_per_MIestimator_and_MIquantity(mi_est, mi_quantity_to_calc, rand_seed=13):
    """
    Measure computation time for different MI estimators and quantities.

    Parameters:
    mi_est (str): Mutual information estimator ("Shannon", "KSG", "KL").
    mi_quantity_to_calc (str): Quantity to calculate (e.g., "MI", "TC").
    rand_seed (int, optional): Random seed for reproducibility. Default is 13.

    Returns:
    np.ndarray: Array of computation times for different data sizes.
    """
    np.random.seed(rand_seed)

    # constants initialization
    MI_matrix_fname = "MI_matrix.dat"
    # list of vector sizes equivalent to a single gene expression profile with Ntot conditions/perturbations
    Ntot_list = [100, 250, 500, 1000]
    # Ntot_list = [50, 100]  # debug
    time_array = np.zeros(len(Ntot_list), dtype=float)

    # We make a list with number of bins to be used corresponding to the different Ntot size
    if mi_est == "Shannon":
        bins_or_neighbors_list = [math.floor(Ntot ** (1/2)) for Ntot in Ntot_list]
    elif mi_est in ["KSG", "KL"]:
        bins_or_neighbors_list = [1] * len(Ntot_list)

    # Generate "gene expression" matrix for 50 gemes amd upto 1000 conditions/perturbations
    m = np.random.normal(8, 1.5, size=(50, 1000))

    # Build MI matrix and save time to build matrix in time_array
    for n, Ntot in enumerate(Ntot_list):
        input1_data_array = m[:, :Ntot]
        bins_or_neighbors = bins_or_neighbors_list[n]

        start_time = time.time()

        if mi_quantity_to_calc == "MI2":
            Building_MI_matrices_mod.mi2_matrix_build(MI_matrix_fname, input1_data_array, bins_or_neighbors, mi_est)

        elif mi_quantity_to_calc == "TC":
            if mi_est == "KSG":
                Building_MI_matrices_mod.tc_matrix_build(MI_matrix_fname, input1_data_array, bins_or_neighbors, mi_est)
            elif mi_est in ["KL", "Shannon"]:
                Building_MI_matrices_mod.tc_matrix_build_from_entropies(MI_matrix_fname, input1_data_array,
                                                                        bins_or_neighbors, mi_est)

        end_time = time.time()
        time_array[n] = end_time - start_time

    print(mi_est, time_array)
    output_fname = f"Time_array_{mi_quantity_to_calc}_{mi_est}{bins_or_neighbors}_50genes_100to1k_perturb_fast.txt"
    np.savetxt(output_fname, time_array)


# Change to folder where data will be saved
path_to_data = os.path.expanduser('../DATA/MI_comparison_FB_vs_KNN/')
os.chdir(path_to_data)

# Calculating time arrays for MI2 for the MI estimators: Shannon, KL, KSG
time_array_per_MIestimator_and_MIquantity("Shannon", "MI2")
time_array_per_MIestimator_and_MIquantity("KL", "MI2")
time_array_per_MIestimator_and_MIquantity("KSG", "MI2")

# Calculating time arrays for TC for the MI estimators: Shannon, KL, KSG
time_array_per_MIestimator_and_MIquantity("Shannon", "TC")
time_array_per_MIestimator_and_MIquantity("KL", "TC")
time_array_per_MIestimator_and_MIquantity("KSG", "TC")
