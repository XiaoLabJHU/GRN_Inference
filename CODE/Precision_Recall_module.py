"""
This module contains functions to evaluate the performance of GRN inference algorithms.
It calculates the precision, recall, and AUPR of inferred networks based on various algorithms.
Written by Lior Shachaf 2021-02-16
"""

import os
import numpy as np
import Inference_algo_module as inference_module


def import_true_network(network_name_input):
    """
    Load unsigned true synthetic network structure from DREAM3/4 networks.

    Parameters:
    network_name_input (str): Name of the network input file (excluding the "_goldstandard.tsv" extension).

    Returns:
    dict: Dictionary containing the true network pairs.
    """
    true_structure_input = f"{network_name_input}_goldstandard.tsv"
    with open(true_structure_input) as in1:
        in1_data = in1.readlines()

    trueNetwork_dict = {}
    for line in in1_data:
        parts = line.strip().split('\t')
        g1, g2, value = int(parts[0].lstrip('G')), int(parts[1].lstrip('G')), int(parts[2])
        key = f"G{min(g1, g2)}-G{max(g1, g2)}"
        if key not in trueNetwork_dict:
            trueNetwork_dict[key] = value

    return trueNetwork_dict


def PR_calc(trueNet_dict, predictNet):
    """
    Calculate precision and recall for each prediction pair.

    Parameters:
    trueNet_dict (dict): Dictionary containing the true network structure.
    predictNet (np.ndarray): Array containing the predicted network pairs.

    Returns:
    tuple: Arrays containing precision and recall values.
    """
    TP = 0
    FP = 0
    recall = np.zeros(predictNet.shape[0])
    precision = np.zeros(predictNet.shape[0])

    for line in range(predictNet.shape[0]):
        key = f"G{int(predictNet[line][0])}-G{int(predictNet[line][1])}"

        if key not in trueNet_dict:
            raise ValueError(f"{key} [Predicted key] not in true network")

        if trueNet_dict[key] == 1:
            TP += 1
        else:
            FP += 1

        FN = list(trueNet_dict.values()).count(1) - TP
        precision[line] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[line] = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def PR_per_infer_algo(network_name_input, expression_data_type, mi_est_full, infer_algo):
    """
    Calculate PR curves for different expression data types (SS/TS/TSandSS).
    mi_est_full = [bins_or_neighbor, mi_est], for example [10, Shannon].
    Different inference algorithms: MI2, MI2_Zscore, MI2_ARACNE, TC_Z_score, II_Zscore, CMI_CMI, SA_CLR, CMIA_CLR.

    Parameters:
    network_name_input (str): Name of the network input file.
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    mi_est_full (list): List containing bins_or_neighbor and mi_est.
    infer_algo (str): Inference algorithm to use.

    Returns:
    np.ndarray: Array containing precision and recall data.
    """
    inputFileName = f"{network_name_input}_{expression_data_type}_all"

    if "-" in network_name_input:  # DREAM3 style = InSilicoSize100-Ecoli1
        network_size = int(network_name_input.split('-')[0].replace("InSilicoSize", ''))  # DREAM3
    elif "_" in network_name_input:  # DREAM4 style = insilico_size100_1
        network_size = int(network_name_input.split('_')[1].replace("size", ''))  # DREAM4
    max_pairs = int(network_size * (network_size - 1) / 2)  # len(trueNetwork_dict.keys())
    bins_or_neighbor, mi_est = mi_est_full
    PR_data_array = np.zeros((max_pairs, 2))  # Trailing zeros will be trimmed.

    # Load true synthetic network data and save into dictionary
    trueNetwork_dict = import_true_network(network_name_input)

    mi_est_string = {
        "Shannon": f"FB{bins_or_neighbor}_Shan",
        "Miller-Madow": f"FB{bins_or_neighbor}_MM",
        "KSG": f"KNN{bins_or_neighbor}_KSG",
        "KL": f"KNN{bins_or_neighbor}_KL"
    }.get(mi_est, None)

    if mi_est_string is None:
        raise ValueError(f"Invalid mutual information estimator: {mi_est}")

    algo_file_map = {
        "CMI_CMI": f"{expression_data_type}_data_{mi_est_string}_CMIplusCMI_unsigned.dat",
        "SA_CLR_v2": f"{expression_data_type}_data_{mi_est_string}_SA_CLR_v2_unsigned_with_Zscore.dat",
        "SA_CLR_vLior": f"{expression_data_type}_data_{mi_est_string}_SA_CLR_vLior_unsigned_with_Zscore.dat",
        "CMIA_CLR": f"{expression_data_type}_data_{mi_est_string}_CMIA_CLR_unsigned_with_Zscore.dat",
        "CMIA_CLR_vKSG": f"{expression_data_type}_data_{mi_est_string}_CMIA_CLR_vKSG_unsigned_with_Zscore.dat",
        "CLR": f"{expression_data_type}_data_{mi_est_string}_CLR_unsigned.dat",
        "CLRvMinet": f"{expression_data_type}_data_{mi_est_string}_CLRvMinet_unsigned.dat",
        "RL": f"{expression_data_type}_data_{mi_est_string}_RL_unsigned.dat",
        "ARACNE": f"{expression_data_type}_data_{mi_est_string}_ARACNE_unsigned.dat"
    }

    infer_algo_sorted_file = algo_file_map.get(infer_algo, None)

    if infer_algo_sorted_file is None:
        raise ValueError(f"Inference algorithm not recognized: {infer_algo}")

    if infer_algo_sorted_file not in os.listdir():
        algo_func_map = {
            "CMI_CMI": inference_module.CMI_CMI_inference_algo,
            "SA_CLR_v2": inference_module.SA_CLR_v2_inference_algo,
            "SA_CLR_vLior": inference_module.SA_CLR_vLior_inference_algo,
            "CMIA_CLR": inference_module.CMIA_CLR_inference_algo,
            "CMIA_CLR_vKSG": inference_module.CMIA_CLR_vKSG_inference_algo,
            "CLR": inference_module.CLR_inference_algo,
            "CLRvMinet": lambda *args: inference_module.CLR_inference_algo(*args, method="CLRvMinet"),
            "RL": inference_module.RL_inference_algo,
            "ARACNE": inference_module.ARACNE_inference_algo
        }

        algo_func = algo_func_map.get(infer_algo, None)
        if algo_func is None:
            raise ValueError(f"Inference algorithm function not recognized: {infer_algo}")

        algo_func(inputFileName, expression_data_type, str(bins_or_neighbor), mi_est, network_size)

    # Loading sorted files and correcting gene count (shifting 1 number up)
    predictNetwork = np.loadtxt(infer_algo_sorted_file, comments='#', delimiter=',', dtype=int, usecols=(0, 1))
    predictNetwork.T[0] += 1  # data starts from G0 while the goldstandard starts from G1
    predictNetwork.T[1] += 1  # data starts from G0 while the goldstandard starts from G1
    PR_data_array[:predictNetwork.shape[0], 0], PR_data_array[:predictNetwork.shape[0], 1] = PR_calc(trueNetwork_dict,
                                                                                                     predictNetwork)

    return PR_data_array


def AUPR_calc(PR_data):
    """
    Calculate the area under the Precision-Recall (PR) curve using numpy's trapezoid calculation.

    Parameters:
    PR_data (np.ndarray): Array containing precision and recall data.
                          First column should be precision and second column should be recall.

    Returns:
    float: Calculated area under the PR curve (AUPR).
    """
    if PR_data.ndim != 2 or PR_data.shape[1] != 2:
        raise ValueError("PR_data must be a 2D array with two columns: precision and recall.")

    precision = np.trim_zeros(PR_data[:, 0], 'b')
    recall = np.trim_zeros(PR_data[:, 1], 'b')

    if len(precision) == 0 or len(recall) == 0:
        raise ValueError("PR_data contains only zeros or is empty after trimming.")

    AUPR = np.trapz(precision, x=recall)

    return AUPR


def AUPR_replicates_func(topology, expression_data_type, mi_est, infer_algo):
    """
    Calculate AUPR for different replicates of a specific synthetic network.
    This function should be run in the DREAM_X folder.

    Parameters:
    topology (str): Name of the network topology.
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    infer_algo (str): Inference algorithm to use.

    Returns:
    np.ndarray: Array containing AUPR values for each replicate.

    Example:
    mi_est = ["Shannon", "Miller-Madow", "KSG", "Shannon", "Miller-Madow", "KSG"]
    infer_algo = ["MI2", "MI2", "MI2", "MI2_Zscore", "MI2_Zscore", "MI2_Zscore"]
    expression_data_type = "SS"
    topology = "InSilicoSize50-Ecoli2"

    AUPR_array_for_comparison = np.zeros((10, len(mi_est)))
    for counter, (mi, infer) in enumerate(zip(mi_est, infer_algo)):
        AUPR_array_for_comparison[:, counter] = AUPR_replicates_func(topology, expression_data_type, mi, infer)

    print(AUPR_array_for_comparison)
    """
    mi_est_dict = {"Shannon": "Shan", "Miller-Madow": "MM", "KSG": "KSG", "KL": "KL"}

    # Writing output file for summary statistics
    output_filename = f"AUPR_{topology}_{expression_data_type}_{mi_est}_{infer_algo}.dat"

    with open(output_filename, "w") as output_file:
        if os.path.isdir(topology):
            os.chdir(topology)

            replicates = len(os.listdir())
            AUPR_replicate_array = np.zeros(replicates, dtype=float)

            for rep_counter, replicate in enumerate(os.listdir()):
                if os.path.isdir(replicate):
                    os.chdir(replicate)

                    for file in os.listdir():
                        if f"_{expression_data_type}" in file and "_triplets_calc_v3_with_Zscore_v3.dat" in file:
                            if mi_est in ["Shannon", "Miller-Madow"] and mi_est_dict[mi_est] in file:
                                bins_or_neighbor = file.split("FB")[1].split("_")[0]
                            elif mi_est in ["KSG", "KL"] and mi_est_dict[mi_est] in file:
                                bins_or_neighbor = file.split("KNN")[1].split("_")[0]
                            else:
                                continue

                            mi_est_full = [bins_or_neighbor, mi_est]

                            # Calculating the AUPR
                            AUPR_replicate_array[rep_counter] = AUPR_calc(PR_per_infer_algo(topology, expression_data_type,
                                                                                            mi_est_full, infer_algo))
                            print(f"Done {replicate} {file}")

                            if "-" in topology:  # DREAM3 style = InSilicoSize100-Ecoli1
                                output_file.write(f"{AUPR_replicate_array[rep_counter]:.3f},"
                                                  f"{int(topology.split('-')[0].replace('InSilicoSize', ''))},"
                                                  f"{topology.split('-')[1]},{expression_data_type},{mi_est},"
                                                  f"{int(bins_or_neighbor)},{infer_algo},{replicate}\n")
                            elif "_" in topology:  # DREAM4 style = insilico_size100_1
                                output_file.write(f"{AUPR_replicate_array[rep_counter]:.3f},"
                                                  f"{int(topology.split('_')[1].replace('size', ''))},"
                                                  f"{topology.split('_')[2]},{expression_data_type},{mi_est},"
                                                  f"{int(bins_or_neighbor)},{infer_algo},{replicate}\n")
                            break

                    os.chdir('..')

            os.chdir('..')

    return AUPR_replicate_array
