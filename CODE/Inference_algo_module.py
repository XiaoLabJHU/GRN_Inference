"""
This module contains functions for GRN inference algorithms (Relevance Networks, ARACNE, CLR, SA-CLR, CMIA-CLR, CMI-CMI).
It generates files with gene pairs based on ranking for each inference algorithm.
Written by Lior Shachaf 2021-08-20
"""

import numpy as np
import pandas as pd


def mi_est_string_func(mi_est, bins_or_neighbors):
    """
    Combine mi_est and bins_or_neighbors into a string to be used in other functions for input and output file names.

    Parameters:
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    bins_or_neighbors (str): Number of bins or neighbors.

    Returns:
    str: Combined string of mi_est and bins_or_neighbors.
    """
    if mi_est == "Shannon":
        return f"FB{bins_or_neighbors}_Shan"
    elif mi_est == "Miller-Madow":
        return f"FB{bins_or_neighbors}_MM"
    elif mi_est in ["KSG", "KL"]:
        return f"KNN{bins_or_neighbors}_{mi_est}"
    else:
        raise ValueError("Invalid mi_est value. Choose from 'Shannon', 'Miller-Madow', 'KSG', 'KL'.")


def RL_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Relevance Networks algorithm based on Butte 2000 paper.

    Parameters:
    filename_data (str): Data filename (without the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Prepare file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Gene X', 'Gene Y', 'MI(X;Y)']

    # Read the data
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X', 'Y', 'MI(X;Y)'])

    # For kNN only: consider setting negative values to zero
    if mi_est in ["KSG", "KL"]:
        df['MI(X;Y)'] = df['MI(X;Y)'].clip(lower=0)

    # Convert specific columns to integer
    df = df.astype({'X': int, 'Y': int})

    # Sort according to MI2 and save to file
    df_sorted = df.sort_values(by=['MI(X;Y)', 'Y'], ascending=False)
    df_sorted = df_sorted.drop_duplicates(subset=['MI(X;Y)'], ignore_index=True)

    # Save to file
    output_filename = f"{expression_data_type}_data_{mi_est_string_with_bins}_RL_unsigned.dat"
    df_sorted.to_csv(output_filename, columns=['X', 'Y', 'MI(X;Y)'], header=['#X', 'Y', 'MI(X;Y)'], index=False)


def ARACNE_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Algorithm for the Reconstruction of Accurate Cellular Networks (ARACNE) based on the paper by Margolin et al. (2006).
    Written by Lior Shachaf 2021-12-02.

    This function adds data processing inequality (DPI) to prun gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])

    # Remove excess pairs as the original df contains all (X,Y) and (Y,X)
    df = df[(df['X'] < df['Y']) & (df['Y'] < df['Z'])]

    # For kNN only: consider setting negative values to zero
    df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']].clip(lower=0)

    # Initialize DataFrame for storing results
    tmp_df = df[['X', 'Y', 'MI(X;Y)']].copy()

    for gene1_index in range(network_size - 2):
        for gene2_index in range(gene1_index + 1, network_size - 1):
            for gene3_index in range(gene2_index + 1, network_size):
                df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index) & (df['Z'] == gene3_index)]

                # Remove rows according to DPI
                if not df_filtered.empty:
                    if df_filtered['MI(X;Y)'].item() < df_filtered[['MI(X;Z)', 'MI(Y;Z)']].min().min():
                        tmp_df.drop(tmp_df[(tmp_df['X'] == gene1_index) & (tmp_df['Y'] == gene2_index)].index, inplace=True)
                        break

    # Convert specific columns to integer
    tmp_df = tmp_df.astype({'X': int, 'Y': int})

    # Sort according to MI2 and save to file
    df_sorted = tmp_df.sort_values(by=['MI(X;Y)', 'Y'], ascending=False)
    # Remove duplicate rows and use new index for row name
    df_sorted = df_sorted.drop_duplicates(subset=['MI(X;Y)'], ignore_index=True)

    # Save to file
    output_filename = f"{expression_data_type}_data_{mi_est_string_with_bins}_ARACNE_unsigned.dat"
    df_sorted.to_csv(output_filename, columns=['X', 'Y', 'MI(X;Y)'], header=['#X', 'Y', 'MI(X;Y)'], index=False)


def CLR_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size, method="CLR"):
    """
    Context-Likelihood-Relevance (CLR) algorithm based on the paper by Faith et al. (2007).
    Written by Lior Shachaf 2021-08-27.

    This function calculates the Z-score for gene pairs based on mutual information (MI) values and CLR or CLRvMinet method.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.
    CLRvMinet is similar to CLR but is using standard deviation (STD) instead of variance (VAR).
    This seems to increase AUPR by few a percent in downstream analysis.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.
    method (str, optional): Calculation method to use. Can be "CLR" or "CLRvMinet". Default is "CLR".

    Returns:
    None
    """
    # Prepare file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Gene X', 'Gene Y', 'MI(X;Y)']

    # Read the data
    df_raw = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X', 'Y', 'MI(X;Y)'])

    # Remove duplicate rows and use new index for row name
    df = df_raw.drop_duplicates(ignore_index=True)

    # For kNN only: consider setting negative values to zero
    if mi_est in ["KSG", "KL"]:
        df['MI(X;Y)'] = df['MI(X;Y)'].clip(lower=0)

    # Initialize DataFrame for storing results
    MI_df = pd.DataFrame(columns=['X', 'Y', 'MI(X;Y)', 'X-mean', 'X-var', 'Y-mean', 'Y-var'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            MI_value = df_filtered['MI(X;Y)'].unique()
            if MI_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI(X;Y)': [MI_value.item()]})
            MI_df = pd.concat([MI_df, new_row], ignore_index=True)

    # Calculate mean and variance for each gene
    for gene_index in range(network_size):
        df_filtered = MI_df[(MI_df['X'] == gene_index) | (MI_df['Y'] == gene_index)]

        mean_x = df_filtered['MI(X;Y)'].mean()
        if method == "CLR":
            var_x = df_filtered['MI(X;Y)'].var()
            MI_df.loc[MI_df['X'] == gene_index, ['X-mean', 'X-var']] = mean_x, var_x
            MI_df.loc[MI_df['Y'] == gene_index, ['Y-mean', 'Y-var']] = mean_x, var_x
        elif method == "CLRvMinet":
            # NOTE: pandas.std() uses ddof=1 while numpy ddof=0 => checked on dream3/Yeast1-size50 and found no difference
            std_x = df_filtered['MI(X;Y)'].std()
            MI_df.loc[MI_df['X'] == gene_index, ['X-mean', 'X-std']] = mean_x, std_x
            MI_df.loc[MI_df['Y'] == gene_index, ['Y-mean', 'Y-std']] = mean_x, std_x

    # start DEBUG
    # output_filename = f"{expression_data_type}_data_{mi_est_string_with_bins}_{method}_unsigned.dat"
    # MI_df.to_csv(output_filename, index=False)
    # End DEBUG

    # Calculate Z-score for each pair
    if method == "CLR":
        # [Faith 2007]: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
        MI_df['Zscore'] = np.sqrt(((MI_df['MI(X;Y)'] - MI_df['X-mean'])**2 / MI_df['X-var'])
                                  + ((MI_df['MI(X;Y)'] - MI_df['Y-mean'])**2 / MI_df['Y-var']))
    elif method == "CLRvMinet":
        # [minet 2008]: Zscore_X = max(0, MI(X;Y)-meanX)/stdX), Zscore_XY = sqrt((Zscore_X)**2 + (Zscore_Y)**2)
        MI_df['Zscore_X'] = np.maximum(0, (MI_df['MI(X;Y)'] - MI_df['X-mean']) / MI_df['X-std'])
        MI_df['Zscore_Y'] = np.maximum(0, (MI_df['MI(X;Y)'] - MI_df['Y-mean']) / MI_df['Y-std'])
        MI_df['Zscore'] = np.sqrt((MI_df['Zscore_X'])**2 + (MI_df['Zscore_Y'])**2)

    # Convert specific columns to integer
    MI_df = MI_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    MI_df_sorted = MI_df.sort_values(by='Zscore', ascending=False)
    output_filename = f"{expression_data_type}_data_{mi_est_string_with_bins}_{method}_unsigned.dat"
    MI_df_sorted.to_csv(output_filename, columns=['X', 'Y', 'Zscore'], header=['#X', 'Y', 'Zscore'], index=False)


def SA_CLR_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    Written by Lior Shachaf 2021-08-31.

    This function calculates the SA-CLR scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Prepare file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    output_filename_with_Zscore = f"{expression_data_type}_data_{mi_est_string_with_bins}_SA_CLR_unsigned_with_Zscore.dat"

    # Read input file and save as pandas DataFrame
    column_names = [
        'X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)',
        'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)'
    ]
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'II(XYZ)'])

    # For kNN only: consider setting negative values to zero
    if mi_est in ["KSG", "KL"]:
        df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']].clip(lower=0)

    # Apply first step in SA-CLR
    df = df[(df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)'])]

    # Initialize DataFrame for storing results
    SA_df = pd.DataFrame(columns=['X', 'Y', 'MI+maxII', 'X-mean', 'X-var', 'Y-mean', 'Y-var'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].max()
            if SA_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI+maxII': [SA_value.item()]})
            SA_df = pd.concat([SA_df, new_row], ignore_index=True)

    # Calculate mean and variance for each gene
    for gene_index in range(network_size):
        df_filtered = SA_df[(SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index)]

        mean_x = df_filtered['MI+maxII'].mean()
        var_x = df_filtered['MI+maxII'].var()

        SA_df.loc[SA_df['X'] == gene_index, ['X-mean', 'X-var']] = mean_x, var_x
        SA_df.loc[SA_df['Y'] == gene_index, ['Y-mean', 'Y-var']] = mean_x, var_x

    # Calculate Z-score for each pair
    SA_df['Zscore'] = np.sqrt(((SA_df['MI+maxII'] - SA_df['X-mean'])**2 / SA_df['X-var'])
                              + ((SA_df['MI+maxII'] - SA_df['Y-mean'])**2 / SA_df['Y-var']))

    # Convert specific columns to integer
    SA_df = SA_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore', ascending=False)
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns=['X', 'Y', 'Zscore'], header=['#X', 'Y', 'Zscore'], index=False)


def SA_CLR_v2_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    Similar to the previous SA_CLR_inference_algo function but using STD instead of VAR. Like the CLR version in Minet.
    Written by Lior Shachaf 2021-08-31.

    This function calculates the SA-CLR scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])

    # For kNN only: consider setting negative values to zero
    df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']].clip(lower=0)

    # Recalculate CMI(X;Y|Z)
    df['II(XYZ)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)'] - df['MI(Y;Z)']

    # Initialize DataFrame for storing results
    SA_df = pd.DataFrame(columns=['X', 'Y', 'MI+maxII', 'X-mean', 'X-std', 'Y-mean', 'Y-std'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].max()
            if SA_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI+maxII': [SA_value.item()]})
            SA_df = pd.concat([SA_df, new_row], ignore_index=True)

    # Calculate mean and standard deviation for each gene
    for gene_index in range(network_size):
        df_filtered = SA_df[(SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index)]

        mean_x = df_filtered['MI+maxII'].mean()
        std_x = df_filtered['MI+maxII'].std()  # pandas.std() uses ddof=1 while numpy ddof=0

        SA_df.loc[SA_df['X'] == gene_index, ['X-mean', 'X-std']] = mean_x, std_x
        SA_df.loc[SA_df['Y'] == gene_index, ['Y-mean', 'Y-std']] = mean_x, std_x

    # Calculate Z-score for each pair as in [minet 2008]
    SA_df['Zscore_X'] = np.maximum(0, (SA_df['MI+maxII'] - SA_df['X-mean']) / SA_df['X-std'])
    SA_df['Zscore_Y'] = np.maximum(0, (SA_df['MI+maxII'] - SA_df['Y-mean']) / SA_df['Y-std'])
    SA_df['Zscore_XY'] = np.sqrt(SA_df['Zscore_X']**2 + SA_df['Zscore_Y']**2)

    # Convert specific columns to integer
    SA_df = SA_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore_XY', ascending=False)

    # Write output to file
    output_filename_with_Zscore = f"{expression_data_type}_data_{mi_est_string_with_bins}_SA_CLR_v2_unsigned_with_Zscore.dat"
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns=['X', 'Y', 'Zscore_XY'],
                        header=['#X', 'Y', 'Zscore'], index=False)


def SA_CLR_vLior_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    Changed MI2+maxII to MI2+minIIi, rest is same as SA_CLR_v2_inference_algo.
    Written by Lior Shachaf 2021-09-14.

    This function calculates the SA-CLR scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Prepare file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    output_filename_with_Zscore = (f"{expression_data_type}_data_{mi_est_string_with_bins}"
                                   "_SA_CLR_vLior_unsigned_with_Zscore.dat")

    # Read input file and save as pandas DataFrame
    column_names = [
        'X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)',
        'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)'
    ]
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])

    # For kNN only: consider setting negative values to zero
    df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']].clip(lower=0)

    # Recalculate CMI(X;Y|Z)
    df['II(XYZ)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)'] - df['MI(Y;Z)']

    # Initialize DataFrame for storing results
    SA_df = pd.DataFrame(columns=['X', 'Y', 'MI+minII', 'X-mean', 'X-std', 'Y-mean', 'Y-std'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].min()
            if SA_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI+minII': [SA_value.item()]})
            SA_df = pd.concat([SA_df, new_row], ignore_index=True)

    # Calculate mean and standard deviation for each gene
    for gene_index in range(network_size):
        df_filtered = SA_df[(SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index)]

        mean_x = df_filtered['MI+minII'].mean()
        std_x = df_filtered['MI+minII'].std()  # pandas.std() uses ddof=1 while numpy ddof=0

        SA_df.loc[SA_df['X'] == gene_index, ['X-mean', 'X-std']] = mean_x, std_x
        SA_df.loc[SA_df['Y'] == gene_index, ['Y-mean', 'Y-std']] = mean_x, std_x

    # Calculate Z-score for each pair as in [minet 2008]
    SA_df['Zscore_X'] = np.maximum(0, (SA_df['MI+minII'] - SA_df['X-mean']) / SA_df['X-std'])
    SA_df['Zscore_Y'] = np.maximum(0, (SA_df['MI+minII'] - SA_df['Y-mean']) / SA_df['Y-std'])
    SA_df['Zscore_XY'] = np.sqrt(SA_df['Zscore_X']**2 + SA_df['Zscore_Y']**2)

    # Convert specific columns to integer
    SA_df = SA_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore_XY', ascending=False)
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns=['X', 'Y', 'MI+minII', 'Zscore_XY'],
                        header=['#X', 'Y', 'MI+minII', 'Zscore'], index=False)


def CMIA_CLR_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    but replacing Synergy with CMI.
    Written by Lior Shachaf 2021-07-23.

    This function calculates the CMIA+CLR scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    output_filename_with_Zscore = f"{expression_data_type}_data_{mi_est_string_with_bins}_CMIA_CLR_unsigned_with_Zscore.dat"

    # Preparing temp file where MI(i;j) > MI(i;k) and MI(j;k) > MI(i;k)
    column_names = [
        'X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)',
        'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)'
    ]
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'CMI(X;Y|Z)'])

    # For kNN only: setting negative values to zero
    if mi_est in ["KSG", "KL"]:
        df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']].clip(lower=0)

    # Apply first step in SA-CLR
    df = df[(df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)'])]

    # Initialize DataFrame for storing results
    CMIA_df = pd.DataFrame(columns=['X', 'Y', 'MI+maxCMI', 'X-mean', 'X-var', 'Y-mean', 'Y-var'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            CMIA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['CMI(X;Y|Z)'].max()
            if CMIA_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI+maxCMI': [CMIA_value.item()]})
            CMIA_df = pd.concat([CMIA_df, new_row], ignore_index=True)

    # Calculate mean and variance for each gene
    for gene_index in range(network_size):
        df_filtered = CMIA_df[(CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)]

        mean_x = df_filtered['MI+maxCMI'].mean()
        var_x = df_filtered['MI+maxCMI'].var()

        CMIA_df.loc[CMIA_df['X'] == gene_index, ['X-mean', 'X-var']] = mean_x, var_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, ['Y-mean', 'Y-var']] = mean_x, var_x

    # Calculate Z-score for each pair
    CMIA_df['Zscore'] = np.sqrt(((CMIA_df['MI+maxCMI'] - CMIA_df['X-mean'])**2 / CMIA_df['X-var'])
                                + ((CMIA_df['MI+maxCMI'] - CMIA_df['Y-mean'])**2 / CMIA_df['Y-var']))

    # Convert specific columns to integer
    CMIA_df = CMIA_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    CMIA_df_sorted = CMIA_df.sort_values(by='Zscore', ascending=False)
    CMIA_df_sorted.to_csv(output_filename_with_Zscore, columns=['X', 'Y', 'Zscore'], header=['#X', 'Y', 'Zscore'], index=False)


def CMIA_CLR_vKSG_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    but replacing Synergy with CMI.
    Handles KSG specific case by setting negative MI & TC to 0 and recalculating CMI.
    Written by Lior Shachaf 2021-09-07.

    This function calculates the CMIA+CLR scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])

    # For kNN only: consider setting negative values to zero
    df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']].clip(lower=0)

    # Recalculate CMI(X;Y|Z)
    df['CMI(X;Y|Z)'] = df['TC'] - df['MI(X;Z)'] - df['MI(Y;Z)']

    # Apply first step in SA-CLR
    # df = df[(df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)'])]

    # Initialize DataFrame for storing results
    CMIA_df = pd.DataFrame(columns=['X', 'Y', 'MI+maxCMI', 'X-mean', 'X-std', 'Y-mean', 'Y-std'], dtype=np.float64)

    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1, network_size):
            df_filtered = df[(df['X'] == gene1_index) & (df['Y'] == gene2_index)]

            # Append rows to the DataFrame
            CMIA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['CMI(X;Y|Z)'].max()
            if CMIA_value.size == 0:
                continue
            new_row = pd.DataFrame({'X': [gene1_index], 'Y': [gene2_index], 'MI+maxCMI': [CMIA_value.item()]})
            CMIA_df = pd.concat([CMIA_df, new_row], ignore_index=True)

    # Calculate mean and standard deviation for each gene
    for gene_index in range(network_size):
        df_filtered = CMIA_df[(CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)]

        mean_x = df_filtered['MI+maxCMI'].mean()
        std_x = df_filtered['MI+maxCMI'].std()  # pandas.std() uses ddof=1 while numpy ddof=0

        CMIA_df.loc[CMIA_df['X'] == gene_index, ['X-mean', 'X-std']] = mean_x, std_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, ['Y-mean', 'Y-std']] = mean_x, std_x

    # Calculate Z-score for each pair as in [minet 2008]
    CMIA_df['Zscore_X'] = np.maximum(0, (CMIA_df['MI+maxCMI'] - CMIA_df['X-mean']) / CMIA_df['X-std'])
    CMIA_df['Zscore_Y'] = np.maximum(0, (CMIA_df['MI+maxCMI'] - CMIA_df['Y-mean']) / CMIA_df['Y-std'])
    CMIA_df['Zscore_XY'] = np.sqrt(CMIA_df['Zscore_X']**2 + CMIA_df['Zscore_Y']**2)

    # Convert specific columns to integer
    CMIA_df = CMIA_df.astype({'X': int, 'Y': int})

    # Sort according to Z-score and save to file
    CMIA_df_sorted = CMIA_df.sort_values(by='Zscore_XY', ascending=False)

    # Write output to file
    output_filename_with_Zscore = (f"{expression_data_type}_data_{mi_est_string_with_bins}"
                                   "_CMIA_CLR_vKSG_unsigned_with_Zscore.dat")
    CMIA_df_sorted.to_csv(output_filename_with_Zscore, columns=['X', 'Y', 'Zscore_XY'],
                          header=['#X', 'Y', 'Zscore'], index=False)


def CMI_CMI_inference_algo(filename_data, expression_data_type, bins_or_neighbors, mi_est, network_size):
    """
    Based on: "Learning transcriptional regulatory networks from high throughput gene expression
    data using continuous three-way mutual information"
    Written by Lior Shachaf 2021-03-16

    This function calculates the CMI+CMI scores for gene pairs based on mutual information (MI) values.
    The MI values are calculated using different estimators and the scores are used to infer gene regulatory networks.

    Parameters:
    filename_data (str): Data filename (excluding the ".tsv" extension).
    expression_data_type (str): Expression data type ("SS", "TS", "TSandSS").
    bins_or_neighbors (str): Number of bins or neighbors.
    mi_est (str): Mutual information estimator ("Shannon", "Miller-Madow", "KSG", "KL").
    network_size (int): Size of the network.

    Returns:
    None
    """
    # Load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est, bins_or_neighbors)
    filename_MI_table = f"{filename_data}_MI2andTC_{mi_est_string_with_bins}.dat"
    column_names = ['X', 'Y', 'Z', 'Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names,
                     usecols=['X', 'Y', 'Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])

    # Remove excess pairs as the original df contains all (X,Y) and (Y,X)
    df = df[df['X'] < df['Y']]

    # For kNN only: consider setting negative values to zero
    df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']] = df[['MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)']].clip(lower=0)

    # Recalculate CMI(X;Z|Y) and CMI(Y;Z|X)
    df['CMI(X;Z|Y)'] = df['TC'] - df['MI(X;Y)'] - df['MI(Y;Z)']
    df['CMI(Y;Z|X)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)']
    df['CMI(X;Z|Y)+CMI(Y;Z|X)'] = df['CMI(X;Z|Y)'] + df['CMI(Y;Z|X)']

    # Convert specific columns to integer
    df = df.astype({'X': int, 'Y': int, 'Z': int})

    # Sort according to CMI+CMI and save to file
    df_sorted = df.sort_values(by=['Z', 'CMI(X;Z|Y)+CMI(Y;Z|X)'], ascending=False)
    df_sorted = df_sorted.drop_duplicates(subset=['Z'], ignore_index=True)

    # Create a new DataFrame for R1-T and R2-T pairs
    CMIplusCMI_df = df_sorted[['X', 'Z', 'MI(X;Z)', 'CMI(X;Z|Y)+CMI(Y;Z|X)']].copy()
    CMIplusCMI_df.rename(columns={'X': 'Y', 'MI(X;Z)': 'MI(Y;Z)'}, inplace=True)
    CMIplusCMI_df = pd.concat([CMIplusCMI_df, df_sorted[['Y', 'Z', 'MI(Y;Z)', 'CMI(X;Z|Y)+CMI(Y;Z|X)']].copy()])
    CMIplusCMI_df.rename(columns={'Y': 'R', 'Z': 'T', 'MI(Y;Z)': 'MI(R;T)'}, inplace=True)

    # Arrange pairs as R < T to comply with downstream code
    CMIplusCMI_df['Rnew'] = np.where(CMIplusCMI_df['R'] < CMIplusCMI_df['T'], CMIplusCMI_df['R'], CMIplusCMI_df['T'])
    CMIplusCMI_df['Tnew'] = np.where(CMIplusCMI_df['R'] < CMIplusCMI_df['T'], CMIplusCMI_df['T'], CMIplusCMI_df['R'])
    CMIplusCMI_df.drop(columns=['R', 'T'], inplace=True)
    CMIplusCMI_df.rename(columns={'Rnew': 'R', 'Tnew': 'T'}, inplace=True)
    CMIplusCMI_df = CMIplusCMI_df[['R', 'T', 'MI(R;T)', 'CMI(X;Z|Y)+CMI(Y;Z|X)']]

    # Sort and remove duplicates
    CMIplusCMI_df_sorted = CMIplusCMI_df.sort_values(by=['MI(R;T)', 'CMI(X;Z|Y)+CMI(Y;Z|X)'], ascending=False)
    CMIplusCMI_df_sorted = CMIplusCMI_df_sorted.drop_duplicates(subset=['MI(R;T)'], ignore_index=True)

    # Save to file
    output_filename = f"{expression_data_type}_data_{mi_est_string_with_bins}_CMIplusCMI_unsigned.dat"
    CMIplusCMI_df_sorted.to_csv(output_filename, columns=['R', 'T', 'MI(R;T)'], header=['#X', 'Y', 'MI(X;Y)'], index=False)
