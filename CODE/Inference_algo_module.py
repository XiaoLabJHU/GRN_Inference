#!/usr/bin/env python3.8
"""
This module contains functions of GRN inference algo (Relevance Networks, ARACNE, CLR, SA-CLR, CMIA-CLR, CMI-CMI).
It generate files with gene pairs based on ranking for each inference algo.
Written by Lior Shachaf 2021-08-20
"""

import numpy as np
import pandas as pd


def mi_est_string_func(mi_est,bins_or_neighbors):
    """This function combines mi_est and bins_or_neighbors in a string to be used in other functions for input and output file names
    Written by: Lior Shachaf
    2021-08-31"""
    
    if mi_est == "Shannon":
        mi_est_string_with_bins = "FB" + bins_or_neighbors + "_Shan"
        #filename_MI_table = filename_data + "_MI_FB" + bins_or_neighbors + "_Shan_triplets_calc_v3.dat"
        #output_filename = expression_data_type + "_data_FB" + bins_or_neighbors + "_Shan_CLR_unsigned.dat"
    elif mi_est == "Miller-Madow":
        mi_est_string_with_bins = "FB" + bins_or_neighbors + "_MM"
    elif mi_est == "KSG":
        mi_est_string_with_bins = "KNN" + bins_or_neighbors + "_KSG"
    elif mi_est == "KL":
        mi_est_string_with_bins = "KNN" + bins_or_neighbors + "_KL"

    return mi_est_string_with_bins


def RL_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: Relevance Networks algo paper [Butte 2000]
    ### Written by Lior Shachaf 2021-08-31
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    # preparing file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
    
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"  ### version prior to Sep-2021
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### new format 
    filename_MI_table = filename_data + "_MI2_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Gene X', 'Gene Y', 'MI(X;Y)']
    
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','MI(X;Y)'])

    # For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
                    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}  
    df = df.astype(convert_dict)

    ### Sort according to MI2 and save to file
    #df_sorted = df.sort_values(by='MI(X;Y)', ascending=False)
    df_sorted = df.sort_values(by=['MI(X;Y)','Y'], ascending=False)
    # Remove duplicate rows and use new index for row name
    df_sorted = df_sorted.drop_duplicates(subset=['MI(X;Y)'], ignore_index=True)
    
    ### save to file
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_RL_unsigned.dat"
    df_sorted.to_csv(output_filename, columns = ['X','Y','MI(X;Y)'], header = ['#X','Y','MI(X;Y)'], index=False)
    
    
def CLR_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: Context-Likelihood-Relevance paper [Faith 2007]
    ### Written by Lior Shachaf 2021-08-27
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    # preparing file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
    
    filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_CLR_unsigned.dat"
        
    ### Read input file and save as pandas data frame
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    df_raw = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','MI(X;Y)'])

    # Remove duplicate rows and use new index for row name
    df = df_raw.drop_duplicates(ignore_index=True)
   
    # For kNN only: consider setting negative values to zero 
    # df.loc[df['MI(X;Y)'] < 0, 'MI(X;Y)'] = 0
   
    #df.to_csv("temp_df.tmp") # Debug
    
    ### open new DataFrame or file for writing
    MI_df = pd.DataFrame(columns = ['X', 'Y', 'MI(X;Y)','X-mean','X-var','Y-mean','Y-var'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            MI_value = df_filtered['MI(X;Y)'].unique()
            if MI_value.size == 0:
                continue
            MI_df = MI_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI(X;Y)' : MI_value.item()}, ignore_index = True)

    ### calculating Z-score for the MI(X;Y) quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = MI_df[((MI_df['X'] == gene_index) | (MI_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI(X;Y)'].mean()
        var_x = df_filtered['MI(X;Y)'].var()
        
        # The following 4 lines can probably be done faster
        MI_df.loc[MI_df['X'] == gene_index, 'X-mean'] = mean_x
        MI_df.loc[MI_df['X'] == gene_index, 'X-var'] = var_x
        MI_df.loc[MI_df['Y'] == gene_index, 'Y-mean'] = mean_x
        MI_df.loc[MI_df['Y'] == gene_index, 'Y-var'] = var_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair as in [Faith 2007]: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    MI_df['Zscore'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] )**2 / MI_df['X-var'] + ( MI_df['MI(X;Y)'] - MI_df['Y-mean'] )**2 / MI_df['Y-var'] )**(1/2)
    ### calc Zscore for each pair as in [minet 2008]: Zscore_X = max( 0, MI(X;Y)-meanX)/stdX ), Zscore_XY = sqrt( (Zscore_X)**2 + (Zscore_Y)**2 ) 
    #MI_df['Zscore'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] )**2 / MI_df['X-var'] + ( MI_df['MI(X;Y)'] - MI_df['Y-mean'] )**2 / MI_df['Y-var'] )**(1/2)
    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    MI_df = MI_df.astype(convert_dict)
    #print(CMIA_df.dtypes) # debug
    #MI_df.to_csv(output_filename,index=False) # Debug

    ### Sort according to Zscore and save to file
    MI_df_sorted = MI_df.sort_values(by='Zscore', ascending=False)
    MI_df_sorted.to_csv(output_filename, columns = ['X','Y','Zscore'], header = ['#X','Y','Zscore'], index=False)
    
    
def CLRvMinet_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: Context-Likelihood-Relevance paper [Faith 2007]
    ### Similar to the previous CLR_inference_algo function but using STD instead of VAR. This increase AUPR by few percent.
    ### Written by Lior Shachaf 2021-08-27
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    # preparing file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
            
    ### Read input file and save as pandas data frame
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### new format when scanning on bin number
    filename_MI_table = filename_data + "_MI2_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Gene X', 'Gene Y', 'MI(X;Y)']
    
    df_raw = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','MI(X;Y)'])

    # Remove duplicate rows and use new index for row name
    df = df_raw.drop_duplicates(ignore_index=True)
   
    # For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
    #df.loc[df['MI(X;Y)'] < 0, 'MI(X;Y)'] = 0
   
    #df.to_csv("temp_df.tmp") # Debug
    
    ### open new DataFrame or file for writing
    MI_df = pd.DataFrame(columns = ['X', 'Y', 'MI(X;Y)','X-mean','X-std','Y-mean','Y-std'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            MI_value = df_filtered['MI(X;Y)'].unique()
            if MI_value.size == 0:
                continue
            MI_df = MI_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI(X;Y)' : MI_value.item()}, ignore_index = True)

    ### calculating Z-score for the MI(X;Y) quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = MI_df[((MI_df['X'] == gene_index) | (MI_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI(X;Y)'].mean()
        std_x = df_filtered['MI(X;Y)'].std() ### NOTE: pandas.std() uses ddof=1 while numpy ddof=0 => checked on dream3/Yeast1-size50 and found no difference
        
        # The following 4 lines can probably be done faster
        MI_df.loc[MI_df['X'] == gene_index, 'X-mean'] = mean_x
        MI_df.loc[MI_df['X'] == gene_index, 'X-std'] = std_x
        MI_df.loc[MI_df['Y'] == gene_index, 'Y-mean'] = mean_x
        MI_df.loc[MI_df['Y'] == gene_index, 'Y-std'] = std_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair as in [Faith 2007]: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    #MI_df['Zscore'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] )**2 / MI_df['X-var'] + ( MI_df['MI(X;Y)'] - MI_df['Y-mean'] )**2 / MI_df['Y-var'] )**(1/2)
    ### calc Zscore for each pair as in [minet 2008]: Zscore_X = max( 0, MI(X;Y)-meanX)/stdX ), Zscore_XY = sqrt( (Zscore_X)**2 + (Zscore_Y)**2 ) 
    MI_df['Zscore_X'] = np.maximum( 0, ( MI_df['MI(X;Y)'] - MI_df['X-mean'] ) / MI_df['X-std'] )
    #MI_df['Zscore_X'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] ) / MI_df['X-std'] ).clip(lower=0)
    MI_df['Zscore_Y'] = np.maximum( 0, ( MI_df['MI(X;Y)'] - MI_df['Y-mean'] ) / MI_df['Y-std'] )
    
    MI_df['Zscore_XY'] = ( ( MI_df['Zscore_X'] )**2 + ( MI_df['Zscore_Y'] )**2 )**(1/2)
    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
    MI_df = MI_df.astype(convert_dict)
    
    ### Sort according to Zscore and save to file
    MI_df_sorted = MI_df.sort_values(by='Zscore_XY', ascending=False)
    
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_CLRvMinet_unsigned.dat"
    MI_df_sorted.to_csv(output_filename, columns = ['X','Y','Zscore_XY'], header = ['#X','Y','Zscore'], index=False)  
    #MI_df_sorted.to_csv(output_filename, index=False)  # Debug
        
    
def SA_CLR_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    ### Written by Lior Shachaf 2021-08-31
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    # preparing file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
    
    filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_SA_CLR_unsigned.dat"
    output_filename_with_Zscore = expression_data_type + "_data_" + mi_est_string_with_bins + "_SA_CLR_unsigned_with_Zscore.dat"
        
    ### Preparing temp file where MI(i;j) > MI(i;k) and MI(j;k) > MI(i;k)
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'II(XYZ)'])
    
    ### For kNN only: consider setting negative values to zero 
    #df.loc[df['MI(X;Y)'] < 0, 'MI(X;Y)'] = 0
    #df.loc[df['MI(X;Z)'] < 0, 'MI(X;Z)'] = 0
    #df.loc[df['MI(Y;Z)'] < 0, 'MI(Y;Z)'] = 0

    ### Apply first step in SA-CLR
    df = df[((df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)']))]
    #df.to_csv("temp_df.tmp") # Debug
    
    ### output data structure: G_i,G_j,MI(i;j)+max(II(i;j;k))    
    ### open new DataFrame or file for writing
    SA_df = pd.DataFrame(columns = ['X', 'Y', 'MI+maxII','X-mean','X-var','Y-mean','Y-var'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].max()
            if CMIA_value.size == 0:
                continue
            SA_df = SA_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI+maxII' : SA_value.item()}, ignore_index = True)

    ### calculating Z-score for the new MI+II quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = SA_df[((SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI+maxII'].mean()
        var_x = df_filtered['MI+maxII'].var()
        
        # The following 4 lines can probably be done faster
        SA_df.loc[SA_df['X'] == gene_index, 'X-mean'] = mean_x
        SA_df.loc[SA_df['X'] == gene_index, 'X-var'] = var_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-mean'] = mean_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-var'] = var_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    SA_df['Zscore'] = ( ( SA_df['MI+maxII'] - SA_df['X-mean'] )**2 / SA_df['X-var'] + ( SA_df['MI+maxII'] - SA_df['Y-mean'] )**2 / SA_df['Y-var'] )**(1/2)
    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    SA_df = SA_df.astype(convert_dict)
    
    ### Sort according to Zscore and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore', ascending=False)
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns = ['X','Y','Zscore'], header = ['#X','Y','Zscore'], index=False)
    

def SA_CLR_v2_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"i
    ### Similar to the previous SA_CLR_inference_algo function but using STD instead of VAR. Like the CLR version in Minet.
    ### Written by Lior Shachaf 2021-08-31
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    ### load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)    
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### for scanning k=1..10
    filename_MI_table = filename_data + "_MI2andTC_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'] 
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)','TC'])
                   
    ### For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
    
    ### recalc CMI(X;Y|Z)
    df['II(XYZ)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)'] - df['MI(Y;Z)']
    
    ### Apply first step in SA-CLR => this gives lower AUPR for Dream3-50 gene networks
    #df = df[((df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)']))]
    #df.to_csv("temp_df.tmp") # Debug
    
    ### output data structure: G_i,G_j,MI(i;j)+max(II(i;j;k))    
    ### open new DataFrame or file for writing
    SA_df = pd.DataFrame(columns = ['X', 'Y', 'MI+maxII','X-mean','X-std','Y-mean','Y-std'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].max()
            if SA_value.size == 0:
                continue
            SA_df = SA_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI+maxII' : SA_value.item()}, ignore_index = True)

    ### calculating Z-score for the new MI+II quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = SA_df[((SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI+maxII'].mean()
        std_x = df_filtered['MI+maxII'].std() ### NOTE: pandas.std() uses ddof=1 while numpy ddof=0 
          
        # The following 4 lines can probably be done faster
        SA_df.loc[SA_df['X'] == gene_index, 'X-mean'] = mean_x
        SA_df.loc[SA_df['X'] == gene_index, 'X-std'] = std_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-mean'] = mean_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-std'] = std_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    #SA_df['Zscore'] = ( ( SA_df['MI+maxII'] - SA_df['X-mean'] )**2 / SA_df['X-var'] + ( SA_df['MI+maxII'] - SA_df['Y-mean'] )**2 / SA_df['Y-var'] )**(1/2)
    ### calc Zscore for each pair as in [minet 2008]: Zscore_X = max( 0, MI(X;Y)-meanX)/stdX ), Zscore_XY = sqrt( (Zscore_X)**2 + (Zscore_Y)**2 ) 
    SA_df['Zscore_X'] = np.maximum( 0, ( SA_df['MI+maxII'] - SA_df['X-mean'] ) / SA_df['X-std'] )
    #MI_df['Zscore_X'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] ) / MI_df['X-std'] ).clip(lower=0)
    SA_df['Zscore_Y'] = np.maximum( 0, ( SA_df['MI+maxII'] - SA_df['Y-mean'] ) / SA_df['Y-std'] )
    SA_df['Zscore_XY'] = ( ( SA_df['Zscore_X'] )**2 + ( SA_df['Zscore_Y'] )**2 )**(1/2)
        
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    SA_df = SA_df.astype(convert_dict)
    
    ### Sort according to Zscore and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore_XY', ascending=False)
    
    ### write output to file
    output_filename_with_Zscore = expression_data_type + "_data_" + mi_est_string_with_bins + "_SA_CLR_v2_unsigned_with_Zscore.dat"
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns = ['X','Y','Zscore_XY'], header = ['#X','Y','Zscore'], index=False)
    
    
def SA_CLR_vLior_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
    ### changed MI2+maxII to MI2+minIIi, rest is same as SA_CLR_v2_inference_algo
    ### Written by Lior Shachaf 2021-09-14
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    # preparing file names for input and output
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
    
    filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    output_filename_with_Zscore = expression_data_type + "_data_" + mi_est_string_with_bins + "_SA_CLR_vLior_unsigned_with_Zscore.dat"
        
    ### Preparing temp file where MI(i;j) > MI(i;k) and MI(j;k) > MI(i;k)
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'])
    
    ### For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
    
    ### recalc CMI(X;Y|Z)
    df['II(XYZ)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)'] - df['MI(Y;Z)']
    
    ### Apply first step in SA-CLR => this gives lower AUPR for Dream3-50 gene networks
    #df = df[((df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)']))]
    #df.to_csv("temp_df.tmp") # Debug
    
    ### output data structure: G_i,G_j,MI(i;j)+max(II(i;j;k))    
    ### open new DataFrame or file for writing
    SA_df = pd.DataFrame(columns = ['X', 'Y', 'MI+minII','X-mean','X-std','Y-mean','Y-std'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            ### append rows to an empty DataFrame
            #SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].max() ### Original algo
            SA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['II(XYZ)'].min() ### Lior's modification to kill co-reg
            if SA_value.size == 0:
                continue
            SA_df = SA_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI+minII' : SA_value.item()}, ignore_index = True)

    ### calculating Z-score for the new MI+II quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = SA_df[((SA_df['X'] == gene_index) | (SA_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI+minII'].mean()
        std_x = df_filtered['MI+minII'].std() ### NOTE: pandas.std() uses ddof=1 while numpy ddof=0 
          
        # The following 4 lines can probably be done faster
        SA_df.loc[SA_df['X'] == gene_index, 'X-mean'] = mean_x
        SA_df.loc[SA_df['X'] == gene_index, 'X-std'] = std_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-mean'] = mean_x
        SA_df.loc[SA_df['Y'] == gene_index, 'Y-std'] = std_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    #SA_df['Zscore'] = ( ( SA_df['MI+maxII'] - SA_df['X-mean'] )**2 / SA_df['X-var'] + ( SA_df['MI+maxII'] - SA_df['Y-mean'] )**2 / SA_df['Y-var'] )**(1/2)
    ### calc Zscore for each pair as in [minet 2008]: Zscore_X = max( 0, MI(X;Y)-meanX)/stdX ), Zscore_XY = sqrt( (Zscore_X)**2 + (Zscore_Y)**2 ) 
    SA_df['Zscore_X'] = np.maximum( 0, ( SA_df['MI+minII'] - SA_df['X-mean'] ) / SA_df['X-std'] )
    #MI_df['Zscore_X'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] ) / MI_df['X-std'] ).clip(lower=0)
    SA_df['Zscore_Y'] = np.maximum( 0, ( SA_df['MI+minII'] - SA_df['Y-mean'] ) / SA_df['Y-std'] )
    SA_df['Zscore_XY'] = ( ( SA_df['Zscore_X'] )**2 + ( SA_df['Zscore_Y'] )**2 )**(1/2)
        
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    SA_df = SA_df.astype(convert_dict)
    
    ### Sort according to Zscore and save to file
    SA_df_sorted = SA_df.sort_values(by='Zscore_XY', ascending=False)
    SA_df_sorted.to_csv(output_filename_with_Zscore, columns = ['X','Y','MI+minII','Zscore_XY'], header = ['#X','Y','MI+minII','Zscore'], index=False)
    
    
def CMIA_CLR_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Inference of regulatory gene interactions from expression data using three-way mutual information" but replacing Synergy with CMI
    ### Written by Lior Shachaf 2021-07-23
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)
    
    filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    output_filename_with_Zscore = expression_data_type + "_data_" + mi_est_string_with_bins + "_CMIA_CLR_unsigned_with_Zscore.dat"
        
    ### Preparing temp file where MI(i;j) > MI(i;k) and MI(j;k) > MI(i;k)
    """awk -F, '$7>$8 && $9>$8' $filename_MI_table > CMIA_CLR.tmp"""
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'CMI(X;Y|Z)'])
    
    ### For kNN only: consider setting negative values to zero 
    #df.loc[df['MI(X;Y)'] < 0, 'MI(X;Y)'] = 0
    #df.loc[df['MI(X;Z)'] < 0, 'MI(X;Z)'] = 0
    #df.loc[df['MI(Y;Z)'] < 0, 'MI(Y;Z)'] = 0
    
    ### Apply first step in SA-CLR
    df = df[((df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)']))]
    #df.to_csv("temp_df.tmp") # Debug
    
    ### output data structure: G_i,G_j,MI(i;j)+max(II(i;j;k))
    """
    for g1 in `seq 0 $((${max_genes}-2))`;
    do
        for g2 in `seq $((${g1}+1)) $((${max_genes}-1))`;
        do
            awk -v g11=$g1 -v g22=$g2 -F, '$1==g11 && $2==g22' CMIA_CLR.tmp | sort -t, -rnk15 | head -1 | awk -F, '{printf "%d,%d,%2.9f\n",$1,$2,$7+$15}' >> $output_filename
            # 20210723: column 15 =  (CMI(X;Y|Z)) while 11 = (II(X;Y;Z))
        done
    done"""
    
    ### open new DataFrame or file for writing
    CMIA_df = pd.DataFrame(columns = ['X', 'Y', 'MI+maxCMI','X-mean','X-var','Y-mean','Y-var'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            CMIA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['CMI(X;Y|Z)'].max()
            if CMIA_value.size == 0:
                continue
            CMIA_df = CMIA_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI+maxCMI' : CMIA_value.item()}, ignore_index = True)

    ### calculating Z-score for the new MI+CMI quantity by first calculating mean and variance for each gene
    """
    for g1 in `seq 0 $((${max_genes}-1))`;
    do
        awk -v g11=$g1 -F, '$1==g11 || $2==g11 { for(i=3;i<=NF;i++) {total[i]+=$i ; sq[i]+=$i*$i ; } }
        END {
           printf "%3s,", g11 ;
           for(i=3;i<=NF;i++) printf "%2.9f,%2.9f\n", total[i]/NR, sq[i]/NR-(total[i]/NR)**2 ;
        }' $output_filename >> mean_and_variance.tmp
    done"""
    for gene_index in range(network_size):
        
        df_filtered = CMIA_df[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI+maxCMI'].mean()
        var_x = df_filtered['MI+maxCMI'].var()
        
        # The following 4 lines can probably be done faster
        CMIA_df.loc[CMIA_df['X'] == gene_index, 'X-mean'] = mean_x
        CMIA_df.loc[CMIA_df['X'] == gene_index, 'X-var'] = var_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, 'Y-mean'] = mean_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, 'Y-var'] = var_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
    
    ### calc Zscore for each pair: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    CMIA_df['Zscore'] = ( ( CMIA_df['MI+maxCMI'] - CMIA_df['X-mean'] )**2 / CMIA_df['X-var'] + ( CMIA_df['MI+maxCMI'] - CMIA_df['Y-mean'] )**2 / CMIA_df['Y-var'] )**(1/2)
    #print(CMIA_df) # debug
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    CMIA_df = CMIA_df.astype(convert_dict)
    #print(CMIA_df.dtypes) # debug
    #CMIA_df.to_csv(output_filename,index=False) # Debug

    """
    for g1 in `seq 0 $((${max_genes}-2))`;
    do
        mean1=`awk -v g11=$g1 -F, '$1==g11 {print $2}' mean_and_variance.tmp`
        var1=`awk -v g11=$g1 -F, '$1==g11 {print $3}' mean_and_variance.tmp`
        for g2 in `seq $((${g1}+1)) $((${max_genes}-1))`;
        do
            mean2=`awk -v g22=$g2 -F, '$1==g22 {print $2}' mean_and_variance.tmp`
            var2=`awk -v g22=$g2 -F, '$1==g22 {print $3}' mean_and_variance.tmp`
            value=`awk -v g11=$g1 -v g22=$g2  -F, '$1==g11 && $2==g22 {print $3}' ${output_filename}` 
            #echo $(( ( (${value}-${mean1})**2/${var1} + (${value}-${mean2})**2/${var2} )**(1/2) )) | bc #>> ${output_filename}_with_Zscore.dat
            # Using bc = Basic Calculator
            Zscore=`echo "sqrt((${value}-${mean1})^2/${var1}+(${value}-${mean2})^2/${var2})" | bc -l` 
            echo "$g1,$g2,$Zscore" >> ${output_filename_with_Zscore}.tmp
            #awk -v g11=$g1 -v g22=$g2  -F, '$1==g11 && $2==g22 {printf "%d,%d,%2.9f,%2.9f\n",$1,$2,$7+$15}' $output_filename >> ${output_filename}_with_Zscore.dat
        done
    done"""

    ### Sort according to Zscore and save to file
    """sort -t, -rnk3 ${output_filename_with_Zscore}.tmp > ${output_filename_with_Zscore} """
    CMIA_df_sorted = CMIA_df.sort_values(by='Zscore', ascending=False)
    CMIA_df_sorted.to_csv(output_filename_with_Zscore, columns = ['X','Y','Zscore'], header = ['#X','Y','Zscore'], index=False)
    #CMIA_df_sorted.to_csv(output_filename_with_Zscore_debug, columns = ['X','Y','Zscore'], header = ['#X','Y','Zscore'], index=False)
    

def CMIA_CLR_vKSG_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Inference of regulatory gene interactions from expression data using three-way mutual information" but replacing Synergy with CMI
    ### Handles KSG specific case by setting negative MI & TC to 0 and recalc CMI
    ### Written by Lior Shachaf 2021-09-07
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    
    ### load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)    
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### for scanning k=1..10
    filename_MI_table = filename_data + "_MI2andTC_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'] 
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)','TC'])
    
    ### Preparing temp file where MI(i;j) > MI(i;k) and MI(j;k) > MI(i;k)
    """awk -F, '$7>$8 && $9>$8' $filename_MI_table > CMIA_CLR.tmp"""
   
    ### For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
    
    ### recalc CMI(X;Y|Z)
    df['CMI(X;Y|Z)'] = df['TC'] - df['MI(X;Z)'] - df['MI(Y;Z)']
    
    ### Apply first step in SA-CLR
    #df = df[((df['MI(X;Y)'] > df['MI(X;Z)']) & (df['MI(Y;Z)'] > df['MI(X;Z)']))]
    #df.to_csv("temp_df.tmp") # Debug
    
    ### open new DataFrame or file for writing
    CMIA_df = pd.DataFrame(columns = ['X', 'Y', 'MI+maxCMI','X-mean','X-var','Y-mean','Y-var'])
        
    for gene1_index in range(network_size - 1):
        for gene2_index in range(gene1_index + 1,network_size):
        
            df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index))]

            # append rows to an empty DataFrame
            CMIA_value = df_filtered['MI(X;Y)'].unique() + df_filtered['CMI(X;Y|Z)'].max()
            if CMIA_value.size == 0:
                continue
            CMIA_df = CMIA_df.append({'X' : gene1_index, 'Y' : gene2_index, 'MI+maxCMI' : CMIA_value.item()}, ignore_index = True)

    ### calculating Z-score for the new MI+CMI quantity by first calculating mean and variance for each gene
    for gene_index in range(network_size):
        
        df_filtered = CMIA_df[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index))]
        
        mean_x = df_filtered['MI+maxCMI'].mean()
        std_x = df_filtered['MI+maxCMI'].std() ### NOTE: pandas.std() uses ddof=1 while numpy ddof=0 
          
        # The following 4 lines can probably be done faster
        CMIA_df.loc[CMIA_df['X'] == gene_index, 'X-mean'] = mean_x
        CMIA_df.loc[CMIA_df['X'] == gene_index, 'X-std'] = std_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, 'Y-mean'] = mean_x
        CMIA_df.loc[CMIA_df['Y'] == gene_index, 'Y-std'] = std_x
        
        ### Attempt to combine 2 lines from above into one to save time
        #CMIA_df.loc[((CMIA_df['X'] == gene_index) | (CMIA_df['Y'] == gene_index)), ['X-mean','Y-mean']] = mean_x
     
    ### calc Zscore for each pair: Zscore=( (value-mean1)**2/var1 + (value-mean2)**2/var2 )**(1/2)
    #CMIA_df['Zscore'] = ( ( CMIA_df['MI+maxCMI'] - CMIA_df['X-mean'] )**2 / CMIA_df['X-var'] + ( CMIA_df['MI+maxCMI'] - CMIA_df['Y-mean'] )**2 / CMIA_df['Y-var'] )**(1/2)
    ### calc Zscore for each pair as in [minet 2008]: Zscore_X = max( 0, MI(X;Y)-meanX)/stdX ), Zscore_XY = sqrt( (Zscore_X)**2 + (Zscore_Y)**2 ) 
    CMIA_df['Zscore_X'] = np.maximum( 0, ( CMIA_df['MI+maxCMI'] - CMIA_df['X-mean'] ) / CMIA_df['X-std'] )
    #MI_df['Zscore_X'] = ( ( MI_df['MI(X;Y)'] - MI_df['X-mean'] ) / MI_df['X-std'] ).clip(lower=0)
    CMIA_df['Zscore_Y'] = np.maximum( 0, ( CMIA_df['MI+maxCMI'] - CMIA_df['Y-mean'] ) / CMIA_df['Y-std'] )
    CMIA_df['Zscore_XY'] = ( ( CMIA_df['Zscore_X'] )**2 + ( CMIA_df['Zscore_Y'] )**2 )**(1/2)
    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}
  
    CMIA_df = CMIA_df.astype(convert_dict)
    #print(CMIA_df.dtypes) # debug
    #CMIA_df.to_csv(output_filename,index=False) # Debug

    ### Sort according to Zscore and save to file
    CMIA_df_sorted = CMIA_df.sort_values(by='Zscore_XY', ascending=False)
    
    ### write output to file
    output_filename_with_Zscore = expression_data_type + "_data_" + mi_est_string_with_bins + "_CMIA_CLR_vKSG_unsigned_with_Zscore.dat"
    CMIA_df_sorted.to_csv(output_filename_with_Zscore, columns = ['X','Y','Zscore_XY'], header = ['#X','Y','Zscore'], index=False)
    
def ARACNE_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: ARACNE algo paper [Margolin 2006]
    ### New python only version (without Unix commands)
    ### Written by Lior Shachaf 2021-12-02
    ### This function get 4 variables as input: 
    # 1) data filename (not including the ".tsv" extension)
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
        
    ### load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)    
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### for scanning k=1..10
    filename_MI_table = filename_data + "_MI2andTC_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'] 
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)','TC'])
    
    # Removes excess pairs as the original df contains all (X,Y) and (Y,X)
    df.drop(df[((df['X'] > df['Y']) | (df['Y'] > df['Z']))].index, inplace = True)
    
    ### For kNN only: consider setting negative values to zero 
    df[df < 0] = 0
    
    ### From old script
    #pos_MI2_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_sorted.dat"
    #pos_MI2_ARACNE_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_ARACNE_sorted.dat"
    #
    # Removing lowest MI2 pair in a triplet (ARACNE style)
    #cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$7>$11 && $9>$11 {printf fmt,$2,$3,$11,$12}' '''+ DB_file_name + " > unpair_list.dat"
    #subprocess.run(cmd, shell=True)
    #cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$7>$9 && $11>$9 {printf fmt,$1,$3,$9,$10}' '''+ DB_file_name + " >> unpair_list.dat"
    #subprocess.run(cmd, shell=True)
    #cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$9>$7 && $11>$7 {printf fmt,$1,$2,$7,$8}' '''+ DB_file_name + " >> unpair_list.dat"
    #subprocess.run(cmd, shell=True)
    #
    #subprocess.run('''awk -F, '$3>0' unpair_list.dat | sort -t, -urnk3 > unpair_list_sorted.dat''', shell=True)
    #
    #cmd = "diff --ignore-all-space --suppress-common-lines " + pos_MI2_sorted_file + ''' unpair_list_sorted.dat | grep '<' | sed 's/< //g' > ''' + pos_MI2_ARACNE_sorted_file
    #subprocess.run(cmd, shell=True)
    ### End of old script
    
    ### open new DataFrame or file for writing
    tmp_df = df[['X', 'Y', 'MI(X;Y)']].copy() 
        
    for gene1_index in range(network_size - 2):
        for gene2_index in range(gene1_index + 1,network_size-1):
            for gene3_index in range(gene2_index + 1,network_size):
                df_filtered = df[((df['X'] == gene1_index) & (df['Y'] == gene2_index) & (df['Z'] == gene3_index))]
                #print(df_filtered) # debug

                # remove rows according to DPI
                #print(df_filtered['MI(X;Y)']) # debug
                if df_filtered['MI(X;Y)'].item() < df_filtered['MI(X;Z)'].item() and df_filtered['MI(X;Y)'].item() < df_filtered['MI(Y;Z)'].item():
                    tmp_df.drop(tmp_df[((tmp_df['X'] == gene1_index) & (tmp_df['Y'] == gene2_index))].index, inplace = True)
                    #print(gene1_index,gene2_index,gene3_index) # debug
                    break
       
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int}  
    tmp_df = tmp_df.astype(convert_dict)
    
    ### Sort according to MI2 and save to file
    #df_sorted = df.sort_values(by='MI(X;Y)', ascending=False)
    df_sorted = tmp_df.sort_values(by=['MI(X;Y)','Y'], ascending=False)
    # Remove duplicate rows and use new index for row name
    df_sorted = df_sorted.drop_duplicates(subset=['MI(X;Y)'], ignore_index=True)
    
    ### save to file
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_ARACNE_unsigned.dat"
    df_sorted.to_csv(output_filename, columns = ['X','Y','MI(X;Y)'], header = ['#X','Y','MI(X;Y)'], index=False)
    
def CMI_CMI_pandas_inference_algo(filename_data,expression_data_type,bins_or_neighbors,mi_est,network_size):
    """
    ### based on: "Learning transcriptional regulatory networks from high throughput gene expression data using continuous three-way mutual information"
    ### Written by Lior Shachaf 2021-03-16
    ### This function get 4 variables as input:
    # 1) data filename
    # 2) expression data type: SS/TS/TSandSS
    # 3) number of bins or neighbors
    # 4) MI estimator: "Shannon" or "Miller-Madow" or "KSG" or "KL"
    """
    ### load DB file
    mi_est_string_with_bins = mi_est_string_func(mi_est,bins_or_neighbors)    
    #filename_MI_table = filename_data + "_MI_" + mi_est_string_with_bins + "_triplets_calc_v3.dat"
    #column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC', 'II(XYZ)', 'MI3((X,Y);Z)', 'MI3((Z,X);Y)', 'MI3((Y,Z);X)', 'CMI(X;Y|Z)', 'CMI(Z;X|Y)', 'CMI(Y;Z|X)']
    ### for scanning k=1..10
    filename_MI_table = filename_data + "_MI2andTC_" + mi_est_string_with_bins + ".dat"
    column_names = ['X','Y','Z','Gene X', 'Gene Y', 'Gene Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)', 'TC'] 
    df = pd.read_csv(filename_MI_table, comment='#', names=column_names, usecols=['X','Y','Z', 'MI(X;Y)', 'MI(X;Z)', 'MI(Y;Z)','TC'])
    
    ### Removes excess pairs as the original df contains all (X,Y) and (Y,X)
    #df.drop(df[((df['X'] > df['Y']) | (df['Y'] > df['Z']))].index, inplace = True)
    df.drop(df[(df['X'] > df['Y'])].index, inplace = True)
    
    ### For kNN only: consider setting negative values to zero 
    df[df < 0] = 0

    ### output data structure: R1,R2,T,MI(R1;T),MI(R2;T),CMI(T;R1|R2)+CMI(T;R2|R1),TC
    ### recalc CMI(X;Y|Z)
    #df['CMI(X;Y|Z)'] = df['TC'] - df['MI(X;Z)'] - df['MI(Y;Z)']
    df['CMI(X;Z|Y)'] = df['TC'] - df['MI(X;Y)'] - df['MI(Y;Z)']
    df['CMI(Y;Z|X)'] = df['TC'] - df['MI(X;Y)'] - df['MI(X;Z)']
    df['CMI(X;Z|Y)+CMI(Y;Z|X)'] = df['CMI(X;Z|Y)'] + df['CMI(Y;Z|X)']
    
    # using dictionary to convert specific columns
    convert_dict = {'X': int, 'Y': int, 'Z': int}  
    df = df.astype(convert_dict)
    
    ### Sort according to CMI+CMI and save to file
    df_sorted = df.sort_values(by=['Z','CMI(X;Z|Y)+CMI(Y;Z|X)'], ascending=False)
    #print(df_sorted) # debug
    # Remove duplicate rows and use new index for row name
    df_sorted = df_sorted.drop_duplicates(subset=['Z'], ignore_index=True)
    
    #print(df_sorted) # debug
    
    ### open new DataFrame or file for writing: R1 or R2, T, MI(R1 or R2;T), CMI(T;R1|R2)+CMI(T;R2|R1)
    # save R1-T pairs to a new DF
    CMIplusCMI_df = df_sorted[['X', 'Z', 'MI(X;Z)','CMI(X;Z|Y)+CMI(Y;Z|X)']].copy()
    # Change columns name temporarly to append other columns
    CMIplusCMI_df.rename(columns={'X':'Y','MI(X;Z)':'MI(Y;Z)'}, inplace=True)
    # append R2-T pairs
    CMIplusCMI_df = CMIplusCMI_df.append(df_sorted[['Y', 'Z', 'MI(Y;Z)','CMI(X;Z|Y)+CMI(Y;Z|X)']].copy())
    # Change columns name to be consisted with other inference algorithms output
    #CMIplusCMI_df.rename(columns={'Y':'X','Z':'Y','MI(Y;Z)':'MI(X;Y)'}, inplace=True)
    CMIplusCMI_df.rename(columns={'Y':'R','Z':'T','MI(Y;Z)':'MI(R;T)'}, inplace=True)
    
    # To arrange pairs as R < T to comply with downstream code we add this extra step
    CMIplusCMI_df['Rnew'] = np.where(CMIplusCMI_df['R'] < CMIplusCMI_df['T'], CMIplusCMI_df['R'], CMIplusCMI_df['T'])
    CMIplusCMI_df['Tnew'] = np.where(CMIplusCMI_df['R'] < CMIplusCMI_df['T'], CMIplusCMI_df['T'], CMIplusCMI_df['R'])
    CMIplusCMI_df.drop(columns=['R','T'], inplace=True)
    CMIplusCMI_df.rename(columns={'Rnew':'R','Tnew':'T'}, inplace=True)
    CMIplusCMI_df = CMIplusCMI_df [['R', 'T', 'MI(R;T)','CMI(X;Z|Y)+CMI(Y;Z|X)']]
    
    # Next sorting will determine ranking of pairs and the final AUPR value. Different sort can be tested
    CMIplusCMI_df_sorted = CMIplusCMI_df.sort_values(by=['MI(R;T)','CMI(X;Z|Y)+CMI(Y;Z|X)'], ascending=False)
    #CMIplusCMI_df_sorted = CMIplusCMI_df.sort_values(by=['CMI(X;Z|Y)+CMI(Y;Z|X)', 'MI(R;T)'], ascending=False) ### This is worse (x2) than first sorting by MI
    CMIplusCMI_df_sorted = CMIplusCMI_df_sorted.drop_duplicates(subset=['MI(R;T)'], ignore_index=True)
    
    #print(CMIplusCMI_df_sorted)
    
    ### save to file
    output_filename = expression_data_type + "_data_" + mi_est_string_with_bins + "_CMIplusCMI_pandas_unsigned.dat"
    CMIplusCMI_df_sorted.to_csv(output_filename, columns = ['R','T','MI(R;T)'], header = ['#X','Y','MI(X;Y)'], index=False)