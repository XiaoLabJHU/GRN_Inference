#!/usr/bin/env python3.8
"""
This module contains functions to evaluate the perfoemence of GRN inference algo.
It calculate the precision, recall and AUPR of infered networks based on various algo.
Written by Lior Shachaf 2021-02-16
"""

import os
import subprocess
import numpy as np
import Inference_algo_module as inference_module

def PR_calc(trueNet_dict,predictNet):
    """
    Returns the precision vs. recall data in steps of 1 prediction pair at a time
    """
    TP = 0; FP = 0; FN = 0;
    recall = np.zeros(predictNet.shape[0])
    precision = np.zeros(predictNet.shape[0])

    for line in range(predictNet.shape[0]):
        key = 'G' + str(int(predictNet[line][0])) + '-' + 'G' + str(int(predictNet[line][1]))

        #print(key) # testing
        if key not in trueNet_dict.keys(): # Fixing
            raise ValueError(key,'[Predicted key] not in true network')
        #    key = 'G' + str(int(predictNet[line][1])) + '-' + 'G' + str(int(50))   ## Lior - maybe was used before advancing by +1 all predicted pairs
        if trueNet_dict[key] == 1:
                TP += 1
                #print("TP :-)") # Testing
        else:
                FP += 1
                #print("FP :-(") # Testing

        FN = str(trueNet_dict.values()).count('1') - TP
        precision[line] = TP / (TP + FP) # denominator is the total predicted links per threshold
        recall[line] = TP / (TP + FN) # denominator is the total true links
    #print(precision, recall)
    return precision, recall


def import_true_network(network_name_input):
        """Load unsigned true synthetic network structure from DREAM3/4 networks.
        Input: 3 columns (G_i,G_j, boolean) with tab ("\t") separator.
        Output: undirected network pairs in a dictionary structure."""
        
        true_structure_input = network_name_input + "_goldstandard.tsv"
        in1 = open(true_structure_input)
        in1_data = in1.readlines()
        in1.close()

        trueNetwork_dict = {}
        for line in in1_data:
            if int(line.split('\t')[0].strip('G')) > int(line.split('\t')[1].strip('G')):
                key = str(line.split('\t')[1] + '-' + line.split('\t')[0])
            else:
                key = str(line.split('\t')[0] + '-' + line.split('\t')[1])
            value = int(line.split('\t')[2])
            if key not in trueNetwork_dict.keys():
                trueNetwork_dict[key] = value
                
        #print(len(trueNetwork_dict.keys()),trueNetwork_dict.keys())  ### Testing
        return trueNetwork_dict

    
def PR_per_infer_algo(network_name_input,expression_data_type,mi_est_full,infer_algo):

    """Calculate PR curves for 
    different expression data type (SS/TS/TSandSS)
    mi_est_full = [bins_or_neighbor, mi_est] for example [10, Shannon]
    different inference algo: MI2, MI2_Zscore, MI2_ARACNE, TC_Z_score, II_Zsscore, CMI_CMI, SA_CLR, CMIA_CLR"""

    inputFileName = network_name_input + "_" + expression_data_type + "_all"
    
    if "-" in network_name_input: #DREAM3 style = InSilicoSize100-Ecoli1
        network_size = int(network_name_input.split('-')[0].replace("InSilicoSize",'')) # DREAM3
    elif "_" in network_name_input: #DREAM4 style = insilico_size100_1
        network_size = int(network_name_input.split('_')[1].replace("size",'')) # DREAM4
    max_pairs = int(network_size * (network_size-1)/2) #len(trueNetwork_dict.keys())
    bins_or_neighbor = mi_est_full[0]
    mi_est = mi_est_full[1]
    PR_data_array = np.zeros((max_pairs,2)) # Trailing zeros will be trimmed.
    
    ### Load true synthetic network data and save into dictionary
    trueNetwork_dict = import_true_network(network_name_input)
    #print(len(trueNetwork_dict.keys()),trueNetwork_dict.keys())  ### Testing

    if mi_est == "Shannon":
        mi_est_string = "FB" + str(bins_or_neighbor) + "_Shan"
    elif mi_est == "Miller-Madow":
        mi_est_string = "FB" + str(bins_or_neighbor) + "_MM"
    elif mi_est == "KSG":
        mi_est_string = "KNN" + str(bins_or_neighbor) + "_KSG"
    elif mi_est == "KL":
        mi_est_string = "KNN" + str(bins_or_neighbor) + "_KL"

    #print(mi_est_string) ### Testing
    
    if infer_algo == "MI2" or infer_algo == "MI2_Zscore" or infer_algo == "MI2_ARACNE":
        DB_file_name = network_name_input + "_" + expression_data_type + "_all_MI_" + mi_est_string + "_triplets_calc_v3_with_Zscore_v3.dat"
        posMIquantitiesFile = expression_data_type + "_data_" + mi_est_string + "_posTCandMI2.dat"        

        # Depending on expression_data_type (SS/TS/TSandSS), check if sorted files exist, and if not generate them
        #if "True" == "False":  ### For testing
        #if "pos_TCandMI2.dat" in ''.join(os.listdir()):
        if posMIquantitiesFile in ''.join(os.listdir()):
            #print(posMIquantitiesFile," exists ")  ### Testing
            pass
        else:
            print("making new file with all positive MI2 pairs")
            # making new file with all positive MI2 pairs
            #cmd = "awk -F, '$7>0' " + DB_file_name + " | cut -d, -f1,2,7,8,13-16 > pos_MI2.dat"
            cmd = "awk -F, '$7>0 && $13>0' " + DB_file_name + " | cut -d, -f1,2,7,8,13-16 > " + posMIquantitiesFile
            subprocess.run(cmd, shell=True)
            #cmd = "awk -F, '$9>0' " + DB_file_name + " | cut -d, -f1,3,9,10,13-16 >> pos_MI2.dat"
            cmd = "awk -F, '$9>0 && $13>0' " + DB_file_name + " | cut -d, -f1,3,9,10,13-16 >> " + posMIquantitiesFile
            subprocess.run(cmd, shell=True)
            #cmd = "awk -F, '$11>0' " + DB_file_name + " | cut -d, -f2,3,11,12,13-16 >> pos_MI2.dat"
            cmd = "awk -F, '$11>0 && $13>0' " + DB_file_name + " | cut -d, -f2,3,11,12,13-16 >> " + posMIquantitiesFile
            subprocess.run(cmd, shell=True)
    
    if infer_algo == "MI2":
        pos_MI2_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_sorted.dat"
        infer_algo_sorted_file = pos_MI2_sorted_file
        if pos_MI2_sorted_file in ''.join(os.listdir()):
            #print(pos_MI2_sorted_file," exists ")  ### Testing
            pass
        else:
            # Sorting pairs according to MI2 value
            cmd = "cut -d, -f 1-4 " + posMIquantitiesFile + " | sort -t, -urnk3 > " + pos_MI2_sorted_file
            subprocess.run(cmd, shell=True)
        
    elif infer_algo == "MI2_Zscore":
        pos_MI2_Zscore_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_Zscore_sorted.dat"
        infer_algo_sorted_file = pos_MI2_Zscore_sorted_file
        if pos_MI2_Zscore_sorted_file in ''.join(os.listdir()):
            #print(pos_MI2_Zscore_sorted_file," exists ")  ### Testing
            pass
        else:
            # Sorting pairs according to MI2-Z-score
            cmd = "cut -d, -f 1-4 " + posMIquantitiesFile + " | sort -t, -urnk4 > " + pos_MI2_Zscore_sorted_file
            subprocess.run(cmd, shell=True)
        
    elif infer_algo == "TC_Zscore":
        pos_TC_Zscore_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_TC_Zscore_sorted.dat"
        infer_algo_sorted_file = pos_TC_Zscore_sorted_file
        if pos_TC_Zscore_sorted_file in ''.join(os.listdir()):
            #print(pos_TC_Zscore_sorted_file," exists ")  ### Testing
            pass
        else:       
            # Sorting pairs according to TC-Z-score
            #subprocess.run('''sort -t, -rnk6 pos_TCandMI2.dat | awk -F, '{printf "%d %d %d %2.9f %2.9f\n",NR,$1,$2,$3,$4}' | sort -k4,4rn -k1,1n | uniq --skip-fields=1 | sort -nk1 | tr ' ' ',' | cut -d, -f 2-5 > pos_TC_Zscore_sorted.dat''', shell=True)
            cmd = "sort -t, -rnk6 " + posMIquantitiesFile + ''' | awk -F, -v fmt="%d %d %d %2.9f %2.9f\n" '{printf fmt,NR,$1,$2,$3,$4}' | sort -k4,4rn -k1,1n | uniq --skip-fields=1 | sort -nk1 | tr ' ' ',' | cut -d, -f 2-5 > ''' + pos_TC_Zscore_sorted_file
            subprocess.run(cmd, shell=True)
                
    elif infer_algo == "MI2_ARACNE":
        pos_MI2_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_sorted.dat"
        pos_MI2_ARACNE_sorted_file = expression_data_type + "_data_" + mi_est_string + "_pos_MI2_ARACNE_sorted.dat"
        infer_algo_sorted_file = pos_MI2_ARACNE_sorted_file
        if pos_MI2_ARACNE_sorted_file in ''.join(os.listdir()):
            #print(pos_MI2_ARACNE_sorted_file," exists ")  ### Testing
            pass
        else:       
            # Removing lowest MI2 pair in a triplet (ARACNE style)
            cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$7>$11 && $9>$11 {printf fmt,$2,$3,$11,$12}' '''+ DB_file_name + " > unpair_list.dat"
            subprocess.run(cmd, shell=True)
            cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$7>$9 && $11>$9 {printf fmt,$1,$3,$9,$10}' '''+ DB_file_name + " >> unpair_list.dat"
            subprocess.run(cmd, shell=True)
            cmd = '''awk -F, -v fmt="%d,%d,%2.9f,%2.9f\n" '$9>$7 && $11>$7 {printf fmt,$1,$2,$7,$8}' '''+ DB_file_name + " >> unpair_list.dat"
            subprocess.run(cmd, shell=True)
            subprocess.run('''awk -F, '$3>0' unpair_list.dat | sort -t, -urnk3 > unpair_list_sorted.dat''', shell=True)
            cmd = "diff --ignore-all-space --suppress-common-lines " + pos_MI2_sorted_file + ''' unpair_list_sorted.dat | grep '<' | sed 's/< //g' > ''' + pos_MI2_ARACNE_sorted_file
            subprocess.run(cmd, shell=True)
    
    elif infer_algo == "CMI_CMI":
        # based on: "Learning transcriptional regulatory networks from high throughput gene expression data using continuous three-way mutual information"
        CMI_CMI_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CMIplusCMI_signed.dat"
        # Using unsigned data to calculate the AUPR
        infer_algo_sorted_file = CMI_CMI_sorted_file.replace("signed","unsigned")
        if CMI_CMI_sorted_file in ''.join(os.listdir()):
            #print(CMI_CMI_sorted_file," exists ")  ### Testing
            pass
        else:       
            cmd = "~/Dropbox/Roberts/CODE_rsync/CMI_CMI_inference_algo.bash " + inputFileName + " " + expression_data_type + " " + str(bins_or_neighbor) + " " + mi_est
            subprocess.run(cmd, shell=True)
            
    elif infer_algo == "CMI_CMI_pandas":
        # based on: "Learning transcriptional regulatory networks from high throughput gene expression data using continuous three-way mutual information"
        CMI_CMI_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CMIplusCMI_pandas_unsigned.dat"
        # Using unsigned data to calculate the AUPR
        infer_algo_sorted_file = CMI_CMI_sorted_file
        if CMI_CMI_sorted_file in ''.join(os.listdir()):
            #print(CMI_CMI_sorted_file," exists ")  ### Testing
            pass
        else:       
            inference_module.CMI_CMI_pandas_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size)
            
    elif infer_algo == "SA_CLR":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
        SA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_SA_CLR_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = SA_CLR_sorted_file
        if SA_CLR_sorted_file in ''.join(os.listdir()):
            #print(SA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:       
            cmd = "~/Dropbox/Roberts/CODE_rsync/SA_CLR_inference_algo.bash " + inputFileName + " " + expression_data_type + " " + str(bins_or_neighbor) + " " + mi_est
            subprocess.run(cmd, shell=True)
            
    elif infer_algo == "SA_CLR_v2":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
        SA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_SA_CLR_v2_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = SA_CLR_sorted_file
        #if True == False: ### Debug
        if SA_CLR_sorted_file in ''.join(os.listdir()):
            #print(SA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.SA_CLR_v2_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size)
                    
    elif infer_algo == "SA_CLR_vLior":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information"
        SA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_SA_CLR_vLior_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = SA_CLR_sorted_file
        #if True == False: ### Debug
        if SA_CLR_sorted_file in ''.join(os.listdir()):
            #print(SA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.SA_CLR_vLior_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size)                 
    elif infer_algo == "CMIA_CLR":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information" but using CMI instead of Synergy
        CMIA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CMIA_CLR_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = CMIA_CLR_sorted_file
        if CMIA_CLR_sorted_file in ''.join(os.listdir()):
            #print(CMIA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            cmd = "~/Dropbox/Roberts/CODE_rsync/CMIA_CLR_inference_algo.bash " + inputFileName + " " + expression_data_type + " " + str(bins_or_neighbor) + " " + mi_est
            subprocess.run(cmd, shell=True)

    elif infer_algo == "CMIA_CLR_pandas":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information" but using CMI instead of Synergy
        CMIA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CMIA_CLR_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = CMIA_CLR_sorted_file
        if True == False: # Debug
        #if CMIA_CLR_sorted_file in ''.join(os.listdir()):
            #print(CMIA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.CMIA_CLR_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size)
            
    elif infer_algo == "CMIA_CLR_vKSG":
        # based on: "Inference of regulatory gene interactions from expression data using three-way mutual information" but using CMI instead of Synergy
        CMIA_CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CMIA_CLR_vKSG_unsigned_with_Zscore.dat"
        infer_algo_sorted_file = CMIA_CLR_sorted_file
        if True == False: # Debug
        #if CMIA_CLR_sorted_file in ''.join(os.listdir()):
            #print(CMIA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.CMIA_CLR_vKSG_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size)
    
    elif infer_algo == "CLR_pandas":
        # based on Faith 2007 
        CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CLR_unsigned.dat"       
        infer_algo_sorted_file = CLR_sorted_file
        #if True == False: # Debug
        if CLR_sorted_file in ''.join(os.listdir()):
            #print(CMIA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.CLR_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size) # Debug

    elif infer_algo == "CLRvMinet_pandas":
        # based on Faith 2007 
        CLR_sorted_file = expression_data_type + "_data_" + mi_est_string + "_CLRvMinet_unsigned.dat"    
        infer_algo_sorted_file = CLR_sorted_file
        #if True == False: # Debug
        if CLR_sorted_file in ''.join(os.listdir()):
            #print(CMIA_CLR_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.CLRvMinet_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size) # Debug

    elif infer_algo == "RL":
        # based on Butte 2000 
        RL_sorted_file = expression_data_type + "_data_" + mi_est_string + "_RL_unsigned.dat"    
        infer_algo_sorted_file = RL_sorted_file
        #if True == False: # Debug
        if RL_sorted_file in ''.join(os.listdir()):
            #print(RL_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.RL_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size) # Debug
    
    elif infer_algo == "ARACNE":
        # based on Margolin 2006 
        ARACNE_sorted_file = expression_data_type + "_data_" + mi_est_string + "_ARACNE_unsigned.dat"    
        infer_algo_sorted_file = ARACNE_sorted_file
        #if True == False: # Debug
        if ARACNE_sorted_file in ''.join(os.listdir()):
            #print(ARACNE_sorted_file," exists ")  ### Testing
            pass
        else:
            inference_module.ARACNE_inference_algo(inputFileName,expression_data_type,str(bins_or_neighbor),mi_est,network_size) # Debug
    
    else:
        print("inference algo not recognize")
        return
    
    # Loading sorted files and correcting gene count (shifting 1 number up)
    print(infer_algo_sorted_file) # Debugging
    predictNetwork = np.loadtxt(infer_algo_sorted_file, comments='#' , delimiter=',', dtype=int, usecols = (0,1))
    predictNetwork.T[0] = predictNetwork.T[0]+1 ### My data starts from G0 while the goldstandard starts from G1
    predictNetwork.T[1] = predictNetwork.T[1]+1 ### My data starts from G0 while the goldstandard starts from G1
    PR_data_array[:predictNetwork.shape[0],0], PR_data_array[:predictNetwork.shape[0],1] = PR_calc(trueNetwork_dict,predictNetwork)

    return PR_data_array


def AUPR_calc(PR_data):

    """Calculating area under the PR curve using numpy trapezoid calc"""

    AUPR = np.trapz(np.trim_zeros(PR_data[:,0],'b'),x=np.trim_zeros(PR_data[:,1],'b'))
    #print("AUPR = ",AUPR) ### testing

    return AUPR 


def AUPR_replicates_func(topology,expression_data_type,mi_est,infer_algo):
    """
    This function should be run in the DREAM_X folder 
    os.chdir('/home/local/WIN/lshacha1/DATA/Networks/Replicates_for_network_inference/dream3/')  
    """
    
    # writing output file for summary statistics
    output_filename = "AUPR_" + topology + "_" + expression_data_type + "_" + mi_est + "_" + infer_algo + ".dat"
    output_file = open(output_filename,"w")
    
    if os.path.isdir(topology) == True:
        os.chdir(str('./'+topology))
        
        replicates = len(os.listdir())
        AUPR_replicate_array = np.zeros(replicates, dtype=float)
        
        for rep_counter,replicate in enumerate(os.listdir()):

            if os.path.isdir(replicate) == True:
                os.chdir(str('./'+replicate))
                
                for file in os.listdir():
                    if str("_"+expression_data_type) in file and "_triplets_calc_v3_with_Zscore_v3.dat" in file:
                        #if (mi_est == "Shannon" or mi_est == "Miller-Madow") and ("Shan" in file or "MM" in file):
                         #   bins_or_neighbor = file.split("FB")[1].split("_")[0]
                        if (mi_est == "Shannon") and ("Shan" in file):
                            bins_or_neighbor = file.split("FB")[1].split("_")[0]
                        elif (mi_est == "Miller-Madow") and ("MM" in file):
                            bins_or_neighbor = file.split("FB")[1].split("_")[0]
                        elif (mi_est == "KSG") and ("KSG" in file):
                            bins_or_neighbor = file.split("KNN")[1].split("_")[0]
                        elif (mi_est == "KL") and ("KL" in file):
                            bins_or_neighbor = file.split("KNN")[1].split("_")[0]
                        else:
                            continue
                            
                        mi_est_full = [bins_or_neighbor, mi_est]
                        #print(mi_est_full) ### Testing

                        ### Calculating the AURP
                        print("Debug: ",replicate) #Debug
                        AUPR_replicate_array[rep_counter] = AUPR_calc(PR_per_infer_algo(topology,expression_data_type,mi_est_full,infer_algo))
                        #print("AUPR = %.2f" %AUPR_replicate_array[rep_counter]) ### Testing
                        print("Done ",replicate,file)
                        if "-" in topology: #DREAM3 style = InSilicoSize100-Ecoli1
                            output_file.write("%.3f,%d,%s,%s,%s,%d,%s,%s\n" %(AUPR_replicate_array[rep_counter],int(topology.split("-")[0].replace("InSilicoSize",'')),topology.split("-")[1],expression_data_type,mi_est,int(bins_or_neighbor),infer_algo,replicate))
                        elif "_" in topology: #DREAM4 style = insilico_size100_1
                            output_file.write("%.3f,%d,%s,%s,%s,%d,%s,%s\n" %(AUPR_replicate_array[rep_counter],int(topology.split("_")[1].replace("size",'')),topology.split("_")[2],expression_data_type,mi_est,int(bins_or_neighbor),infer_algo,replicate))
                        break

                os.chdir('../')
        
    os.chdir('../')
    output_file.close()
    return AUPR_replicate_array


"""
# Example: Comparing methods for specific synthetic network
#Constants
mi_est = ["Shannon","Miller-Madow","KSG","Shannon","Miller-Madow","KSG"] 
infer_algo = ["MI2","MI2","MI2","MI2_Zscore","MI2_Zscore","MI2_Zscore"] #"MI2_ARACNE"
expression_data_type = "SS"
topology = "InSilicoSize50-Ecoli2"

AUPR_array_for_comparison = np.zeros((10,len(mi_est)))
for counter,(mi,infer) in enumerate(zip(mi_est,infer_algo)):
    #print(counter,mi,infer) ### Testing
    AUPR_array_for_comparison[:,counter] = AUPR_replicates_func(topology,expression_data_type,mi,infer)

print(AUPR_array_for_comparison)
"""
