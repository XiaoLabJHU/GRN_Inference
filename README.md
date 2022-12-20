This README file contains the introduction to the repo "GRN_Inference", including short descriptions of the other scripts.<br>
<br>
## Introduction:<br>
This repo contains instructions and scripts to generate data, analyze, and plot as mentioned in our paper ["Gene regulation network inference using k-nearest neighbor-based mutual information estimation- Revisiting an old DREAM"](https://www.biorxiv.org/content/10.1101/2021.12.20.473242v1).<br>
The GRN inference pipeline implemented here is modular, one can use specific functions to calculate MI quantities based on kNN and integrate the output matrix into a different inference algorithm than the ones implemented here.<br>
![GRN inference pipeline:](/GRN_inference_pipeline.svg)<br>

### Prerequisite - Creating a python virtual environment and installing dependencies:<br>
#### Install venv
sudo apt install python3.8-venv

#### In your project folder run the command below. venv will create a virtual Python installation in the env folder.
python3 -m venv env

#### Activating a virtual environment will put the virtual environment-specific python and pip executables into your shell’s PATH.
source env/bin/activate

#### Confirm you are using the python environment
which python

#### Tell pip to install all packages in the requirements file (for specific python project)
python3 -m pip install -r requirements.txt

#### Leaving the virtual environment when done
deactivate<br>

### Simulating/generating gene expression data:<br>
The software GeneNetWeaver used to generate the datasets in the current study is available in the GitHub repository, [https://github.com/tschaffter/genenetweaver](https://github.com/tschaffter/genenetweaver)<br>
We used our Jupyter notebook [Generate_replicates_for_network_inference.ipynb](/CODE/Generate_replicates_for_network_inference.ipynb) to call GeneNetWeaver to generate multiple replicates of steady-state and time-series gene expression datasets for realistic in silico networks of sizes of 50, and 100 genes containing various experimental conditions (knockouts, knockdowns, multifactorial perturbation, etc.).<br>
We then combine all the steady-state expression data to a single file: {network name}_all.tsv<br>
<br>
The data we used for the paper can also be downloaded from the [DATA](/DATA/) folder in this repo.<br>

### Density estimation and MI calculation:<br>
The following python modules: Mutual_Info_based_binning_module.py, Mutual_Info_KNN_nats_module.py, Building_MI_matrices.py are responsible to discretize continous data (i.e. gene expression profiles) to fixed width bins or organize the data according to k-NN. Following by mutual information calculations by one of four estimators ((i) Maximum Likelihood (ML, given by Shannon), (ii) Miller-Madow correction (MM), (iii) Kozachenko-Leonenko (KL), and (iv) Kraskov-Stoögbauer-Grassberger (KSG).<br>

To test the accuracy of each MI estimator, we use the Jupyter ntoebook [Mutual_Information_estimators_comparisson_for_Gaussian_distribution.ipynb](/CODE/Mutual_Information_estimators_comparisson_for_Gaussian_distribution.ipynb) to generate random variables (of different sizes) drawn from a joint Gaussian distribution (with different correlations) and calculate the various 2d and 3d MI quantities. Afterwards the Jupyter ntoebook [CMIAwKSG_paper_Figure2_and_SI.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_Figure2_and_SI.ipynb) compares the calculated results with the analytical solution.<br>

To test the computational costs of each MI estimator, we use the Jupyter ntoebook [Computation_time_vs_data_size.ipynb](/CODE/Computation_time_vs_data_size.ipynb) to generate a representative gene expression data matrix for a network of size 50 (nodes or genes) with up to 1000 data points (conditions/perturbations/time-series) and calculate the time to build 2d and 3d MI matrices. Afterwards the Jupyter ntoebook [CMIAwKSG_paper_Figure6_computation_cost.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_Figure6_computation_cost.ipynb) plots the results.<br>

### GRN inference (including density estimation and MI calculation):<br>
The Jupyter ntoebook [AUPR_calc_for_filling_missing_MI_or_Inference.ipynb](/CODE/AUPR_calc_for_filling_missing_MI_or_Inference.ipynb) together with the python modules: Mutual_Info_based_binning_module.py, Mutual_Info_KNN_nats_module.py, Building_MI_matrices.py, Inference_algo_module.py, Precision_Recall_module.py, is responsible to discretize the continous gene expression data to fixed width bins or organize the data according to k-NN. Following by mutual information calculations by one of four estimators ((i) Maximum Likelihood (ML, given by Shannon), (ii) Miller-Madow correction (MM), (iii) Kozachenko-Leonenko (KL), and (iv) Kraskov-Stoögbauer-Grassberger (KSG). Finally, the network structure is infered by one of six inference algorithms (Relevance Networks, RL; Algorithm for the Reconstruction of Accurate Cellular Networks, ARACNE; Context-Likelihood-Relatedness, CLR; Synergy-Augmented CLR, SA-CLR; Luo et al. MI3; CMI2rt, and our Conditional-Mutual-Information-Augmentation, CMIA) 

### GRN performance evaluation and plotting:<br>
To compare the performence and plot it we use the Jupyter notebook [CMIAwKSG_paper_AUPR_figures_and_SI.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_AUPR_figures_and_SI.ipynb).<br>
Data is organized in a pandas dataframe and plotted using matplotlib.
