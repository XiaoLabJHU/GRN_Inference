This README file contains the introduction to the repo "GRN_Inference", including short descriptions of the other scripts.<br>
<br>
## Introduction:<br>
This repo contains instructions and scripts to generate data, analyze, and plot as mentioned in our paper ["Gene regulation network inference using k-nearest neighbor-based mutual information estimation- Revisiting an old DREAM"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05047-5).<br>
The GRN inference pipeline implemented here is modular, one can use specific functions to calculate MI quantities based on kNN and integrate the output matrix into a different inference algorithm than the ones implemented here.<br>
![GRN inference pipeline:](/GRN_inference_pipeline.svg)<br>

### Prerequisite - Cloning the repo, creating a python virtual environment and installing dependencies:
```
# Clone the project
git clone https://github.com/XiaoLabJHU/GRN_Inference.git

# Go into the project folder
cd GRN_Inference

# Create a virtual Python installation in the .venv folder.
python3 -m venv .venv

# Activating a virtual environment will put the virtual environment-specific python and pip
# executables into your shell’s PATH.
# For Windows OS see: https://docs.python.org/3/library/venv.html
source .venv/bin/activate

# Confirm you are using the python environment
which python

# Tell pip to install all packages in the requirements file (for specific python project)
python3 -m pip install -r requirements.txt

# Leaving the virtual environment when done
deactivate
```

### Simulating/generating gene expression data:<br>
The software GeneNetWeaver used to generate the datasets in the current study is available in the GitHub repository, [https://github.com/tschaffter/genenetweaver](https://github.com/tschaffter/genenetweaver)<br>
We used our Jupyter notebook [Generate_replicates_for_network_inference.py](/CODE/Generate_replicates_for_network_inference.py) to call GeneNetWeaver to generate multiple replicates of steady-state and time-series gene expression datasets for realistic in silico networks of sizes of 50, and 100 genes containing various experimental conditions (knockouts, knockdowns, multifactorial perturbation, etc.).<br>
We then combine all the steady-state expression data to a single file: {network name}_all.tsv<br>
<br>
The data we used for the paper can also be downloaded from the [DATA](/DATA/) folder in this repo.<br>

### Density estimation and MI calculation:<br>
The following python modules: Mutual_Info_based_binning_module.py, Mutual_Info_KNN_nats_module.py, Building_MI_matrices.py are responsible to discretize continuous data (i.e. gene expression profiles) to fixed width bins or organize the data according to k-NN. Following by mutual information calculations by one of four estimators ((i) Maximum Likelihood (ML, given by Shannon), (ii) Miller-Madow correction (MM), (iii) Kozachenko-Leonenko (KL), and (iv) Kraskov-Stoögbauer-Grassberger (KSG).<br>

To test the accuracy of each MI estimator, we use the Python script [Mutual_Information_estimators_comparisson_for_Gaussian_distribution.py](/CODE/Mutual_Information_estimators_comparisson_for_Gaussian_distribution.py) to generate random variables (of different sizes) drawn from a joint Gaussian distribution (with different correlations) and calculate the various 2d and 3d MI quantities. Afterwards the Jupyter notebook [CMIAwKSG_paper_Figure2_and_SI.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_Figure2_and_SI.ipynb) compares the calculated results with the analytical solution.<br>

To test the computational costs of each MI estimator, we use the script [Computation_time_vs_data_size.py](/CODE/Computation_time_vs_data_size.py) to generate a representative gene expression data matrix for a network of size 50 (nodes or genes) with up to 1000 data points (conditions/perturbations/time-series) and calculate the time to build 2d and 3d MI matrices. Afterwards the Jupyter notebook [CMIAwKSG_paper_Figure6_computation_cost.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_Figure6_computation_cost.ipynb) plots the results.<br>

### GRN inference (including density estimation and MI calculation):<br>
The script [GRN_inference_and_AUPR_calc.py](/CODE/GRN_inference_and_AUPR_calc.py) together with the python modules: Mutual_Info_based_binning_module.py, Mutual_Info_KNN_nats_module.py, Building_MI_matrices.py, Inference_algo_module.py, Precision_Recall_module.py, is responsible to discretize the continous gene expression data to fixed width bins or organize the data according to k-NN. Following by mutual information calculations by one of four estimators ((i) Maximum Likelihood (ML, given by Shannon), (ii) Miller-Madow correction (MM), (iii) Kozachenko-Leonenko (KL), and (iv) Kraskov-Stoögbauer-Grassberger (KSG). Finally, the network structure is infered by one of six inference algorithms (Relevance Networks, RL; Algorithm for the Reconstruction of Accurate Cellular Networks, ARACNE; Context-Likelihood-Relatedness, CLR; Synergy-Augmented CLR, SA-CLR; Luo et al. MI3; CMI2rt, and our Conditional-Mutual-Information-Augmentation, CMIA) 

### GRN performance evaluation and plotting:<br>
To compare the performence and plot it we use the Jupyter notebook [CMIAwKSG_paper_AUPR_figures_and_SI.ipynb](/Scripts_for_paper_plots/CMIAwKSG_paper_AUPR_figures_and_SI.ipynb).<br>
Data is organized in a pandas dataframe and plotted using matplotlib.
