This README file contains the introduction to the repo "GRN_Inference", including short descriptions of the other scripts.<br>
<br>
## Introduction:<br>
This repo contains instructions and scripts to generate data, analyze, and plot as mentioned in our paper "Gene regulation network inference using k-nearest neighbor-based mutual information estimation- Revisiting an old DREAM".<br>
The GRN inference pipeline implemented here is modular, one can use specific functions to calculate MI quantities based on kNN and integrate the output matrix into a different inference algorithm than the ones implemented here.<br>
![GRN inference pipeline:](/GRN_inference_pipeline.svg)<br>

### Simulating/generating gene expression data:<br>
The software GeneNetWeaver used to generate the datasets in the current study is available in the GitHub repository, [https://github.com/tschaffter/genenetweaver](https://github.com/tschaffter/genenetweaver)<br>
We used our Jupyter notebook [Generate_replicates_for_network_inference.ipynb](/Generate_replicates_for_network_inference.ipynb) to call GeneNetWeaver to generate multiple replicates of steady-state and time-series gene expression datasets for realistic in silico networks of sizes of 50, and 100 genes containing various experimental conditions (knockouts, knockdowns, multifactorial perturbation, etc.).<br>
We then combine all the steady-state expression data to a single file: {network name}_all.tsv<br>

### GRN inference (including density estimation and MI calculation):<br>
The Jupyter ntoebook [AUPR_calc_for_filling_missing_MI_or_Inference.ipynb](/AUPR_calc_for_filling_missing_MI_or_Inference.ipynb) together with the python modules: Building_MI_matrices.py, Inference_algo_module.py, Precision_Recall_module.py, is responsible to discretize the continous gene expression data to fixed width bins or organize the data according to k-NN. Following by mutual information calculations by one of four estimators ((i) Maximum Likelihood (ML, given by Shannon), (ii) Miller-Madow correction (MM), (iii) Kozachenko-Leonenko (KL), and (iv) Kraskov-Stoögbauer-Grassberger (KSG). Finally, the network structure is infered by one of six inference algorithms (Relevance Networks, RL; Algorithm for the Reconstruction of Accurate Cellular Networks, ARACNE; Context-Likelihood-Relatedness, CLR; Synergy-Augmented CLR, SA-CLR; Luo et al. MI3; CMI2rt, and our Conditional-Mutual-Information-Augmentation, CMIA) 

### GRN performance evaluation and plotting:<br>
To compare the performence and plot it we use the Jupyter notebook [CMIAwKSG_paper_AUPR_figures_and_SI.ipynb](/CMIAwKSG_paper_AUPR_figures_and_SI.ipynb).<br>
Data is organized in a pandas dataframe and plotted using matplotlib.
