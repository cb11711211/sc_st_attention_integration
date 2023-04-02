# Background
The data set has been proved to be the best thing I ever done in my coding life.  
I have tried to learn pre-request packages for machine learning including pandas, matplotlib, numpy and pytorch. While, I have no idea how to build a package that could help myself to learning or analysis data. So I have decided to build this repository, to share some skills in coding, machine learning or artificial intelligence.  
However, this repo mainly focus on developing a new methods that could integrate single-cell RNA-seq data with spatial transcriptomics data particularly.  
# Integration of spatial transcriptoimcs and single-cell RNA-seq data
For the past few years, CNN has accelerated the application of AI in bioinformatics study particular for single-cell RNA-seq analysis. However, the neural network could not find the global scale   
The multi-view learning in this project could view that the most important direction in this field. How to integrate the high quality single-cell RNA-seq with low-quality spatial transcriptomics data should be one specfic task that should be done both in bioinformatics and machine learning field.  
I have review some articles that perscribe the computational methods that coudl integrate single-cell multi-omics data. For multi-omics data, the features of each omics is different which means the feature extraction should be done in one omics independently. However, for the integration of spatial transcriptomics and single-cell RNA-seq data, the features from both omics both representing the exprssion profile of some votex which is single-cell for scRNA-seq and spot for spatial transcriptomics, respectively. 
Here I will firstly construct a begining notebook storing many types of methods that utilizing deep neural networks especially that applied in attention mechanism. 

# Data preprocessing
There could be some alignment and difference between the gene expression profile and protein expression profile. In this project, we will implement some methods that could denoise the data and then align the multi-omics profiles to the same feature space.

The data preprocessing should consider the situation that the coordinations of the proteomics and transcriptomics profiles are not consistent. So that we need to do some implementation for the profiles such as optimal transport (OT) algorithm.

For this step, we will firstly implement the OT algorithm to align the two omics data. That the 