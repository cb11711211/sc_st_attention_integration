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


# Research plan
## 1. Dataset using
SNL dataset used in TOTALVI paper. Which is a dataset that contains spleen and lymph node cells. 
10X Genomics PBMC dataset. Which is a dataset that contains peripheral blood mononuclear cells.
## 2. Data preprocessing
### 2.1 transcriptomics data preprocessing
For transcriptomics data, implement the scale normalization and log transformation.  

### 2.2 proteomics data preprocessing
For proteomics data, implement the centering and log transformation.

## 3. Data integration
Using the TOTALVI, WNN, MOFA+ and our own method to integrate the data.

## 4. DE analysis
Implement DE analysis based on the embedding of the integrated data. 

## 5. Imputation analysis

## 6. Clustering analysis

## 7. Spatial CITE-seq Fine-tuning
There are three types of fine-tune strategy:
    1. Addition-based: add the module encoding the spatial information to the original model.
    2. Specification-based: specifically fine-tune the parameters of the pre-trained model and preserve the remaining parameters.
    3. Reparameterization-based: reparameterize the pre-trained model

Apply LoRA (Low rank adptation) to single-cell Multi-omics integration, we could easily fine-tune the pre-trained neural network to the specific downstream task like spatial multi-omics integration, spatially-resolved cell-cell communication, imputation and other. 

With the understandin of the spatial transcriptomics, we could encode the spatial location into a latent space using the attention mechanism, which is an encoder for the spatial coordinates. 

For the spatial encoding of the spatial transcriptomics, we could use the attention mechanism to encode the spatial coordinates into a latent space. And the graph is still constructed based on the expression profile of the spots.