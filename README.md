# Introduction
Here is the repository that I have built for the project of integrating spatial transcriptomics and single-cell RNA-seq data, especially for spatially-resovled CITE-seq data.

# Integration of spatial transcriptoimcs and single-cell RNA-seq data
For the past few years, CNN has accelerated the application of AI in bioinformatics study particular for single-cell RNA-seq analysis. However, the neural network could not find the global scale.
The multi-view learning in this project could view that the most important direction in this field. How to integrate the high quality single-cell RNA-seq with low-quality spatial transcriptomics data should be one specfic task that should be done both in bioinformatics and machine learning field.  
I have review some articles that perscribe the computational methods that coudl integrate single-cell multi-omics data. For multi-omics data, the features of each omics is different which means the feature extraction should be done in one omics independently. However, for the integration of spatial transcriptomics and single-cell RNA-seq data, the features from both omics both representing the exprssion profile of some votex which is single-cell for scRNA-seq and spot for spatial transcriptomics, respectively. 
Here I will firstly construct a begining notebook storing many types of methods that utilizing deep neural networks especially that applied in attention mechanism. 

# Data preprocessing
There could be some alignment and difference between the gene expression profile and protein expression profile. In this project, we will implement some methods that could denoise the data and then align the multi-omics profiles to the same feature space.

The data preprocessing should consider the situation that the coordinations of the proteomics and transcriptomics profiles are not consistent. So that we need to do some implementation for the profiles such as optimal transport (OT) algorithm.


## 1. Data available
SNL dataset used in TOTALVI paper. Which is a dataset that contains spleen and lymph node cells. 
10X Genomics PBMC dataset. Which is a dataset that contains peripheral blood mononuclear cells.
Spatial CITE-seq mouse spleen dataset, exported from original published data. 
## 2. Data preprocessing
*transcriptomics data preprocessing*
For transcriptomics data, implement the scale normalization and log transformation.  

*proteomics data preprocessing*
For proteomics data, implement the centering and log transformation.

## 3. Benchmark data integration methods
Using the TOTALVI, WNN, MOFA+ and our own method to integrate the data. 

### Metrics
The clustering analysis is the main part to evaluate the performance of the GCAT model.
The metrics used to account for the clustering evaluation including: Homogeneity score, Sillueitte score, and mutual information. 

The V-measure is the harmonic mean between homogeneity and completeness
`$$v = (1 + beta) * homogeneity * completeness / (beta * hemogeneity + completeness)$$`

#### Homogeneity score:
A clustering result satisfies homogeneity values of the labels: a permutation of the class or cluster label values won't change the score value in anyway

#### Completeness: 
Symmetrical to homogeneity, a clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.

#### Silhouette score:
Compute the mean Silhouette Coefficient of all samples.
The mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample.
`(b - a) / max(a, b)`

#### Adjusted Rand index (ARI)
Measures the similarity of the two assignments, ignoring permutations and with chance normalization.
`ARI = (RI - E[RI]) / (max(RI) - E[RI])`

#### Normalized Mutual Information (NMI) and Adjusted Mutual Information (AMI)
Measures the agreement of the two assignments, ignoring permutations.
`NMI(U, V) = MI(U, V) / mean(H(U), H(V))`
`AMI = (MI - E[MI]) / (mean(H(U), H(V)) - E[MI] )`

#### Calinski-Harabasz index (Variance Ratio Criterion)
The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)

#### Davies-Bouldin Index
This index signifies the average 'similarity' between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.

#### Spatial Coherence
The spatial coherence score evaluates to what extent that the spatial region or labels assigned to a certain type of spots is coherent, continous or homogeneous in spatial neighborhood. This metrics could help to evaluate the performance of spatial multi-omics integration methods. 

#### Moran's I

```python
import numpy as np

def calculate_adjacency_matrix(labels_matrix, adjacency_matrix):
    label_agreement = np.zeros_like(adjacency_matrix)
    for i in range(len(labels_matrix)):
        for j in range(len(labels_matrix)):
            if adjacency_matrix[i, j] and labels_matrix[i] == labels_matrix[j]:
                label_agreement[i, j] = 1
    return label_agreement

def compute_spatial_coherence(label_agreement, num_shuffles=100):
    observed_score = np.sum(label_agreement, axis=1)
    shuffled_scores = []
    
    for _ in range(num_shuffles):
        np.random.shuffle(labels_matrix)  # Shuffles the labels matrix in-place
        shuffled_agreement = calculate_adjacency_matrix(labels_matrix, adjacency_matrix)
        shuffled_scores.append(np.sum(shuffled_agreement, axis=1))
    
    expected_min = np.min(shuffled_scores, axis=0)
    expected_max = np.max(shuffled_scores, axis=0)  # Assuming max is calculated similarly or predefined
    
    spatial_coherence_scores = (observed_score - expected_min) / (expected_max - expected_min)
    return spatial_coherence_scores

def morans_i(labels_matrix, spatial_coherence, adjacency_matrix):
    w = adjacency_matrix  # weight matrix using adjacency
    mean_labels = np.mean(labels_matrix)
    S0 = np.sum(w)
    
    num = n = len(labels_matrix)
    num = np.sum(w * (labels_matrix[:, None] - mean_labels) * (labels_matrix - mean_labels)[:, None])
    denom = np.sum((labels_matrix - mean_labels) ** 2)
    
    morans_I = (num / S0) / denom
    return morans_I

```

### Explanation of Moran's I Computation

**Moran's I** is a statistical measure used to assess the degree of spatial autocorrelation in data measured across geographic or spatial areas. This statistic ranges from -1 (indicating perfect dispersion) to +1 (indicating perfect correlation), with a value close to zero suggesting a random spatial pattern.

#### **Steps in the Moran's I Calculation**

1. **Weight Matrix (W):**
   - The weight matrix, represented as $W$ in the computation, defines the spatial relationship or adjacency between data points. In this case, the adjacency matrix is used directly as the weight matrix. Here, each entry $w_{ij}$ is set to 1 if spots $i$ and $j$ are neighbors (i.e., directly adjacent) and 0 otherwise.

2. **Key Elements of the Formula:**
   - **N**: The total number of spatial units indexed by $i$ and $j$.
   - **\(x_i\) and \(x_j\)**: The attribute values for the spatial units.
   - **\(\overline{x}\)**: The average of all attribute values.
   - **\(w_{ij}\)**: The spatial weight between units $i$ and $j$.
   - **\(S_0\)**: The sum of all spatial weights, calculated as \(\sum_{i=1}^n \sum_{j=1}^n w_{ij}\).

3. **Moran's I Formula:**
   $$I = \frac{n}{S_0} \times \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij} (x_i - \overline{x})(x_j - \overline{x})}{\sum_{i=1}^n (x_i - \overline{x})^2}$$

   - **Numerator**: Calculates the product of deviations from the mean, weighted by the adjacency matrix. This term assesses how attribute values deviate from the mean in a spatially dependent manner.
   - **Denominator**: Sums up the squared deviations from the mean, normalizing the statistic.



#### **Interpretation of Results:**

- **Positive Moran's I**: Suggests that similar values are clustered together. For example, high values are near high values, and low values are near low values.
- **Negative Moran's I**: Indicates that high values are typically surrounded by low values, suggesting a dispersed or checkerboard pattern.
- **Moran's I Near Zero**: Implies a random spatial distribution with no significant autocorrelation.

This measure is invaluable in geographic and environmental data analysis, helping to understand spatial patterns in phenomena ranging from disease distribution in 
epidemiology
 to economic activities in regional studies. In the context of molecular profiles in biological samples, Moran's I serves as a validation tool for assessing spatial coherence, ensuring that the spatial distribution of molecular profiles is not random but exhibits a significant pattern of spatial autocorrelation.

## 4. Spatial CITE-seq data analysis
There are three types of fine-tune strategy:
    1. Addition-based: add the module encoding the spatial information to the original model.
    2. Specification-based: specifically fine-tune the parameters of the pre-trained model and preserve the remaining parameters.
    3. Reparameterization-based: reparameterize the pre-trained model

Apply LoRA (Low rank adptation) to single-cell Multi-omics integration, we could easily fine-tune the pre-trained neural network to the specific downstream task like spatial multi-omics integration, spatially-resolved cell-cell communication, imputation and other. 

With the understandin of the spatial transcriptomics, we could encode the spatial location into a latent space using the attention mechanism, which is an encoder for the spatial coordinates. 

For the spatial encoding of the spatial transcriptomics, we could use the attention mechanism to encode the spatial coordinates into a latent space. And the graph is still constructed based on the expression profile of the spots.