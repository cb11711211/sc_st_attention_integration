# %% import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import stereo as st
from scipy.sparse import coo_matrix

# %%
gene_map = {
    "CD11c": "Itgax",
    "CD27" : "Cd27",
    "CD8a" : "Cd8a",
    "CD68" : "Cd68",
    "2" : "Thy1",
    "CD3" : "Cd3g",
    "CD44" : "Cd44",
    "CD5" : "Cd5",
    "CD4" : "Cd4",
    "CD29" : "Itgb1"
}

prot_data_path = "/home/wuxinchao/data/st_cite_data/B01825A4_protein_filter.csv"
rna_data_path = "/home/wuxinchao/data/st_cite_data/B01825A4_rna_raw.csv"

prot_data = pd.read_csv(prot_data_path)
rna_data = pd.read_csv(rna_data_path)

prot_data.columns = prot_data.columns.str.split("_").str[-1]
# rename the columns of the prot_data based on the gene_map
prot_data.rename(columns=gene_map, inplace=True)

# using the data in prot_data replace the columns in rna_data
rna_data.index = rna_data["Unnamed: 0"]
rna_data.drop("Unnamed: 0", axis=1, inplace=True)
prot_data.index = prot_data["Unnamed: 0"]
prot_data.drop("Unnamed: 0", axis=1, inplace=True)

# rna_coords
rna_coords = np.asarray(rna_data.index.str.split("_", expand=True))
rna_coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in rna_coords])

# prot_coords
prot_coords = np.asarray(prot_data.index.str.split("_", expand=True))
prot_coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in prot_coords])

# alignment the two matrix
def align_prot_rna_mtx(df, gene_name):
    data = df[gene_name]
    coords = np.asarray(data.index.str.split("_", expand=True))
    coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coords])
    # delete the data with coordination[1] less than 13
    x_mask_ind = coordination[:, 0] >= 13
    y_mask_ind = coordination[:, 1] <= 78
    mask_ind = x_mask_ind & y_mask_ind
    data = data[mask_ind]
    coordination = coordination[mask_ind]
    row_ind = coordination[:, 0] - 13
    col_ind = coordination[:, 1]
    data = data.values
    coo = coo_matrix((data, (row_ind, col_ind)), shape=np.max(coordination, axis=0)+1)
    csr = coo.tocsr()
    plt.scatter(row_ind, col_ind, c=data, cmap="inferno", s=2)
    plt.title(f"{gene_name} spatial expression")
    plt.colorbar()
    plt.show()
    return csr

# get the spatial expression of the gene_name
def get_spa_gene_exp(df, gene_name):
    df = df[gene_name]
    coords = np.asarray(df.index.str.split("_", expand=True))
    coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coords])
    row_ind = coordination[:,0]
    col_ind = coordination[:,1]
    data = df.values
    coo = coo_matrix((data, (row_ind, col_ind)), shape=np.max(coordination, axis=0)+1)
    csr = coo.tocsr()
    plt.scatter(row_ind, col_ind, c=data, cmap="inferno", s=2)
    plt.title(f"{gene_name} spatial expression")
    plt.colorbar()
    plt.show()
    return csr

# %%
h5ad_filepath = "/home/wuxinchao/data/project/spatial-CITE-seq/mid_result/B01825A4_rna_prot_raw_rep.h5ad"
adata = sc.read_h5ad(h5ad_filepath)
# %%
sc.pl.clustermap(adata)
# %%
adata.var
# %%
fig, axes = plt.subplots(2,2 ,figsize=(10, 10), layout="tight")
axes = sc.pl.spatial(adata, spot_size=0.5, color=["Cd8a", "Cd4", "log1p_total_counts", "log1p_n_genes_by_counts"], show=False)
axes[0].set_title("CD8a")
axes[1].set_title("CD4")
axes[2].set_title("log1p_total_counts")
axes[3].set_title("log1p_n_genes_by_counts")
# set the layout of the figure
fig.tight_layout()

# %%
adata.obs

# %%
adata
# %%
adata.X
# %%
print(adata.raw)
import matplotlib.pyplot as plt
# %%
# Here we would like to discuss some of the basic assumptions for the st-CITE-seq data
# The first assumption is that the protein count of one spot is mainly determinded by the RNA count
# of the spot, and the second assumption is that the RNA count in neighbor spot also contribute to the
# protein count through transfer function.
# In summary, we could make the assumption that the protein count of one spot is a linear
# combination of several different variables
# Protein count of one spot = alpha * RNA count of the spot + beta * RNA count of neighbor spots + technical noise
# In detail, we believe that protein count ~ NB distribution across the slice, and 
# the RNA count ~ ZINB distribution, but these model should be check through BIC that whether they are the best model
# if the model is not the best model, we could choose the model with the lowest BIC score
# %% model the protein count of one spot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import scanpy as sc
import torch
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide import (AutoDiagonalNormal, AutoMultivariateNormal,
                                  init_to_mean)
# %% read the protein and rna data
rna_coordination, prot_coordination
# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(rna_coordination[:, 0], rna_coordination[:, 1], s=2)
axes[0].set_title("rna_coordination")
axes[1].scatter(prot_coordination[:, 0], prot_coordination[:, 1], s=2)
axes[1].set_title("prot_coordination")

# %%
prot_data.shape, rna_data.shape
# %%
prot_data
# %% transformation of prot and rna raw data
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.optimize import minimize

def log_likelihood_poisson(y, X, coef):
    """
        y: np.array, containing the observed counts for each cell in the scRNA-seq data.
        X: np.array, containing the feature matrix for the data
    """
    eta = np.dot(X, coef)
    mu = np.exp(eta)
    loglik = np.sum(y * np.log(mu) - mu - gammaln(y+1))
    return loglik

def log_likelihood_negbin(y, X, coef):
    eta = np.dot(X, coef)
    alpha = np.exp(eta)
    mu = alpha / (alpha+1) * y.mean(axis=0)
    size = mu / (alpha + 1e-8)
    prob = size / (size + y)
    loglik = np.sum(gammaln(size + y) - gammaln(y+1) - gammaln(size) + size * np.log(prob) + y * np.log(1 - prob))
    return loglik

def log_likelihood_zip(y, X, coef):
    eta = np.dot(X, coef[:-1])
    pi = np.exp(coef[-1]) / (1 + np.exp(coef[-1]))
    mu = np.exp(eta)
    loglik = np.sum(y==0) * np.log(pi + (1-pi)*np.exp(-mu)) + np.sum(y>0) * np.log(1-pi) + np.sum((y>0) * (y * np.log(mu) - gammaln(y+1) - mu))
    return loglik

def log_likelihood_zinb(y, X, coef):
    eta = np.dot(X, coef[:-1])
    alpha = np.exp(coef[-1])
    mu = np.exp(eta)
    size = mu / (alpha + mu)
    prob = alpha / (alpha + mu)
    loglik = np.sum(gammaln(size + y) - gammaln(y+1) - gammaln(size) + size * np.log(prob) + y * np.log(1 - prob + 1e-8))
    return loglik
# %%
def fit_glm(data, covariates):
    # fit poisson model
    poisson_coef = minimize(log_likelihood_poisson, x0=np.zeros(covariates.shape[1]), args=(covariates, data)).x
    poisson_ll = log_likelihood_poisson(data, covariates, poisson_coef)
    poisson_bic = -2 * poisson_ll + np.log(data.shape[0]) * covariates.shape[1]

    # fit negative binomial model
    nb_coef = minimize(log_likelihood_negbin, x0=np.zeros(covariates.shape[1]), args=(covariates, data)).x
    nb_ll = log_likelihood_negbin(data, covariates, nb_coef)
    nb_bic = -2 * poisson_ll + np.log(data.shape[0]) * covariates.shape[1]

    # fit zero-inflated Poisson model
    zip_coef = minimize(log_likelihood_zip, x0=np.zeros(covariates.shape[1] + 1), args=(covariates, np.hstack([data == 0, data]))).x
    zip_ll = log_likelihood_zip(np.hstack([data == 0, data]), covariates, zip_coef)
    zip_bic = -2 * zip_ll + np.log(data.shape[0]) * (covariates.shape[1] + 1)

    # fit zero-inflated negative binomial model
    zinb_coef = minimize(log_likelihood_zinb, x0=np.zeros(covariates.shape[1] + 1), args=(covariates, np.hstack([data == 0, data]))).x
    zinb_ll = log_likelihood_zinb(np.hstack([data == 0, data]), covariates, zinb_coef)
    zinb_bic = -2 * zinb_ll + np.log(data.shape[0]) * (covariates.shape[1] + 1)

    # choose the best fit model based on the BIC score
    bic_scores = [poisson_bic, nb_bic, zip_bic, zinb_bic]
    best_model_idx = np.argmin(bic_scores)
    if best_model_idx == 0:
        return 'poisson', poisson_coef
    elif best_model_idx == 1:
        return 'nb', nb_coef
    elif best_model_idx == 2:
        return 'zip', zip_coef
    else:
        return 'zinb', zinb_coef

def sctransform(data, size_factors, covariates):
    data = csr_matrix(data)
    size_factors = np.array(size_factors)
    covariates = np.array(covariates)

    # normalize the data using size factors
    data_norm = data.multiply(1 / size_factors[:, None])
    data_log = np.log1p(data_norm)

    # fit the GLM to determine the appropriate normalization factors
    model, coef = fit_glm(data_log, covariates)
    if model == 'poisson':
        norm_factors = np.exp(np.dot(covariates, coef))
    elif model == 'nb':
        alpha = np.exp(coef[-1])
        mu = np.exp(np.dot(covariates, coef[:-1]))
        norm_factors = (alpha + mu) / alpha
    elif model == 'zip':
        pi = np.exp(coef[-1]) / (1 + np.exp(coef[-1]))
        mu = np.exp(np.dot(covariates, coef[:-1]))
        norm_factors = (1 - pi) * np.exp(-mu)
    else:
        alpha = np.exp(coef[-1])
        mu = np.exp(np.dot(covariates, coef[:-1]))
        size = mu / (1 - mu) * alpha
        prob = size / (size + mu)
        norm_factors = (1 - prob) * np.exp(-mu)

    # normalize the data using the normalization factors
    data_norm = data.multiply(1 / norm_factors[:, None])
    return data_norm, norm_factors
# %% convert the prot_data to a sparse matrix
from scipy.sparse import csr_matrix, vstack

spatial_mtx = None
for gene_name in prot_data.columns:
    if spatial_mtx is None:
        spatial_mtx = align_prot_rna_mtx(prot_data, gene_name)
        # spatial_mtx = csr_matrix(spatial_mtx)
    else:
        spatial_mtx_gene = align_prot_rna_mtx(prot_data, gene_name)
        # spatial_mtx = vstack([spatial_mtx, spatial_mtx_gene])
        spatial_mtx = np.hstack([spatial_mtx, spatial_mtx_gene])
        # spatial_mtx = np.concatenate([spatial_mtx, spatial_mtx_gene], axis=2)
        # spatial_mtx = vstack([spatial_mtx, spatial_mtx_gene])
# spatial_mtx = csr_matrix(spatial_mtx)
print(spatial_mtx.shape)
# %% convert the spatial_mtx to a sparse matrix
get_spa_gene_exp(rna_data, "Cd3g")

# %%
rna_data["total_counts"] = rna_data.sum(axis=1)
get_spa_gene_exp(rna_data, "total_counts")

# %% reindex the prot_data with the aligned coord
def align_coord(df):
    coords = np.asarray(df.index.str.split("_", expand=True))
    coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coords])
    # delete the data with coordination[1] less than 13
    x_mask_ind = coordination[:, 0] >= 13
    y_mask_ind = coordination[:, 1] <= 78
    mask_ind = x_mask_ind & y_mask_ind
    df = df[mask_ind]
    coordination = coordination[mask_ind]
    row_ind = coordination[:, 0] - 13
    col_ind = coordination[:, 1]
    ind = [str(row) + "_" + str(col) for row, col in zip(row_ind, col_ind)]
    df.index = ind
    return df
# %%
prot_data = align_coord(prot_data)
prot_data.columns = prot_data.columns + "_prot"
combine_data = pd.concat([rna_data, prot_data], axis=1)
combine_data.fillna(0, inplace=True)
# %%
get_spa_gene_exp(combine_data, "Cd8a")
# %% construct the adjacency matrix 
"""
For each spot, the spot in neighbor is linked, which means if the x or y coord is close
to the spot, they are neighboring spot. And the neighboring spot should 
"""
def get_adj_mtx(df):
    """
    define the x and y are close to 1 unit, the spots are neighbor
    """
    coords = np.asarray(df.index.str.split("_", expand=True))
    coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coords])
    adj_mtx = np.zeros((coordination.shape[0], coordination.shape[0]))
    for i in range(coordination.shape[0]):
        for j in range(coordination.shape[0]):
            # if the x and y are close to 1 unit, the spots are neighbor
            if abs(coordination[i, 0] - coordination[j, 0]) <= 1 and abs(coordination[i, 1] - coordination[j, 1]) <= 1:
                adj_mtx[i, j] = 1
    # set the diagonal of the adj_mtx to 0
    np.fill_diagonal(adj_mtx, 0)
    return adj_mtx

adj_mtx = get_adj_mtx(combine_data)
# %%
import statsmodels.api as sm
from patsy import dmatrices

df = sm.datasets.get_rdataset("Guerry", "HistData").data
# %%
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df[-5:]
# %%
df = df.dropna()
# %%
y, X = dmatrices("Lottery ~ Literacy + Wealth + Region", data=df, return_type="dataframe")
# %%
y[-5:], X[-5:]
# %%
model = sm.OLS(y, X)
res = model.fit()
print(res.summary())
# %%
res.params, res.rsquared
# %%
dir(res)
# %%
sm.stats.linear_rainbow(res)
# %%
sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],
                             data=df, obs_labels=False) 
# %%
gene = "Cd8a"
def fit_glm(data, gene):
    prot_name = gene + "_prot"
    rna_count = data[gene]
    prot_count = data[prot_name]
    # fit poisson model
# %% find the row with all zero
def find_all_zero_row(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df
# %%
combine_data = find_all_zero_row(combine_data)

# %%
import seaborn as sns

sns.heatmap(combine_data.corr())
# %%
import scipy.stats as stats
import statsmodels.api as sm
from patsy import dmatrices

distributions = [
    ("Poisson", stats.poisson),
    ("Negative Binomial", stats.nbinom),
    ("Zero-Inflated Poisson", stats.zipoisson),
    ("Zero-Inflated Negative Binomial", stats.zinegbinom),
]

def fit_glm(data, size_factor, adj_mtx, distributions):
    name, dist = distributions
    # fit poisson model: prot_count ~ alpha * poisson(rna_count) + beta * poisson(rna_count_neighbor) + technical noise
    prot_count = data[:, 0]
    rna_count = data[:, 1]
    # get the rna_count_neighbor
    rna_count_neighbor = adj_mtx.dot(rna_count)
    # fit the poisson model
    glm_model = sm.GLM(prot_count, rna_count, rna_count_neighbor, family=sm.families.Poisson())
    glm_results = glm_model.fit()
    print(glm_results.summary())

# %%
