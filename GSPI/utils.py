import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import muon as mu
from sklearn.metrics import mean_squared_error
from torch import neg_

from data.project.scBERT_test.scBERT.test import POS_EMBED_USING

def rmse(pred, target):
    return np.sqrt(mean_squared_error(pred, target))


def get_pos_neg_pairs(adata, neighbors=10, resolution=1.0, method='leiden', sim_thres=0.5):
    """
    Define positive pairs and negative pairs for each cell in adata.
    
    The positive pairs are defined as the pairs of cells that similar in expression profile.
    The negative pairs are defined as the pairs of cells that dissimilar in expression profile.
    The similarity is calculated based on the community finding algorithm (leiden, louvain, etc.).
    """
    adata_temp = adata.copy()
    adata_temp.X = adata_temp.X.toarray()
    adata_temp.X = adata_temp.X.astype('float32')
    adata_temp.obs['cell_id'] = adata_temp.obs.index
    adata_temp.obs['cell_id'] = adata_temp.obs['cell_id'].astype('int')
    adata_temp.obs['cell_id'] = adata_temp.obs['cell_id'].astype('str')

    # get the similarity matrix
    sc.pp.neighbors(adata_temp, n_neighbors=neighbors, use_rep='X')
    if method == 'leiden':
        sc.tl.leiden(adata_temp, resolution=resolution)
    elif method == 'louvain':
        sc.tl.louvain(adata_temp, resolution=resolution)
    else:
        # randomly assign the connectivity matrix
        adata_temp.obsp['connectivities'] = sp.random(adata_temp.shape[0], adata_temp.shape[0], density=0.1, format='csr')
        adata_temp.obsp['connectivities'] = adata_temp.obsp['connectivities'] + adata_temp.obsp['connectivities'].T
        adata_temp.obsp['connectivities'] = adata_temp.obsp['connectivities'].toarray()
    sim_mat = adata_temp.obsp['connectivities']

    # get the positive pairs
    pairs_mtx = np.zeros((adata_temp.shape[0], adata_temp.shape[0]))
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if i < j:
                if sim_mat[i, j] > sim_thres:
                    pairs_mtx[i, j] = 1
                    pairs_mtx[j, i] = 1
    pos_pairs = pairs_mtx
    # get the negative pairs by randomly assign the pairs
    neg_pairs = np.ones((adata_temp.shape[0], adata_temp.shape[0])) - pairs_mtx
    return pos_pairs, neg_pairs


def constrative_loss(adata, pos_pairs, neg_pairs, alpha=1.0, beta=1.0):
    """
    Calculate the constrative loss for each cell in adata.
    The constrative loss is defined as neg distance - pos distance.
    """
    adata_temp = adata.copy()
    adata_temp.X = adata_temp.X.toarray()
    adata_temp.X = adata_temp.X.astype('float32')
    adata_temp.obs['cell_id'] = adata_temp.obs.index
    adata_temp.obs['cell_id'] = adata_temp.obs['cell_id'].astype('int')
    adata_temp.obs['cell_id'] = adata_temp.obs['cell_id'].astype('str')

    # get the similarity matrix, using optimal transport distance
    from scipy.stats import wasserstein_distance
    
    pos_dis = np.zeros((adata_temp.shape[0], adata_temp.shape[0]))
    neg_dis = np.zeros((adata_temp.shape[0], adata_temp.shape[0]))
    # the positive pairs distance is defined as the wasserstein distance between any pairs of cells
    for i in range(pos_pairs.shape[0]):
        for j in range(pos_pairs.shape[1]):
            if i < j:
                if pos_pairs[i, j] == 1:
                    pos_dis[i, j] = wasserstein_distance(adata_temp.X[i, :], adata_temp.X[j, :])
                    pos_dis[j, i] = pos_dis[i, j]
    # the negative pairs distance is defined as the wasserstein distance between any pairs of cells
    for i in range(neg_pairs.shape[0]):
        for j in range(neg_pairs.shape[1]):
            if i < j:
                if neg_pairs[i, j] == 1:
                    neg_dis[i, j] = wasserstein_distance(adata_temp.X[i, :], adata_temp.X[j, :])
                    neg_dis[j, i] = neg_dis[i, j]
    # calculate the constrative loss
    loss = np.mean(neg_dis) - np.mean(pos_dis)
    return loss