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

def move_mtx(df, depth):
    """
    move the coordinate of the (x,y)
    """
    x_lim = df['X'].max()
    y_lim = df['Y'].max()
    result = [(i, j) for i in range(-depth, 
                    depth+1) for j in range(-depth, 
                    depth+1) if abs(i) + abs(j) <= depth and abs(i) + abs(j) != 0]
    for i in result:
        df_tmp = df.copy()
        df_tmp['X'] = df_tmp['X'] + i[0]
        df_tmp['Y'] = df_tmp['Y'] + i[1]
        df = pd.concat([df, df_tmp], axis=0)

    # remove the coordinate that is out of the boundary
    df = df[(df['X'] >= 0) & (df['X'] <= x_lim)]
    df = df[(df['Y'] >= 0) & (df['Y'] <= y_lim)]
    return df

def spatial_depth_rank(adata, depth=4, source_type='A', target_type="B", threshold=0.5):
    """
    Getting the spatial depth for the target type and source type
    depth: the depth of the target type, formula: one bin spot equals to 50 units of distance
    """
    # get the all spots belong to the source type
    source_spots = adata.obs[adata.obs['cell_type'] == source_type].index
    source_spots = adata[source_spots, :].obs[source_type] > threshold
    adata_temp = adata[target_spots, :]
    # get the all spots belong to the target type within depth
    target_spots = adata.obs[adata.obs['cell_type'] == target_type].index
    target_spots = adata[target_spots, :].obs[target_type] > threshold
    # get the spatial coordinates for the source spots, and find the nearest spots for source spots
    spatial_coord = adata.obs[['X', 'Y']].loc[source_spots, :]
    # get the neighbor spot
    covered_spots = move_mtx(spatial_coord, depth)
    # for all of the target spots, located in the covered spots, get the potential of the target type
    target_spots_valued = adata[target_spots, :].obs[["X", "Y"]] in covered_spots
    # get the potential of the target type
    target_valued = adata[target_spots, :].obs[target_type]
    # get the mean potential of the target type
    target_valued_mean = target_valued.mean()
    return target_valued_mean
        
def permute_node(adj_mtx, preserve_neighbors=0.5):
    """
    Permute the node in the graph, and return the permuted graph
    Preserve a portion of neighbors for each node
    """
    # get the edge index
    edge_index = np.argwhere(adj_mtx > 0)
    # for each node, randomly select 50% of the neighbors, which means 50% of the edges are not changed
    for i in range(adj_mtx.shape[0]):
        # get neighbors that exist in the graph
        neighbors = edge_index[edge_index[:, 0] == i, 1]
        if neighbors is None:
            continue
        # get the number of neighbors that need to be preserved
        num_preserve_neighbors = int(len(neighbors) * preserve_neighbors)
        if num_preserve_neighbors == int(len(neighbors)):
            continue
        # assign which neighbors need to be preserved
        neighbors_preserved = np.random.choice(neighbors, num_preserve_neighbors, replace=False)
        # get the neighbors that need to be permuted
        neighbors_permuted = np.setdiff1d(neighbors, neighbors_preserved)
        # delete the edges that need to be permuted
        adj_mtx[i, neighbors_permuted] = 0
        # randomly assign new edges for the neighbors
        nodes_non_neighbors = np.setdiff1d(np.arange(adj_mtx.shape[0]), neighbors)
        neighbors_assign = np.random.choice(nodes_non_neighbors, len(neighbors_permuted), replace=False)
        adj_mtx[i, neighbors_assign] = 1
    return adj_mtx

def get_permuted_graph(adata, neighbors=10, sim_thres=0.5):
    """
    adata: the anndata object with the spatial information or the expression-based graph
    neighbors: the number of neighbors for the graph
    sim_thres: the threshold for the similarity between two cells

    Calculate process:
    1. build the graph based on the expression profile (For the scRNA-seq data)
    2. get the similarity matrix for the graph
    3. permute the node in the graph with restriction that a portion of the nodes' neighbors are not changed
    """
    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, n_neighbors=neighbors, use_rep='X')
    # get the similarity matrix
    sim_mat = adata_tmp.obsp['connectivities']
    # permute the node in the graph
    permuted_graph = permute_node(sim_mat) # get the permuted graph adjacency matrix
    return permuted_graph


def latent_mixing_metric():
    """
    Calculate the latent mixing metric for the integration results
    """
    pass


def measurement_mixing_metric():
    """
    Calculate the measurement mixing metric for the integration results
    """