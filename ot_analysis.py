# %% import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.sparse import coo_matrix
import stereo as st
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
