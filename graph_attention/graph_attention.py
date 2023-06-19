# given the gene expression matrix and protein expression matrix, we could generate the adj_mtx
# the adj_mtx is the neighbor matrix of the spots
# we could use knn to generate the adj_mtx, e.g. k=3 indicates that each spot has 3 neighbor spots
# And the distance between the spots could be calculated by the euclidean distance in addition to the similarity of the gene expression profile
# The similarity of the gene expression profile could be calculated by the cosine similarity
# test the GAT model with the toy data
# %%
import torch
from torch import nn
from GAT import GAT
import numpy as np
import pandas as pd

toy_data = np.random.randint(0, 100, size=(100, 100))
toy_data = pd.DataFrame(toy_data)
toy_data.columns = [f"gene_{i}" for i in range(100)]
toy_data.index = [f"spot_{i}" for i in range(100)]

adj_mtx = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        # if the x and y are close to 1 unit, the spots are neighbor
        if abs(i - j) <= 1:
            adj_mtx[i, j] = 1
# set the diagonal of the adj_mtx to 0
np.fill_diagonal(adj_mtx, 0)

GAT_model = GAT(num_layers=3, 
                num_heads_per_layer=[4, 4, 6],
                num_features_per_layer=[100, 100, 100, 100],
                add_skip_connection=True,
                bias=True,
                dropout=0.6,
                log_attention_weights=False)

toy_data = torch.tensor(toy_data.values, dtype=torch.float32)
adj_mtx = torch.tensor(adj_mtx, dtype=torch.float32)

edge_index = adj_mtx.nonzero(as_tuple=False).t()
data = (toy_data, edge_index)
GAT_model(data)
# %%
GAT_model
# %%
from GraphCrossAttenNet import GraphCrossAttenNet
import torch
from torch import nn
from GAT import GAT
import numpy as np
import pandas as pd

toy_rna_data = np.random.randint(0, 100, size=(100, 100))
toy_prot_data = np.random.randint(0, 100, size=(100, 10))
adj_mtx = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        # if the x and y are close to 1 unit, the spots are neighbor
        if abs(i - j) <= 1:
            adj_mtx[i, j] = 1
# set the diagonal of the adj_mtx to 0
np.fill_diagonal(adj_mtx, 0)
# %%
toy_rna_data = torch.tensor(toy_rna_data, dtype=torch.float32)
toy_prot_data = torch.tensor(toy_prot_data, dtype=torch.float32)
adj_mtx = torch.tensor(adj_mtx, dtype=torch.float32)
edge_index = adj_mtx.nonzero(as_tuple=False).t()
# %%
data = (toy_rna_data, toy_prot_data, edge_index)
Graph_Cross_Atten_Net = GraphCrossAttenNet(
    prot_feature_dim=10,
    rna_feature_dim=100,
    num_layers=3,
    num_heads_per_layer=[4, 4, 6],
    num_features_per_layer=[100, 100, 100, 100],
    add_skip_connection=True,
    bias=True,
    dropout=0.6,
    log_attention_weights=False)
Graph_Cross_Atten_Net(data)
# %%
Graph_Cross_Atten_Net(data)[0].shape, Graph_Cross_Atten_Net(data)[1].shape
# %%
