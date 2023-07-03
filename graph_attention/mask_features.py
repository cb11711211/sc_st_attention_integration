"""
We try to use the cross attention networks to integrate the spatial mutli-omics data including spatial transcriptomics and spatial proteomics.
The graph cross attention network should be trained by the mask learning strategy.
Here, I would like to randomly mask 15% to 30% of the input features, which the 15% to 30% is randomly selected.
And the masked features will be set to 0. And the proteomics and transcriptomics are indepently selected.
"""
# %% use the mask_input_features function on real data
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

root_dir = "/home/wuxinchao/data/project/spatial-CITE-seq/mid_result"
combine_data = pd.read_csv(f"{root_dir}/B01825A4_rna_prot_concat.csv", index_col=0)
edge_index = pd.read_csv(f"{root_dir}/B01825A4_edge_index.csv", index_col=0)
# %%
mask_rate = 0.3
prot_data = torch.tensor(combine_data.iloc[:, -10:].values, dtype=torch.float32)
rna_data = torch.tensor(combine_data.iloc[:, :-10].values, dtype=torch.float32)
# %%
def data_mask(matrix, mask_rate=0.3):
    # mask the input matrix, and the masked zero will be set to -10000, 
    # while the masked non-zero will be set to 0
    matrix = matrix.clone()
    mask = torch.rand(matrix.shape) < mask_rate
    zero_fill_mtx = np.where(mask, torch.tensor(0.0), matrix)
    min_fill_mtx = np.where(mask, torch.tensor(-10000.0), matrix)
    # zero_mask = np.equal(matrix, zero_fill_mtx)
    zero_mask = np.equal(mask, np.equal(matrix, zero_fill_mtx))
    min_fill_mtx[zero_mask] = 0
    return min_fill_mtx
# %%
prot_data_masked = data_mask(prot_data, mask_rate=0.3)
rna_data_masked = data_mask(rna_data, mask_rate=0.3)
# %%
from GraphCrossAttenNet import GraphCrossAttenNet

# rna_data_masked = torch.tensor(rna_data_masked, dtype=torch.float32)
# prot_data_masked = torch.tensor(prot_data_masked, dtype=torch.float32)
# edge_index_input = torch.tensor(edge_index.values, dtype=torch.long)
# data = (rna_data_masked, prot_data_masked, edge_index_input)
Graph_Cross_Atten_Net = GraphCrossAttenNet(
    prot_feature_dim=prot_data.shape[1],
    rna_feature_dim=rna_data.shape[1],
    num_layers=3,
    num_heads_per_layer=[4, 4, 4],
    num_features_per_layer=[1024, 1024, 1024, 1024],
    add_skip_connection=True,
    bias=True,
    dropout=0.6,
    log_attention_weights=False)

# result = Graph_Cross_Atten_Net(data)
def construct_masked_data(data, mask_rate=0.3):
    """
    We need to construct the different masked data for each epoch in the training process.
    """
    rna_data, prot_data, edge_index = data
    # try to use random seed to make sure the masked data is not the same in different epochs
    np.random.seed(np.random.randint(0, 1000))
    rna_data_masked = data_mask(rna_data, mask_rate=mask_rate)
    prot_data_masked = data_mask(prot_data, mask_rate=mask_rate)
    return (rna_data_masked, prot_data_masked, edge_index)
# %% train the model with the masked data using BCELoss or MSELoss
from torch.optim import Adam
from torch.nn import BCELoss, MSELoss

optimizer = Adam(Graph_Cross_Atten_Net.parameters(), lr=0.001)
# loss_function = BCELoss()
loss_function = MSELoss()

# make sure the data is in the correct format
rna_data = torch.tensor(rna_data, dtype=torch.float32)
prot_data = torch.tensor(prot_data, dtype=torch.float32)
edge_index_input = torch.tensor(edge_index.values, dtype=torch.long)
data = (rna_data, prot_data, edge_index_input)

def train(model, data, optimizer, loss_function, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # gat_input, prot_data, edge_index = data
        gat_input, prot_data, edge_index = construct_masked_data(data, mask_rate=0.3)
        concat_data = torch.cat((gat_input, prot_data), dim=1)
        masked_data = (gat_input, concat_data, edge_index)
        gat_output, cross_attn_output = model(masked_data)
        # loss is the sum of the weighted loss of the two parts
        loss = loss_function(gat_output, gat_input) + loss_function(cross_attn_output, concat_data)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
# %%
train(Graph_Cross_Atten_Net, data, optimizer, loss_function, epochs=100)
# %% try to find which part of the code is the most time consuming
matrix = rna_data.clone()
mask = torch.rand(matrix.shape).lt(mask_rate)
zero_fill_mtx = np.where(mask, torch.tensor(0.0), matrix)
min_fill_mtx = np.where(mask, torch.tensor(-10000.0), matrix)
zero_mask = np.equal(mask, np.equal(matrix, zero_fill_mtx))
min_fill_mtx = np.where(zero_mask, torch.tensor(0.0), min_fill_mtx)
# min_fill_mtx[zero_mask] = 0
# %%
torch.rand(matrix.shape).lt(mask_rate)
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Define the mean and covariance matrix for the distribution
mu = np.array([0, 1])
cov = np.array([[1, 0.5], [0.5, 1]])

# Create a grid of points
x, y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Create a multivariate normal object
rv = multivariate_normal(mu, cov)

# Evaluate the PDF at each point in the grid
pdf = rv.pdf(pos)

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the PDF as a surface plot
ax.plot_surface(x, y, pdf, cmap='viridis')

# Set the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PDF')
ax.set_title('PDF of a 2D Normal Distribution')

# Set the scaling of the 3D axes
# max_range = np.array([x.max()-x.min(), y.max()-y.min(), pdf.max()-pdf.min()]).max()
# X, Y, Z = np.diag([max_range, max_range, max_range])
# ax.auto_scale_xyz([-max_range, max_range], [-max_range, max_range], [-max_range, max_range])

# Show the plot
plt.show()
# %%
# matrix[mask] = 0 if matrix[mask] > 0 else -10000
