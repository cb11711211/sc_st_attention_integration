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
    zero_mask = mask & np.equal(matrix, zero_fill_mtx)
    min_fill_mtx[zero_mask] = 0
    return matrix
# %%
prot_data_masked = data_mask(prot_data, mask_rate=0.3)
rna_data_masked = data_mask(rna_data, mask_rate=0.3)
# %%

# %%

# %%



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
