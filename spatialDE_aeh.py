import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import coo_matrix
import SpatialDE
import NaiveDE


fuzed_data_path = "/home/wuxinchao/data/st_cite_data/prot_rep_rna_in_raw.csv"
rna_data = pd.read_csv(fuzed_data_path, index_col=0, header=0)
# filling the nan with 0
rna_data = rna_data.fillna(0)
# create anndata through the rna_data
adata = sc.AnnData(rna_data)

coordination = np.asarray([coord.split("_") for coord in adata.obs.index])
coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coordination])
adata.obsm["spatial"] = coordination

counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
coord = pd.DataFrame(adata.obsm['spatial'], columns=["x_coord", "y_coord"], index=adata.obs_names)
results = SpatialDE.run(coord, counts)

sample_info = pd.concat([pd.DataFrame(adata.obsm['spatial'], index=adata.obs.index, columns=["x","y"]), 
                            adata.obs['total_counts']], axis=1)
counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T

sign_results = results.query('qval<0.05')

histology_results, patterns = SpatialDE.aeh.spatial_patterns(
    adata.X, resid_expr, sign_results, C=5, l=12.5, verbosity=1, maxiter=10)

# save the histology_results and patterns
result_path = "/DATA/User/wuxinchao/project/spatial-CITE-seq/mid_result"
histology_results.to_csv(result_path + "/prot_rna_raw_rep_aeh_histology_results.csv")
patterns.to_csv(result_path + "/prot_rna_raw_rep_aeh_patterns.csv")