import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import coo_matrix
import SpatialDE
import NaiveDE


h5ad_filepath = "/home/wuxinchao/data/project/spatial-CITE-seq/mid_result/B01825A4_rna_prot_raw_rep.h5ad"
result_path = "/DATA/User/wuxinchao/project/spatial-CITE-seq/mid_result"
adata = sc.read_h5ad(h5ad_filepath)

coordination = np.asarray([coord.split("_") for coord in adata.obs.index])
coordination = np.asarray([[int(coord[0]), int(coord[1])] for coord in coordination])
adata.obsm["spatial"] = coordination
sample_info = pd.concat([pd.DataFrame(adata.obsm['spatial'], index=adata.obs.index, columns=["x","y"]), 
                            adata.obs['total_counts']], axis=1)
counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
coord = pd.DataFrame(adata.obsm['spatial'], columns=["x_coord", "y_coord"], index=adata.obs_names)
results = SpatialDE.run(coord, counts)
# save the results
results.to_csv(result_path + "/prot_rna_raw_rep_sde_results.csv")

# aeh
counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(sample_info, norm_expr.T, 'np.log(total_counts)').T

sign_results = results.query('qval<0.05')

histology_results, patterns = SpatialDE.aeh.spatial_patterns(
    adata.X, resid_expr, sign_results, C=5, l=12.5, verbosity=1, maxiter=40)

# save the histology_results and patterns
histology_results.to_csv(result_path + "/prot_rna_raw_rep_aeh_histology_results.csv")
patterns.to_csv(result_path + "/prot_rna_raw_rep_aeh_patterns.csv")