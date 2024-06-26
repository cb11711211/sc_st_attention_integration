# %% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

# %% load data
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


prot_data = align_coord(prot_data)
prot_data.columns = prot_data.columns + "_prot"
combine_data = pd.concat([rna_data, prot_data], axis=1)
combine_data.fillna(0, inplace=True)
adj_mtx = get_adj_mtx(combine_data)

# %% for each gene, generate 

gene = "Cd3g"
prot_name = gene + "_prot"
# combine_data[gene], combine_data[prot_name]
prot_count_ni = adj_mtx.dot(combine_data[prot_name])
rna_count_ni = adj_mtx.dot(combine_data[gene])
ni_count = pd.DataFrame({"rna_count_ni": rna_count_ni, "prot_count_ni": prot_count_ni}, 
                            index=combine_data.index)
gene_data = pd.concat((combine_data[gene], combine_data[prot_name], ni_count), axis=1)
gene_data

# %% we could intepret the relevence of the rna and protein by the linear regression
# RNA count ~ protein count + protein count of neighbor spots + noise
# RNA count linked to RNA count of neighbor spots, which could be generated by convolution of the adj_RNA count
# Transform the RNA count by the size factor of the RNA count and neighbor RNA count
# Transform the protein count by the size factor of the protein count and neighbor protein count

# using the linear regression to get the coefficient of the protein count and protein count of neighbor
def stTransform(X, adjacent_mtx, alpha=0.01, fit_intercept=True, zero_inflated=False, logarithmic=False):
    """
    X: data matrix for RNA or protein count of each spot
    """
    # spot_size_factor = np.asarray(X.sum(axis=0) / X.median(np.sum(X, axis=0)))
    adjacent_mtx = np.eye(adjacent_mtx.shape[0]) + adjacent_mtx
    spot_size_factor = np.asarray(adjacent_mtx.dot(X).sum(axis=0) / np.median(adjacent_mtx.dot(X)).sum(axis=0))
    X_scaled = X / spot_size_factor
    gene_means = np.asarray(adjacent_mtx.dot(X).mean(axis=0))
    # fit the glm model for the spot size factor and neighbor size factor
    if zero_inflated:
        glm_model = sm.ZeroInflatedNegativeBinomialP(gene_means, sm.add_constant(np.log1p(spot_size_factor)),
                        exog_infl=sm.add_constant(np.log1p(spot_size_factor)))
    else:
        glm_model = sm.GLM(gene_means, sm.add_constant(np.log1p(spot_size_factor)), 
                           family=sm.families.NegativeBinomial(alpha=alpha))
    glm_results = glm_model.fit()
    
    print(glm_results)
    breakpoint()
    # glm transform
    mu = glm_results.predict(sm.add_constant(np.log1p(spot_size_factor)))
    var = np.sqrt(glm_results.mu + spot_size_factor * glm_results.mu ** 2)
    z = (X_scaled - mu) / var
    
    # print the summary of the glm model
    print(glm_results.summary())
    plt.scatter(z.mean(), z.var())
    plt.show()
# %%
col_names = combine_data.columns.str.endswith("_prot")
protein_count = combine_data.loc[:,col_names]
#%%
protein_count_transformed = stTransform(protein_count, adj_mtx, alpha=0.01, zero_inflated=False)
# %%
# X = protein_count
adjacent_mtx = adj_mtx
zero_inflated = False
adjacent_mtx = np.eye(adjacent_mtx.shape[0]) + adjacent_mtx
spot_size_factor = np.asarray(adjacent_mtx.dot(protein_count).sum(axis=0) / np.median(adjacent_mtx.dot(protein_count)))
protein_count_scaled = protein_count / spot_size_factor
gene_means = np.asarray(adjacent_mtx.dot(protein_count).mean(axis=0))
df = pd.DataFrame({"gene_means": gene_means, "spot_size_factor": spot_size_factor})
formula = "gene_means ~ np.log1p(spot_size_factor) + 1"
y, X = dmatrices(formula, df, return_type='dataframe')
# fit the glm model for the spot size factor and neighbor size factor
if zero_inflated:
    glm_model = sm.ZeroInflatedNegativeBinomialP(y, X, exog_infl=X, inflation="logit")
else:
    glm_model = sm.GLM(gene_means, sm.add_constant(np.log1p(spot_size_factor)), 
                        family=sm.families.NegativeBinomial(alpha=0.01))
glm_results = glm_model.fit()
# %%
glm_results.summary()

# %%
import pandas as pd

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']},
                    index=[0, 1, 2, 3])
df2 = pd.DataFrame({'C': ['C2', 'C3', 'C4', 'C5'],
                    'D': ['D2', 'D3', 'D4', 'D5']},
                    index=[0, 1, 2, 3])

df3 = pd.merge(df1, df2, left_index=True, right_index=True)
df3
# %%
