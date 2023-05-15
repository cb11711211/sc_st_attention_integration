# %% import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
# import stereo as st
from scipy.sparse import coo_matrix
import scipy.stats as stats
import statsmodels.api as sm
from patsy import dmatrices
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
fig, axes = plt.subplots(2,2 ,figsize=(10, 10), layout="tight")
axes = sc.pl.spatial(adata, spot_size=0.5, color=["Cd8a", "Cd4", "log1p_total_counts", "log1p_n_genes_by_counts"], show=False)
axes[0].set_title("CD8a")
axes[1].set_title("CD4")
axes[2].set_title("log1p_total_counts")
axes[3].set_title("log1p_n_genes_by_counts")
# set the layout of the figure
fig.tight_layout()

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

# %% convert the prot_data to a sparse matrix
from scipy.sparse import csr_matrix, vstack

spatial_mtx = None
for gene_name in prot_data.columns:
    if spatial_mtx is None:
        spatial_mtx = align_prot_rna_mtx(prot_data, gene_name)
    else:
        spatial_mtx_gene = align_prot_rna_mtx(prot_data, gene_name)
        spatial_mtx = np.hstack([spatial_mtx, spatial_mtx_gene])
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
gene = "Cd3g"
prot_name = gene + "_prot"
# combine_data[gene], combine_data[prot_name]
prot_count_ni = adj_mtx.dot(combine_data[prot_name])
rna_count_ni = adj_mtx.dot(combine_data[gene])
ni_count = pd.DataFrame({"rna_count_ni": rna_count_ni, "prot_count_ni": prot_count_ni}, 
                            index=combine_data.index)
gene_data = pd.concat((combine_data[gene], combine_data[prot_name], ni_count), axis=1)
gene_data
# %%
import statsmodels.formula.api as smf

formula = "Cd3g_prot ~ Cd3g + rna_count_ni + prot_count_ni + rna_count_ni * prot_count_ni"
mod = smf.glm(formula=formula, data=gene_data, family=sm.families.Poisson()).fit()
print(mod.summary())
# %%
mod2 = smf.glm(formula=formula, data=gene_data, family=sm.families.NegativeBinomial()).fit()
print(mod2.summary())
# %%
np.bincount(gene_data["Cd3g_prot"].astype(int))
# %% using the model to reversely predict RNA_count
pred = mod.predict(gene_data)

# %% show correlation between the protein count and RNA count
fig, axs = plt.subplots(1,2, figsize=(10, 5))
axs[0].scatter(np.log10(gene_data["Cd3g_prot"]), gene_data["Cd3g"], s=2)
axs[0].set_title(f"{gene} #RNA vs log #Protein", fontsize=18)
axs[0].set_xlabel("log #Protein", fontsize=18)
axs[0].set_ylabel("#RNA", fontsize=18)
axs[1].scatter(gene_data["Cd3g_prot"], gene_data["prot_count_ni"], s=2)
axs[1].set_title(f"{gene} neighbor #protein vs #protein", fontsize=18)
axs[1].set_xlabel("#Protein", fontsize=18)
axs[1].set_ylabel("Neighbor #Protein", fontsize=18)
plt.tight_layout()
plt.show()
# %%
import seaborn as sns
# gene_data
sns.pairplot(gene_data, kind="reg", diag_kind="kde")

# %%
formula = "Cd3g ~ np.log(Cd3g_prot) + np.log(prot_count_ni) + rna_count_ni"
model_log_nb = smf.glm(formula=formula, data=gene_data, family=sm.families.NegativeBinomial()).fit()
model_log_po = smf.glm(formula=formula, data=gene_data, family=sm.families.Poisson()).fit()
print(model_log_nb.summary())
print(model_log_po.summary())

# %%
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.stats import poisson, nbinom, randint
from sklearn.utils import check_random_state

def neg_bic_score(y_true, y_pred, n_features):
    return -2 * mean_squared_error(y_true, y_pred) + n_features * np.log(len(y_true))

def log_transform(x):
    return np.log(x + 1)

def log_transform_inv(x):
    return np.exp(x) - 1

def poisson_glm(X, y, alpha=0.0, fit_intercept=True):
    glm = Pipeline(steps=[
        ('scalar', StandardScaler()),
        ('model', TransformedTargetRegressor(
            regressor=PoissonRegressor(
                alpha=alpha,
                fit_intercept=fit_intercept), 
            func=log_transform, 
            inverse_func=log_transform_inv
            )
        )
    ])
    glm.fit(X, y)
    return glm

def nb_glm(X, y, alpha=0.0, fit_intercept=True):
    def nb_ll(y, pred, alpha):
        return -nbinom.logpmf(y, alpha, pred/(alpha+pred))
    glm = Pipeline(steps=[
        ('scalar', StandardScaler()),
        ('model', TransformedTargetRegressor(
            regressor=TweedieRegressor(
                power=0,
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=1000,
                tol=1e-6,
                link='log'
                # objective=nb_ll
            ),
            func=log_transform,
            inverse_func=log_transform_inv
            )
        )
    ])
    glm.fit(X, y)
    return glm

def zip_glm(X, y, alpha=0.0, fit_intercept=True):
    # def zip_ll(y, pred, alpha):
    #     return -np.log(1-alpha) - poisson.logpmf(y, pred)
    glm = Pipeline(steps=[
        ('scalar', StandardScaler()),
        ('model', TransformedTargetRegressor(
            regressor=GammaRegressor(
                alpha=alpha,
                fit_intercept=fit_intercept
            ),
            func=log_transform,
            inverse_func=log_transform_inv)
        )]
    )
    param_grid = {
        'model__regressor__alpha': [0, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        'model__regressor__fit_intercept': [True, False],
        'model__regressor__max_iter': [1000],
        'model__regressor__tol': [1e-6],
        'model__regressor__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    random_state = check_random_state(0)
    grid_search = GridSearchCV(glm, param_grid, cv=5, 
                               scoring='neg_bic', 
                               n_jobs=16, verbose=1)
    grid_search.fit(X, y, model__sample_weight=random_state.randint(1, 100, size=len(y)))
    return grid_search.best_estimator_

def zinb_glm(X, y, alpha=0.0, fit_intercept=True):
    def zinb_ll(y, pred, alpha):
        return -nbinom.logpmf(y, alpha, pred/(alpha + pred))
    glm = Pipeline(steps=[
        ('scalar', StandardScaler()),
        ('model', TransformedTargetRegressor(
            regressor=TweedieRegressor(
                power=0,
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=1000,
                tol=1e-6,
                link='log'
                # objective=zinb_ll
            ),
            func=log_transform,
            inverse_func=log_transform_inv
            )
        )
    ])
    param_grid = {
        'model__regressor__alpha': [0, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        'model__regressor__fit_intercept': [True, False],
        'model__regressor__max_iter': [1000],
        'model__regressor__tol': [1e-6],
        'model__regressor__link': ['log'],
        'model__regressor__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    random_state = check_random_state(0)
    grid_search = GridSearchCV(glm, param_grid, cv=5, scoring='neg_bic', n_jobs=16, verbose=1)
    grid_search.fit(X, y, model__sample_weight=random_state.randint(1, 100, size=len(y)))
    return grid_search.best_estimator_
# %% split the data into X and y
X = gene_data[["rna_count_ni", "prot_count_ni", "Cd3g_prot"]]
y = gene_data["Cd3g"]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)
# %% fit the glm model
glm = poisson_glm(X_train, y_train)
# %%
glm.score(X_test, y_test)
# %%
nb_glm = nb_glm(X_train, y_train)
nb_glm.score(X_test, y_test)
# %%
zip_glm = zip_glm(X_train, y_train)
zip_glm.score(X_test, y_test)
# %%
# gene_data.fillna(0)
# gene_data.isna().sum()
df.isna().sum()
# %%
import statsmodels.formula.api as smf

df = gene_data

mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]
ds = df.index.to_series()
# df["Cd3g_prot"] = ds.dt.Cd3g_prot
# df["rna_count_ni"] = ds.dt.rna_count_ni
# df["prot_count_ni"] = ds.dt.prot_count_ni
formula = "Cd3g ~ Cd3g_prot + prot_count_ni + rna_count_ni + 1"
y_train, X_train = dmatrices(formula, df_train, return_type="dataframe")
y_test, X_test = dmatrices(formula, df_train, return_type="dataframe")
poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_training_results.summary())
print(poisson_training_results.mu)
print(len(poisson_training_results.mu))
# %%
# add the lambda vector to the df_train
df_train['BB_LAMBDA'] = poisson_training_results.mu 
# add the values of the dependent variable of the OLS regression
df_train['AUX_OLS_DEP'] = df_train.apply(
    lambda x: ((x["Cd3g"] - x["BB_LAMBDA"])**2 - x["BB_LAMBDA"]) / x["BB_LAMBDA"], axis=1)
ols_formula = "AUX_OLS_DEP ~ BB_LAMBDA - 1"
aux_olsr_results = smf.ols(ols_formula, df_train).fit()
# get the alpha
print(f"The OLSR regression params are: {aux_olsr_results.params}")
# get the t-values
print(f"The OLSR regression coefficient alpha: {aux_olsr_results.tvalues}")
nb2_training_results = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
print(nb2_training_results.summary())
# %% regress zip 
formula = "Cd3g ~ np.log1p(Cd3g_prot) + np.log1p(prot_count_ni) + rna_count_ni + 1"
y_train, X_train = dmatrices(formula, df_train, return_type="dataframe")
y_test, X_test = dmatrices(formula, df_train, return_type="dataframe")
# %%
# X = df_train[["Cd3g_prot", "rna_count_ni", "prot_count_ni"]]
# y = df_train["Cd3g"]
model = sm.ZeroInflatedNegativeBinomialP(y_train, X_train, exog_infl=X_train, inflation='logit').fit()
print(model.summary())
# %%
df_train



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
    nb_bic = -2 * nb_ll + np.log(data.shape[0]) * covariates.shape[1]

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