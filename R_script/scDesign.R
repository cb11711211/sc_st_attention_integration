library(scDesign3)
library(SingleCellExperiment)
library(dplyr)
library(ggplot2)
library(ggh4x)
library(umap)
theme_set(theme_bw())

# Read in the reference data
SCGEMMETH_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581998")))
SCGEMRNA_sce <-readRDS((url("https://figshare.com/ndownloader/files/40582001")))
print(SCGEMMETH_sce)
print(SCGEMRNA_sce)

coldata_bind <- bind_rows(as_tibble(colData(SCGEMRNA_sce)) %>% dplyr::mutate(Tech = "RNA"), as_tibble(colData(SCGEMMETH_sce)) %>% dplyr::mutate(Tech = "Methylation"))
head(coldata_bind)

# Simulation
set.seed(123)
RNA_data <- scDesign3::construct_data(
  SCGEMRNA_sce, 
  assay_use = "logcounts", 
  celltype = "cell_type", 
  pseudotime = NULL, 
  spatial = c("UMAP1_integrated", "UMAP2_integrated"), 
  other_covariates = NULL, 
  corr_by = "1"
  )
METH_data <- scDesign3::construct_data(
  SCGEMMETH_sce, 
  assay_use = "counts", 
  celltype = "cell_type", 
  pseudotime = NULL, 
  spatial = c("UMAP1_integrated", "UMAP2_integrated"), 
  other_covariates = NULL, 
  corr_by = "1")

RNA_marginal <- scDesign3::fit_marginal(
  data = RNA_data, 
  predictor = "gene", 
  mu_formula = "te(UMAP1_integrated, UMAP2_integrated, bs = 'cr', k = 10)", 
  sigma_formula = "te(UMAP1_integrated, UMAP2_integrated, bs = 'cr', k = 5)", 
  family_use = "gaussian", 
  n_cores = 1, 
  usebam = FALSE)

METH_marginal <- scDesign3::fit_marginal(
  data = METH_data, 
  predictor = "gene", 
  mu_formula = "te(UMAP1_integrated, UMAP2_integrated, bs = 'cr', k = 10)", 
  sigma_formula = "1", 
  family_use = "binomial", 
  n_cores = 1, 
  usebam = FALSE)

RNA_copula <- scDesign3::fit_copula(
    sce = SCGEMRNA_sce,
    assay_use = "logcounts",
    marginal_list = RNA_marginal,
    family_use = "gaussian",
    copula = "vine",
    n_cores = 1,
    input_data = RNA_data$dat
  )

METH_copula <- scDesign3::fit_copula(
    sce = SCGEMMETH_sce,
    assay_use = "counts",
    marginal_list = METH_marginal,
    family_use = "binomial",
    copula = "vine",
    n_cores = 1,
    input_data = METH_data$dat
  )

RNA_para <- extract_para(
    sce = SCGEMRNA_sce,
    assay_use = "logcounts",
    marginal_list = RNA_marginal,
    n_cores = 1,
    family_use = "gaussian",
    new_covariate = rbind(RNA_data$dat, METH_data$dat),
    data = RNA_data$dat
  )

METH_para <- extract_para(
    sce = SCGEMMETH_sce,
    marginal_list = METH_marginal,
    n_cores = 1,
    family_use = "binomial",
    new_covariate = rbind(RNA_data$dat, METH_data$dat),
    data = METH_data$dat
  )

# Simulate New Datasets
RNA_res <- simu_new(
    sce = SCGEMRNA_sce,
    assay_use = "logcounts",
    mean_mat = RNA_para$mean_mat,
    sigma_mat = RNA_para$sigma_mat,
    zero_mat = RNA_para$zero_mat,
    quantile_mat = NULL,
    copula_list = RNA_copula$copula_list,
    n_cores = 1,
    family_use = "gaussian",
    input_data = RNA_data$dat,
    new_covariate = rbind(RNA_data$dat, METH_data$dat),
    important_feature = RNA_copula$important_feature,
    filtered_gene = RNA_data$filtered_gene
  )
METH_res <- simu_new(
    sce = SCGEMMETH_sce,
    mean_mat = METH_para$mean_mat,
    sigma_mat = METH_para$sigma_mat,
    zero_mat = METH_para$zero_mat,
    quantile_mat = NULL,
    copula_list = METH_copula$copula_list,
    n_cores = 1,
    family_use = "binomial",
    input_data = METH_data$dat,
    new_covariate = rbind(RNA_data$dat, METH_data$dat),
    important_feature = METH_copula$important_feature,
    filtered_gene = METH_data$filtered_gene
  )

# Visualization
count_combine <- rbind(RNA_res, METH_res)
count_combine_pca <- irlba::prcomp_irlba(t(count_combine), 5, scale. = TRUE)

count_combine_umap <- umap::umap(count_combine_pca$x, n_neighbors=30, min_dist=0.7)$layout
colnames(count_combine_umap) <- c("UMAP1", "UMAP2")

SCGEMNEW_sce <- SingleCellExperiment::SingleCellExperiment(list(logcounts = count_combine))
reducedDims(SCGEMNEW_sce) <- list(PCA = count_combine_pca$x, UMAP = count_combine_umap)

SCGEMRNA_umap <- umap::umap(colData(SCGEMRNA_sce) %>% as_tibble() %>% dplyr::select(paste0("X", 1:5)), n_neighbors=30, min_dist=0.7)
SCGEMRNA_umap <- SCGEMRNA_umap$layout
colnames(SCGEMRNA_umap) <- c("UMAP1", "UMAP2")
reducedDim(SCGEMRNA_sce, "UMAP") <- SCGEMRNA_umap

dat_RNA <- SCGEMRNA_umap %>% as_tibble() %>% dplyr::mutate(Method = "Real data: RNA")
dat_METH <- SCGEMMETH_umap %>% as_tibble() %>% dplyr::mutate(Method = "Real data: Methylation")
dat_NEW <- reducedDim(SCGEMNEW_sce, "UMAP") %>% as_tibble() %>% dplyr::mutate(Method = "scDesign3: RNA + Meythlation")
SCGEM_dat <- bind_rows(list(dat_RNA, dat_METH, dat_NEW))

design <- matrix(c(2,3,1,3), 2, 2) %>% t()
dat_text_SCGEM <- tibble(Method = c(
    "Real data: RNA", "Real data: Methylation", "scDesign3: RNA + Meythlation"), label = c("32 Features*177 Cells", "27 Features*142 Cells", "59 Features*319 Cells")) %>% as.data.frame()

SCGEM_dat <- SCGEM_dat %>% dplyr::mutate(Method = factor(Method, levels = c("Real data: RNA", "scDesign3: RNA + Meythlation", "Real data: Methylation"))) %>% dplyr::mutate(UMAP1 = if_else(Method == "Real data: RNA", -UMAP1, UMAP1), UMAP2 = if_else(Method == "Real data: RNA", -UMAP2, UMAP2))

p_merge_modals <- ggplot(SCGEM_dat, aes(UMAP1, UMAP2, colour = Method)) + ggrastr::rasterize(geom_point(size = 0.5), dpi = 300) +
  guides(colour = "none") + scale_color_brewer(palette = "Set2") + theme_bw() + theme(aspect.ratio = 1,
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank(),
    axis.text.y=element_blank(),
    axis.ticks.y=element_blank()) + facet_manual(~Method, design = design, widths = c(1, 2), heights = c(1, 1), respect = TRUE, scales = "free")+ geom_text(
  data    = dat_text_SCGEM,
  mapping = aes(x = Inf, y = -Inf, label = label), vjust = -6, hjust = 1, color = "black", size = 3)
p_merge_modals


## Simulate CITE-seq data
library(scDesign3)
library(SingleCellExperiment)
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyr)
library(scales)
library(ggh4x)
theme_set(theme_bw())

example_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581968")))
print(example_sce)

keep_gene <- c("CD4",  "CD14", "CD19", "CD34", "CD3E", "CD8A")
keep_adt <- c("ADT_CD4", "ADT_CD14", "ADT_CD19", "ADT_CD34", "ADT_CD3", "ADT_CD8")
keep <- c(keep_gene, keep_adt)
idx <- which(rownames(example_sce) %in% keep)
idx <- c(1:100,idx)
example_sce <- example_sce[idx,]
logcounts(example_sce) <- log1p(counts(example_sce))

# Simulation
set.seed(123)
example_simu <- scdesign3(
    sce = example_sce,
    assay_use = "counts",
    celltype = "cell_type",
    pseudotime = NULL,
    spatial = NULL,
    other_covariates = NULL,
    mu_formula = "cell_type",
    sigma_formula = "cell_type",
    family_use = "nb",
    n_cores = 2,
    usebam = FALSE,
    corr_formula = "cell_type",
    copula = "vine",
    DT = TRUE,
    pseudo_obs = FALSE,
    return_model = FALSE,
    nonzerovar = TRUE,
    nonnegative = TRUE
  )

logcounts(example_sce) <- log1p(counts(example_sce))
simu_sce <- SingleCellExperiment(list(counts = example_simu$new_count), colData = example_simu$new_covariate)
logcounts(simu_sce) <- log1p(counts(simu_sce))

set.seed(123)
 train_pca_fit <- irlba::prcomp_irlba(t(log1p(counts(example_sce))), 
                                          center = TRUE, 
                                          scale. = FALSE, 
                                          n = 50)
reducedDim(simu_sce, "PCA") <- predict(train_pca_fit, newdata= t(log1p(counts(simu_sce))))
simu_pac_fit <- predict(train_pca_fit, newdata= t(logcounts(simu_sce)))
train_umap_fit <- umap::umap(train_pca_fit$x, n_neighbors = 15, min_dist = 0.1)
simu_umap_fit <-  predict(object = train_umap_fit, data= (reducedDim(simu_sce, "PCA")))
colnames(simu_umap_fit ) <- c("UMAP1", "UMAP2")
reducedDim(simu_sce, "UMAP") <- simu_umap_fit 
train_umap <- train_umap_fit$layout
rownames(train_umap) <- colnames(example_sce)
colnames(train_umap) <- c("UMAP1", "UMAP2")

# Visualization
expression_train <- as.matrix(logcounts(example_sce))[c(keep_gene ,keep_adt), ] %>% t()  %>% as_tibble() %>% bind_cols(train_umap) %>% dplyr::mutate(Method = "Train data")
expression_scDesign3 <- as.matrix(logcounts(simu_sce))[c(keep_gene ,keep_adt), ] %>% t() %>% as_tibble() %>% bind_cols(simu_umap_fit) %>% dplyr::mutate(Method = "scDesign3")


CITE_dat <- bind_rows(expression_train, expression_scDesign3) %>% as_tibble() %>%
            dplyr::mutate_at(vars(-c(UMAP1, UMAP2, Method)), funs(scales::rescale)) %>% tidyr::pivot_longer(-c("UMAP1", "UMAP2", "Method"), names_to = "Feature", values_to = "Expression") %>% dplyr::mutate(Type = if_else(str_detect(Feature, "ADT"), "Protein", "RNA")) %>% dplyr::mutate(Gene = str_replace(Feature, "ADT_", "")) %>% dplyr::mutate(Gene = if_else(Gene == "CD3E", "CD3", Gene))%>% dplyr::mutate(Gene = if_else(Gene == "CD8A", "CD8", Gene))%>% dplyr::filter(Gene %in% c("CD14", "CD3", "CD8", "CD19")) %>% dplyr::mutate(Gene = factor(Gene, levels = c("CD3", "CD8", "CD14", "CD19"))) %>% dplyr::mutate(Method = factor(Method, levels = c("Train data", "scDesign3")))
head(CITE_dat)

CITE_dat  %>% ggplot(aes(x = UMAP1, y = UMAP2, color = Expression)) + geom_point(size = 0.1, alpha = 0.5) + scale_colour_gradientn(colors = viridis_pal(option = "A", direction = -1)(10), limits=c(0, 1)) + coord_fixed(ratio = 1) + facet_nested(Method ~ Gene + Type ) + theme(aspect.ratio = 1, legend.position = "bottom")  + theme(aspect.ratio = 1, legend.position = "right") + theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank(),
    axis.text.y=element_blank(),
    axis.ticks.y=element_blank())