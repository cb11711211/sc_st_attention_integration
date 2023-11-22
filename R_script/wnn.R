# knitr::opts_chunk$set(echo = TRUE)

library(Seurat)
library(Matrix)
library(ggplot2)
library(patchwork)
library(dplyr)
library(plyr)

# Set folder location for saving output files. This is also the same location as input data.
mydir <- "data/citeseq/"
setwd("/home/rstudio/materials/") 

# Objects to save.
Rda.quickload.path <- paste0(mydir, "citeseq_quickload.Rda")  # datasets saved as sparse objects
Rda.RNA.path <- paste0(mydir, "citeseq_RNA.Rda")  # cbmc clustered using RNA
Rda.multi.path <- paste0(mydir, "citesq_cbmc_multi.rda")  # cbmc clustered and ADT added as an assay
Rda.protein.path <- paste0(mydir, "citeseq_protein.Rda")  # cbmc clustered using protein

## Load data
# Load in the RNA UMI matrix

# Note that this dataset also contains ~5% of mouse cells, which we can use
# as negative controls for the protein measurements. For this reason, the
# gene expression matrix has HUMAN_ or MOUSE_ appended to the beginning of
# each gene.
cbmc.rna <- as.sparse(read.csv(paste0(mydir, "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz"), sep = ",", header = TRUE, row.names = 1))
cbmc.rna[20400:20403,1:2]

# To make life a bit easier going forward, we're going to discard all but
# the top 100 most highly expressed mouse genes, and remove the 'HUMAN_'
# from the CITE-seq prefix
cbmc.rna <- CollapseSpeciesExpressionMatrix(cbmc.rna, prefix = "HUMAN_", controls = "MOUSE_")

# Load in the ADT UMI matrix
cbmc.adt <- as.sparse(read.csv(paste0(mydir, "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz"), sep = ",", header = TRUE, row.names = 1))

# When adding multimodal data to Seurat, it's okay to have duplicate feature names. Each set of
# modal data (eg. RNA, ADT, etc.) is stored in its own Assay object.  One of these Assay objects
# is called the 'default assay', meaning it's used for all analyses and visualization.  To pull
# data from an assay that isn't the default, you can specify a key that's linked to an assay for
# feature pulling.  To see all keys for all objects, use the Key function.  

# Lastly, we observed poor enrichments for CCR5, CCR7, and CD10 - and therefore 
# remove them from the matrix (optional)
cbmc.adt <- cbmc.adt[setdiff(rownames(x = cbmc.adt), c("CCR5", "CCR7", "CD10")), ]

# Look at structure of ADT matrix.
cbmc.adt[1:10,1:3]

# What fraction of cells in the ADT and RNA matrix overlap?
length(intersect(colnames(cbmc.rna), colnames(cbmc.adt))) / length(union(colnames(cbmc.rna), colnames(cbmc.adt)))

# Save current progress.
save(cbmc.rna, cbmc.adt, file = Rda.quickload.path)

## Set up Seurat object
load(Rda.quickload.path)
cbmc <- CreateSeuratObject(counts = cbmc.rna)
# This code sub-samples the data in order to speed up calculations and not use too much memory.
# Idents(cbmc) <- "orig.ident"
# cbmc <- subset(cbmc, downsample = 2000, seed = 1)
# cbmc.adt <- cbmc.adt[, colnames(cbmc)]

# standard log-normalization
cbmc <- NormalizeData(cbmc)

# choose ~1k variable features
cbmc <- FindVariableFeatures(cbmc)

# standard scaling (no regression)
cbmc <- ScaleData(cbmc)

# Run PCA, select PCs for tSNE visualization and graph-based clustering
cbmc <- RunPCA(cbmc, verbose = FALSE)
ElbowPlot(cbmc, ndims = 50)
# Cluster the cells using the first 25 principal components.
cbmc <- FindNeighbors(cbmc, dims = 1:25)
cbmc <- FindClusters(cbmc, resolution = 0.8)
cbmc <- RunTSNE(cbmc, dims = 1:25, method = "FIt-SNE")

# Find the markers that define each cluster, and use these to annotate the
# clusters, we use max.cells.per.ident to speed up the process
cbmc.rna.markers <- FindAllMarkers(cbmc, max.cells.per.ident = 100, logfc.threshold = log(2), only.pos = TRUE, min.diff.pct = 0.3)

# Which cluster consists of mouse cells?
cbmc.rna.markers %>% filter(cluster == 5)
cbmc.rna.markers %>% filter(cluster == 13)

# Note, for simplicity we are merging two CD14+ Monocyte clusters (that differ in expression of
# HLA-DR genes) and NK clusters (that differ in cell cycle stage)
new.cluster.ids <- c("Memory CD4 T", "CD14+ Mono", "Naive CD4 T", "NK", "CD14+ Mono", "Mouse", "B", "CD8 T", "CD16+ Mono", "T/Mono doublets", "NK", "CD34+", "Multiplets", "Mouse", "Eryth", "Mk", "Mouse", "DC", "pDCs")
names(new.cluster.ids) <- levels(cbmc)
cbmc <- RenameIdents(cbmc, new.cluster.ids)

# Visualize clustering based on RNA.
DimPlot(cbmc, label = TRUE, reduction = "tsne") + NoLegend()

# Save current progress.
save(cbmc, cbmc.rna.markers, cbmc.adt, file = Rda.RNA.path)
# To load the data, run the following command.
# load(Rda.RNA.path)

## Add the protein expression levels to the Seurat Object
# We will define an ADT assay, and store raw counts for it

# If you are interested in how these data are internally stored, you can check out the Assay
# class, which is defined in objects.R; note that all single-cell expression data, including RNA
# data, are still stored in Assay objects, and can also be accessed using GetAssayData
cbmc[["ADT"]] <- CreateAssayObject(counts = cbmc.adt)

GetAssayData(cbmc, slot = "counts", assay = "ADT")[1:3,1:3]
cbmc[["ADT"]]@counts[1:3, 1:3]

# Now we can repeat the preprocessing (normalization and scaling) steps that we typically run
# with RNA, but modifying the 'assay' argument.  For CITE-seq data, we do not recommend typical
# LogNormalization. Instead, we use a centered log-ratio (CLR) normalization, computed
# independently for each feature.  This is a slightly improved procedure from the original
# publication, and we will release more advanced versions of CITE-seq normalizations soon.
cbmc <- NormalizeData(cbmc, assay = "ADT", normalization.method = "CLR")
cbmc <- ScaleData(cbmc, assay = "ADT")

## Visualize protein levels on RNA clusters
DefaultAssay(cbmc) <- "RNA"

# In this plot, protein (ADT) levels are on top, and RNA levels are on the bottom
FeaturePlot(cbmc, features = c("adt_CD3", "adt_CD11c", "adt_CD8", "adt_CD16", "CD3E", "ITGAX", "CD8A", "FCGR3A"), min.cutoff = "q05", max.cutoff = "q95", ncol = 4)

# How do the gene and protein expression levels compare to one another?

# Compare gene and protein expression levels for the other 6 antibodies.
FeaturePlot(cbmc, features = c("adt_CD4", "adt_CD45RA", "adt_CD56", "adt_CD14", "adt_CD19", "adt_CD34", "CD4", "PTPRC", "NCAM1", "CD14", "CD19", "CD34"), min.cutoff = "q05", max.cutoff = "q95", ncol = 6)

# Ridge plots are another useful visualization.
RidgePlot(cbmc, features = c("adt_CD3", "adt_CD11c", "adt_CD8", "adt_CD16"), ncol = 2)

# Draw ADT scatter plots (like biaxial plots for FACS). Note that you can even 'gate' cells if
# desired by using HoverLocator and CellSelector
FeatureScatter(cbmc, feature1 = "adt_CD19", feature2 = "adt_CD3")
# HoverLocator(FeatureScatter(cbmc, feature1 = "adt_CD19", feature2 = "adt_CD3"))
# CellSelector(FeatureScatter(cbmc, feature1 = "adt_CD19", feature2 = "adt_CD3"))

# View relationship between protein and RNA
FeatureScatter(cbmc, feature1 = "adt_CD3", feature2 = "CD3E")
# Let's plot CD4 vs CD8 levels in T cells
tcells <- subset(cbmc, idents = c("Naive CD4 T", "Memory CD4 T", "CD8 T"))
FeatureScatter(tcells, feature1 = "adt_CD4", feature2 = "adt_CD8")

# Let's look at the raw (non-normalized) ADT counts. You can see the values are quite high,
# particularly in comparison to RNA values. This is due to the significantly higher protein copy
# number in cells, which significantly reduces 'drop-out' in ADT data
FeatureScatter(tcells, feature1 = "adt_CD4", feature2 = "adt_CD8", slot = "counts")

# If you look a bit more closely, you'll see that our CD8 T cell cluster is
# enriched for CD8 T cells, but still contains many CD4+ CD8- T cells.  This
# is because Naive CD4 and CD8 T cells are quite similar transcriptomically,
# and the RNA dropout levels for CD4 and CD8 are quite high.  This
# demonstrates the challenge of defining subtle immune cell differences from
# scRNA-seq data alone.

# What fraction of T cells are double negative in gene expression? (CD4- and CD8-)
# You can use an interactive plot to gate on the cells (do.identify = T) or use 
# Boolean conditions on CD4 and CD8A expression to find double negative cells.
FeatureScatter(tcells, feature1 = "CD4", feature2 = "CD8A")

ncol(subset(tcells, subset = CD4 == 0 & CD8A == 0)) / ncol(tcells)

# What fraction of T cells are double negative in protein expression? (CD4- and CD8-)
# length(cells) / length(tcells@cell.names)
DefaultAssay(tcells) <- "ADT"  # work with ADT count matrix
FeatureScatter(tcells, feature1 = "adt_CD4", feature2 = "adt_CD8")

ncol(subset(tcells, subset = adt_CD4 < 1 & adt_CD8 < 1)) / ncol(tcells)
# Save current progress.
save(cbmc, file = Rda.multi.path)
# To load the data, run the following command.
# load(Rda.multi.path)

## Identify differentially expressed proteins between clusters
# Downsample the clusters to a maximum of 300 cells each (makes the heatmap easier to see for
# small clusters)
cbmc.small <- subset(cbmc, downsample = 300)

# Find protein markers for all clusters, and draw a heatmap
adt.markers <- FindAllMarkers(cbmc.small, assay = "ADT", only.pos = TRUE)
DoHeatmap(cbmc.small, features = unique(adt.markers$gene), assay = "ADT", angle = 90) + NoLegend()
# You can see that our unknown cells co-express both myeloid and lymphoid markers (true at the
# RNA level as well). They are likely cell clumps (multiplets) that should be discarded. We'll
# remove the mouse cells now as well
cbmc <- subset(cbmc, idents = c("Multiplets", "Mouse"), invert = TRUE)

## Cluster directly on protein levels
# Because we're going to be working with the ADT data extensively, we're going to switch the
# default assay to the 'CITE' assay.  This will cause all functions to use ADT data by default,
# rather than requiring us to specify it each time
DefaultAssay(cbmc) <- "ADT"
cbmc <- RunPCA(cbmc, features = rownames(cbmc), reduction.name = "pca_adt", reduction.key = "pca_adt_", verbose = FALSE)
DimPlot(cbmc, reduction = "pca_adt")

# Why do we not use PCA to do dimensionality reduction here?
# Is Euclidean distance a good distance metric in this case?
ElbowPlot(cbmc)

# Since we only have 10 markers, instead of doing PCA, we'll just use a standard euclidean
# distance matrix here.  Also, this provides a good opportunity to demonstrate how to do
# visualization and clustering using a custom distance matrix in Seurat
adt.data <- GetAssayData(cbmc, slot = "data")
adt.dist <- dist(t(adt.data))

# Before we recluster the data on ADT levels, we'll stash the RNA cluster IDs for later
cbmc[["rnaClusterID"]] <- Idents(cbmc)

# Now, we rerun tSNE using our distance matrix defined only on ADT (protein) levels.
cbmc[["tsne_adt"]] <- RunTSNE(adt.dist, assay = "ADT", reduction.key = "adtTSNE_")
cbmc[["adt_snn"]] <- FindNeighbors(adt.dist)$snn

cbmc <- FindClusters(cbmc, resolution = 0.2, graph.name = "adt_snn")

# We can compare the RNA and protein clustering, and use this to annotate the protein clustering
# (we could also of course use FindMarkers)
clustering.table <- table(Idents(cbmc), cbmc$rnaClusterID)
clustering.table

new.cluster.ids <- c("CD4 T", "CD14+ Mono", "NK", "B", "CD8 T", "NK", "CD34+", "T/Mono doublets", "CD16+ Mono", "pDCs", "B")
names(new.cluster.ids) <- levels(cbmc)
cbmc <- RenameIdents(cbmc, new.cluster.ids)

tsne_rnaClusters <- DimPlot(cbmc, reduction = "tsne_adt", group.by = "rnaClusterID", pt.size = 0.5) + NoLegend()
tsne_rnaClusters <- tsne_rnaClusters + ggtitle("Clustering based on scRNA-seq") + theme(plot.title = element_text(hjust = 0.5))
tsne_rnaClusters <- LabelClusters(plot = tsne_rnaClusters, id = "rnaClusterID", size = 4)

tsne_adtClusters <- DimPlot(cbmc, reduction = "tsne_adt", pt.size = 0.5) + NoLegend()
tsne_adtClusters <- tsne_adtClusters + ggtitle("Clustering based on ADT signal") + theme(plot.title = element_text(hjust = 0.5))
tsne_adtClusters <- LabelClusters(plot = tsne_adtClusters, id = "ident", size = 4)

# Note: for this comparison, both the RNA and protein clustering are visualized on a tSNE
# generated using the ADT distance matrix.
wrap_plots(list(tsne_rnaClusters, tsne_adtClusters), ncol = 2)

# What differences if any do you see between the clustering based on scRNA-seq
# and the clustering based on ADT signal?
# How could we combine these datasets in a joint, integrative analysis?

# Save current progress.
save(cbmc, file = Rda.protein.path)
# To load the data, run the following command.
# load(Rda.protein.path)

