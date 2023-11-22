library(data.table)
library(purrr)
library(ggplot2)
library(ggpubr)
library(ggrepel)

# MOFA
library(MOFA2)

data <- fread("/Users/ricard/data/gastrulation/metaccrna/mefisto/vignette/scnmt_data.txt.gz")
head(data,n=3)
data[,length(unique(feature)),by="view"]

sample_metadata <- fread("/Users/ricard/data/gastrulation/metaccrna/mefisto/vignette/scnmt_sample_metadata.txt")
colnames(sample_metadata)

table(sample_metadata$lineage10x)
table(sample_metadata$lineage10x_2)

ggscatter(sample_metadata, x="UMAP1", y="UMAP2", color="lineage10x_2") +
  scale_color_manual(values=celltype.colors) +
  theme_classic() +
  ggplot_theme_NoAxes()

# Create and train the MEFISTO object
mefisto <- create_mofa_from_df(data)
mefisto

samples_metadata(mefisto) <- sample_metadata %>% 
  .[sample%in%unlist(samples_names(mefisto))] %>%
  setkey(sample) %>% .[unlist(samples_names(mefisto))]

mefisto <- set_covariates(mefisto, c("UMAP1","UMAP2"))
model_opts <- get_default_model_options(mefisto)
model_opts$num_factors <- 10
mefisto <- prepare_mofa(
  object = mefisto,
  model_options = model_opts,
)
plot_data_overview(mefisto)

mefisto <- run_mofa(mefisto)
mefisto <- readRDS("/Users/ricard/data/gastrulation/metaccrna/mefisto/vignette/scnmt_mefisto_model.rds")
r2 <- get_variance_explained(mefisto, views="RNA")[["r2_per_factor"]][[1]]
factors <- names(which(r2[,"RNA"]>1))
mefisto <- subset_factors(mefisto, factors)
mefisto

plot_variance_explained(mefisto, x="view", y="factor", max_r2 = 9)
plot_smoothness(mefisto)
factors.dt <- get_factors(mefisto, factors=c(1,3,4))[[1]] %>% 
  as.data.table(keep.rownames = T) %>% 
  setnames("rn","sample")

to.plot <- sample_metadata[,c("sample","UMAP1","UMAP2")] %>% 
  merge(factors.dt, by="sample") %>%
  melt(id.vars=c("UMAP1","UMAP2","sample"), variable.name="factor") %>%
  .[,value:=value/max(abs(value)),by="factor"]

ggscatter(to.plot, x="UMAP1", y="UMAP2", fill="value", shape=21, stroke=0.15) +
  facet_wrap(~factor) +
  scale_fill_gradient2(low = "gray50", mid="gray90", high = "red") +
  theme_classic() +
  ggplot_theme_NoAxes()

w.met <- get_weights(mefisto, views="motif_met",  factors=c(1,3), as.data.frame=T) %>% 
  as.data.table %>% .[,feature:=gsub("_met","",feature)] %>%
  .[,value:=value/max(abs(value)),by=c("factor")]

w.acc <- get_weights(mefisto, views="motif_acc", factors=c(1,3), as.data.frame=T) %>% 
  as.data.table %>% .[,feature:=gsub("_acc","",feature)] %>%
  .[,value:=value/max(abs(value)),by=c("factor")]

# Merge loadings
w.dt <- merge(
  w.met[,c("feature","factor","value")], 
  w.acc[,c("feature","factor","value")], 
  by = c("feature","factor")
) %>% .[,feature:=strsplit(feature,"_") %>% map_chr(c(1))]

head(w.dt)
for (i in unique(w.dt$factor)) {
  
  to.plot <- w.dt[factor==i]
  
  to.label <- w.dt %>% 
    .[factor==i] %>%
    .[,value:=abs(value.x)+abs(value.y)] %>% setorder(-value) %>% head(n=10)
  
  p <- ggscatter(to.plot, x="value.x", y="value.y", size=1.5, add="reg.line", conf.int=TRUE) +
    coord_cartesian(xlim=c(-1,1), ylim=c(-1,1)) +
    scale_x_continuous(breaks=c(-1,0,1)) +
    scale_y_continuous(breaks=c(-1,0,1)) +
    geom_text_repel(data=to.label, aes(x=value.x, y=value.y, label=feature), size=3,  max.overlaps=100) +
    geom_vline(xintercept=0, linetype="dashed") +
    geom_hline(yintercept=0, linetype="dashed") +
    stat_cor(method = "pearson") +
    labs(x="Methylation weights", y="Accessibility weights")
  
  print(p)
}

# Imputation
mefisto <- interpolate_factors(mefisto, mefisto@covariates[[1]])
Z_interpol <- t(get_interpolated_factors(mefisto, only_mean = TRUE)[[1]]$mean)

imputed_data <- list()
for (i in views_names(mefisto)) {
  imputed_data[[i]] <- tcrossprod(Z_interpol,get_weights(mefisto,i)[[1]]) %>% t
  colnames(imputed_data[[i]]) <- colnames(mefisto@data[[i]][[1]])
}

for (view in c("motif_met","motif_acc")) {
  
  # Define features to plot
  # features.to.plot <- features_names(mefisto)[view]
  features.to.plot <- list(paste0("MSGN1_412",gsub("motif","",view)))
  names(features.to.plot) <- view
  
  # Get original and imputed data
  data <- get_data(mefisto, views=view, features=features.to.plot)[[1]][[1]]
  data_imputed <- imputed_data[[view]][features.to.plot[[1]],,drop=F]
  
  for (i in features.to.plot[[view]]) {
    
    to.plot <- data.table(
      sample = unlist(samples_names(mefisto)),
      non_imputed = data[i,],
      imputed = data_imputed[i,]
    ) %>% 
      setnames(c("sample",sprintf("%s (original)",i),sprintf(" %s (MEFISTO imputed)",i))) %>%
      melt(id.vars="sample", value.name="value") %>% 
      merge(sample_metadata[,c("sample","UMAP1","UMAP2")], by="sample")
    
    # Scale min/max values for visualisation
    max.value <- max(to.plot[variable==sprintf(" %s (MEFISTO imputed)",i),value])
    min.value <- min(to.plot[variable==sprintf(" %s (MEFISTO imputed)",i),value])
    to.plot[value>max.value,value:=max.value]
    to.plot[value<min.value,value:=min.value]
    
    p <- ggscatter(to.plot, x="UMAP1", y="UMAP2", color="value") +
      facet_wrap(~variable) +
      theme_classic() +
      ggplot_theme_NoAxes() +
      theme(
        legend.position = "none",
      )
    
    if (view=="motif_met") {
      p <- p + scale_color_gradient2(low = "blue", mid="gray90", high = "red")
    } else if (view=="motif_acc") {
      p <- p + scale_color_gradient2(low = "yellow", mid="gray90", high = "purple")
    }
    
    print(p)
  }
}

# Gene set enrichment analysis
library(MOFAdata)
data("MSigDB_v6.0_C5_mouse") 
enrichment.out <- run_enrichment(
  object = mefisto,
  view = "RNA",
  feature.sets = MSigDB_v6.0_C5_mouse,
  statistical.test = "parametric",
  alpha = 0.01
)
plot_enrichment(enrichment.out, factor=4, max.pathways = 15)

