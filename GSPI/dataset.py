# load the dataset and save to a local directory
def get_gene_dict(adata):
    gene_dict = {}
    for i, gene in enumerate(adata.var_names):
        gene_dict[gene] = i
    return gene_dict

class GeneVocab():
    """
    The transfer learning model is based on the feature alignment of diverse datasets.
    The initial dataset is used to build the vocabulary of genes and then using new datasets
    to update the gene vocabulary. The gene vocabulary index should indicate the same gene
    across different datasets.
    """
    def __init__(self, adata_initial):
        self.gene_dict = get_gene_dict(adata_initial)
        self.gene_list = list(self.gene_dict.keys())
        self.gene_list.sort()
        self.gene2idx = {gene: i for i, gene in enumerate(self.gene_list)}
        self.idx2gene = {i: gene for i, gene in enumerate(self.gene_list)}
        self.vocab_size = len(self.gene_list)

    def update_gene_dict(self, adata_new):
        for gene in adata_new.var_names:
            if gene not in self.gene_dict:
                self.gene_dict[gene] = len(self.gene_dict)
                self.gene_list.append(gene)
                self.gene_list.sort()
                self.gene2idx = {gene: i for i, gene in enumerate(self.gene_list)}
                self.idx2gene = {i: gene for i, gene in enumerate(self.gene_list)}
                self.vocab_size = len(self.gene_list)

    def align_features(self, adata_new):
        """
        The index of features/gens should be aligned across different datasets.
        """
        self.update_gene_dict(adata_new)
        adata_new = adata_new[:, self.gene_list]
        return adata_new

    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx2gene[key]
        elif isinstance(key, str):
            return self.gene2idx[key]
        else:
            raise KeyError("key must be either int or str")
        
    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.idx2gene
        elif isinstance(key, str):
            return key in self.gene2idx
        else:
            raise KeyError("key must be either int or str")