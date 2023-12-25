import muon as mu
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
# General rules:
# 1. message passing edge: used for GNN message passing
# 2. supervision edge: used in loss function for backpropagation

## design for transductive learning
transductive_tsf = RandomNodeSplit(
    num_splits=5, # 5-fold cross validation
    num_val=0.2, # 10% of training data for validation
    num_test=0.2, # 10% of training data for testing
)

## design for inductive learning
inductive_tsf = RandomLinkSplit(
    is_undirected=True, # undirected graph
    num_val=0.2, # 10% of training data for validation
    num_test=0.2, # 10% of training data for testing
)

# load the dataset and save to a local directory
def get_gene_dict(adata):
    gene_dict = {}
    for i, gene in enumerate(adata.var_names):
        gene_dict[gene] = i
    return gene_dict

class GeneVocab():
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