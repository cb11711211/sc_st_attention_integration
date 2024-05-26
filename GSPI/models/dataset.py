import os
import sys
import torch
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from .utils import build_adjacency_matrix, build_adjacency_matrix_torch

# load the dataset and save to a local directory
def get_gene_dict(adata):
    gene_dict = {}
    for i, gene in enumerate(adata.var_names):
        gene_dict[gene] = i
    return gene_dict

def get_gene_position(gtf_file, gene_id_split=" "):
    """
    Reorder the genes based on the genomic position of genes
    """
    genomic_dict = {}
    # only print $2 == "gene"
    with open(gtf_file, "r") as f:
        for line in f:
            if not line.startswith("#"):
                line = line.strip().split("\t")
                if line[2] == "gene":
                    chr_name = line[0]
                    start_pos = line[3]
                    ids = line[8].split(";")
                    gene_name = ids[2 if gene_id_split == " " else 3].split(
                        gene_id_split)[2 if gene_id_split == " " else 1]
                    gene_name = gene_name.replace('"', '')
                    genomic_dict[gene_name] = [chr_name, int(start_pos)]
    genomic_df = pd.DataFrame(genomic_dict).T
    genomic_df.index.name = "gene_name"
    genomic_df.rename(columns={0: "chr", 1: "start", }, inplace=True)
    genomic_df.sort_values(by=["chr", "start"], inplace=True)
    return genomic_df
    

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

    def align_features(self, adata):
        """
        Input: adata, has the some genes but not all genes in gene_vocab
        Output: adata_new, has the same genes as gene_vocab
        """
        # filter the genes in adata that are duplicate
        adata = adata[:, ~adata.var_names.duplicated()]
        # filter the genes not present in the gene_vocab
        adata = adata[:, adata.var_names.isin(self.gene_list)]

        # padding the rna features which are in the initial gene_vocab but not in the new adata
        new_gene_list = adata.var_names
        blank_gene_list = list(set(self.gene_list).difference(set(new_gene_list)))
        # set the expression of blank genes to 0
        adata_new_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        blank_gene_df = pd.DataFrame(np.zeros((adata.shape[0], len(blank_gene_list))), 
                                    index=adata.obs_names, columns=blank_gene_list)
        adata_new_df = pd.concat([adata_new_df, blank_gene_df], axis=1)
        adata_new_df = adata_new_df.loc[:, self.gene_list]
        adata_new_var_df = pd.DataFrame(index=adata_new_df.columns)
        adata_new = ad.AnnData(adata_new_df.values, obs=adata.obs, var=adata_new_var_df)
        adata_new.uns = adata.uns
        return adata_new
    
    def sort_by_genomic_position(self, gtf_file):
        """
        Reorder the genes based on the genomic position of genes
        """
        genomic_df = get_gene_position(gtf_file)
        self.gene_rank = genomic_df.index.tolist()
        # sort the self.genelist based on the gene rank
        self.gene_list.sort(key=lambda x: self.gene_rank.index(x))
        self.gene2idx = {gene: i for i, gene in enumerate(self.gene_list)}
        self.idx2gene = {i: gene for i, gene in enumerate(self.gene_list)}
        self.vocab_size = len(self.gene_rank)
        
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
        

class SinglecellData(Data):
    """
    The dataset class for single-cell data.
    """
    def __init__(self, mdata, gene_vocab):
        super(SinglecellData, self).__init__()
        self.gene_vocab = gene_vocab

        rna_adata = mdata.mod["rna"]
        prot_adata = mdata.mod["protein"]
        # get the same cells in both rna and protein
        common_cells = list(set(rna_adata.obs_names).intersection(
            set(prot_adata.obs_names)))
        rna_adata = rna_adata[common_cells]
        prot_adata = prot_adata[common_cells]
        
        self.protein_idx = prot_adata.var_names
        
        adj_mtx = rna_adata.obsp["connectivities"].toarray()
        edge_index = adj_mtx.nonzero()
        edge_index = np.array(edge_index)
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        concat_data = np.concatenate((rna_adata.X, prot_adata.X), axis=1)
        self.data = Data(x=concat_data, edge_index=edge_index)

    def align_features(self, mdata_new):
        """
        The index of features/genes should be aligned 
        across different datasets.
        """
        new_rna_adata = mdata_new.mod["rna"]
        new_prot_adata = mdata_new.mod["protein"]
        common_rna = set(self.gene_vocab.gene_list).intersection(
                    set(new_rna_adata.var_names))
        new_rna_adata_filtered = new_rna_adata[:, common_rna]
        # padding the rna features which are in the initial gene_vocab but not in the new adata
        new_rna_adata_filtered_pad = self.gene_vocab.align_features(new_rna_adata_filtered)
        

        # padding the protein features
        prot_exp = new_prot_adata[:, self.protein_idx].X
        # padding the rna features
        # align the features

def get_edge_index(rna_adata, device):
    adj_mtx = rna_adata.obsp["connectivities"].toarray()
    edge_index = adj_mtx.nonzero()
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).contiguous().to(device)
    return edge_index

def get_concatenated_data(rna_adata, protein_adata, device):
    if isinstance(rna_adata.X, csr_matrix):
        rna_adata.X = rna_adata.X.toarray()
    if isinstance(protein_adata.X, csr_matrix):
        protein_adata.X = protein_adata.X.toarray()
    concat_data = np.concatenate((rna_adata.X, protein_adata.X), axis=1)
    concat_data = torch.tensor(concat_data, dtype=torch.float32).to(device)
    return concat_data

def create_graphData(rna_adata, protein_adata, spatial_basis, device, alpha=0.5):
    if spatial_basis == "spatial":
        adj_mtx = build_adjacency_matrix_torch(rna_adata.obsm["spatial"], alpha, 
                                                  T=0.005, device=device)
        edge_index = adj_mtx.nonzero().T.contiguous().to(device)
    elif spatial_basis == "expression":   
        edge_index = get_edge_index(rna_adata, device)
    concat_data = get_concatenated_data(rna_adata, protein_adata, device)
    graphData = Data(x=concat_data, edge_index=edge_index)
    return graphData


def create_graphData_mu(mudata, features_use="highly_variable", 
                        device="cpu", spatial_basis="spatial", alpha=0.5):
    """Create graphData from MuData"""
    if features_use == "highly_variable":
        rna_adata = mudata["rna"][:, mudata["rna"].var[f"{features_use}"]]
        protein_adata = mudata["protein"]
        graphData = create_graphData(rna_adata, protein_adata, spatial_basis, device, alpha=alpha)
    else:
        rna_adata = mudata["rna"]
        protein_adata = mudata["protein"]
        graphData = create_graphData(rna_adata, protein_adata, spatial_basis, device, alpha=alpha)
    return graphData, rna_adata.shape[1], protein_adata.shape[1]


def split_data(graphData, num_splits=2, num_val=0.2, num_test=0.2):
    tsf = RandomNodeSplit(num_splits=num_splits, num_val=num_val, num_test=num_test, key=None)
    training_data = tsf(graphData)
    return training_data