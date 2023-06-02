# %%
import torch
import torch.nn as nn

class GAT(nn.Module):
    """
    Graph attention network
    """
    def __init__(self, 
                in_feature_dim, 
                out_feature_dim, 
                heads_num, 
                concat=True, 
                activation="LeakeyLU", 
                dropout_prob=0.6, 
                add_skip_conn=True, 
                bias=True, 
                log_attention_weights=False,
                linear_proj=False):
        super(GAT, self).__init__()
        self.heads_num = heads_num
        self.out_feature_dim = out_feature_dim
        self.concat = concat
        self.add_skip_conn = add_skip_conn
        self.linear_proj = linear_proj

        if self.linear_proj:
            self.W = nn.Linear(in_feature_dim, heads_num * out_feature_dim, bias=False)
        else:
            self.W = nn.Parameter(torch.Tensor(heads_num, in_feature_dim, out_feature_dim))

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, heads_num, out_feature_dim))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, heads_num, out_feature_dim))

        if self.linear_proj:
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(heads_num, out_feature_dim, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(heads_num, out_feature_dim, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads_num * out_feature_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_feature_dim))
        else:
            self.register_parameter("bias", None)

        if add_skip_conn:
            self.skip_proj = nn.Linear(in_feature_dim, heads_num * out_feature_dim, bias=False)
        else:
            self.register_parameter("skip_proj", None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        
        ACT_fn_list = ["ReLU", "LeakeyLU", "ELU"]
        if activation == "LeakeyLU":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()

        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()
    
    def skip_concat_bias(self, all_attention_coefficients, in_node_features, out_node_features):
        if self.log_attention_weights:
            self.attention_weights = all_attention_coefficients

        if not out_node_features.is_contiguous():
            out_node_features = out_node_features.contiguous()
        
        if self.add_skip_conn:
            if out_node_features.shape[-1] == in_node_features.shape[-1]:
                out_node_features += in_node_features.unsqueeze(1)
            else:
                out_node_features += self.skip_proj(in_node_features).view(-1, self.heads_num, self.out_feature_dim)
            
        if self.concat:
            # shape: (N, H, F_out) -> (N, H*F_out)
            out_node_features = out_node_features.view(-1, self.heads_num * self.out_feature_dim)
        else:
            # shape: (N, H, F_out) -> (N, F_out)
            out_node_features = out_node_features.mean(dim=1)

        if self.bias is not None:
            out_node_features += self.bias

        return self.activation(out_node_features)

    def init_params(self):
        nn.init.xavier_uniform_(self.W if not self.linear_proj else self.W.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        if self.add_skip_conn and self.skip_proj is not None:
            nn.init.xavier_uniform_(self.skip_proj.weight)
    
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_nodes):
        """
        Softmax over the neighborhoods.
        """
        # calculate the numeraotr
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neighborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_nodes)
        attention_per_edge = exp_scores_per_edge / (neighborhood_aware_denominator + 1e-16)
        # (E, H) -> (E, H, 1)
        return attention_per_edge.unsqueeze(-1)
    
    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape: (N, H)
        size = list(exp_scores_per_edge.shape)
        size[0] = num_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        # position i will contain a sum of exp scores of all edges point to the node i
        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)
        # shape (N, H) -> (E, H)
        return neighborhood_sums.index_select(0, trg_index)

    def aggregate_neighbors(self, nodes_feature_proj_lifted_weighted, edge_index, in_nodes_features, num_nodes):
        """
        Aggregates the embeddings of the neighbors and the center nodes.
        """
        # shape: (E, H, F_out) -> (N, H, F_out), where N is the number of nodes
        size = list(nodes_feature_proj_lifted_weighted.shape)
        size[0] = num_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        # shape (E) -> (E, H, F_out)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_feature_proj_lifted_weighted)
        # shape (E, H, F_out) -> (N, H, F_out)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_feature_proj_lifted_weighted)
        return out_nodes_features

    def explicit_broadcast(self, src, trg):
        # src is shape (E) and trg is shape (E, H, F_out)
        for _ in range(src.dim(), trg.dim()):
            src = src.unsqueeze(-1)
        return src.expand_as(trg)
    
    def lift(self, scores_source, scores_target, nodes_feature_proj, edge_index):
        """
        Lifts the scores to the source and target nodes.
        """
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]

        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)

        nodes_feature_proj_lifted = nodes_feature_proj.index_select(0, src_nodes_index)

        return scores_source, scores_target, nodes_feature_proj_lifted


    def forward(self, x, adj_mtx):
        # Step 1: linear projection + regularization
        cell_nums = x.shape[0]
        assert adj_mtx.shape == (cell_nums, cell_nums), f"The shape of adj_mtx should be {cell_nums, cell_nums}!"

        edge_index = adj_mtx.nonzero(as_tuple=False).t()

        in_nodes_feature = self.dropout(x)
        # (N, F_in) * (F_in, H*F_out) -> (N, H, F_out)
        if self.linear_proj:
            nodes_feature_proj = self.W(in_nodes_feature).reshape(-1, self.heads_num, self.out_feature_dim)
            nodes_feature_proj = self.dropout(nodes_feature_proj)
        else:
            nodes_feature_proj = torch.matmul(in_nodes_feature, self.W)
            nodes_feature_proj = nodes_feature_proj.reshape(-1, self.heads_num, self.out_feature_dim)
            nodes_feature_proj = self.dropout(nodes_feature_proj)

        # Step 2: compute edge attention
        # apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # (N, H, F_out) * (1, H, F_out) -> (N, H, 1) /-> (N, H)
        if self.linear_proj:
            print(nodes_feature_proj.shape, self.scoring_fn_source.shape)
            scores_source = (nodes_feature_proj * self.scoring_fn_source).sum(dim=-1)
            scores_target = (nodes_feature_proj * self.scoring_fn_target).sum(dim=-1)
            # score shape: (E, H), nodes_features_proj_lifted shape: (E, H, F_out), E -> number of edges 
            scores_source_lifted, scores_target_lifted, nodes_feature_proj_lifted = self.lift(scores_source, scores_target, nodes_feature_proj, edge_index)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

            # shape: (E, H, 1)
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_nodes=x.shape[0])
            # add stochasticity to neighborhood aggregation
            attentions_per_edge = self.dropout(attentions_per_edge)
        else:
            scores_source = torch.sum((nodes_feature_proj * self.scoring_fn_source), dim=-1, keepdim=True)
            scores_target = torch.sum((nodes_feature_proj * self.scoring_fn_target), dim=-1, keepdim=True)
            # src shape: (H, N, 1) and trg shape: (H, 1, N)
            scores_source = scores_source.transpose(0, 1)
            scores_target = scores_target.permute(1, 2, 0)

            # (H, N, 1) + (H, 1, N) -> (H, N, N)
            all_scores = self.activation(scores_source + scores_target)
            # shape of all_scores: (H, N, N)
            # shape of adj_mtx: (N, N), set the zero elements of adj_mtx to -9e15
            all_attention_coefficients = self.softmax(all_scores + (-9e15) * (1 - adj_mtx))

        # Step 3: neighborhood aggregation
        if self.linear_proj:
            # shape: (E, H, F_out) * (E, H, 1) -> (E, H, F_out)
            nodes_feature_proj_lifted_weighted = nodes_feature_proj_lifted * attentions_per_edge
            # shape: (N, H, F_out)
            out_node_features = self.aggregate_neighbors(nodes_feature_proj_lifted_weighted, edge_index, in_nodes_feature ,in_nodes_feature.shape[0])

        else:
            # (H, N, N) * (H, N, F_out) -> (N, H, F_out)
            out_node_features = torch.bmm(all_attention_coefficients, nodes_feature_proj.transpose(0, 1))
            out_node_features = out_node_features.permute(1, 0, 2)

        # Step 4: skip connection, concat and bias
        if self.linear_proj:
            out_node_features = self.skip_concat_bias(attentions_per_edge, in_nodes_feature, out_node_features)
        else:
            out_node_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_feature, out_node_features)
        
        return (out_node_features, adj_mtx)
    
    
# given the gene expression matrix and protein expression matrix, we could generate the adj_mtx
# the adj_mtx is the neighbor matrix of the spots
# we could use knn to generate the adj_mtx, e.g. k=3 indicates that each spot has 3 neighbor spots
# And the distance between the spots could be calculated by the euclidean distance in addition to the similarity of the gene expression profile
# The similarity of the gene expression profile could be calculated by the cosine similarity
# %% test the GAT model with the toy data
import numpy as np
import pandas as pd

toy_data = np.random.randint(0, 100, size=(100, 100))
toy_data = pd.DataFrame(toy_data)
toy_data.columns = [f"gene_{i}" for i in range(100)]
toy_data.index = [f"spot_{i}" for i in range(100)]

adj_mtx = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        # if the x and y are close to 1 unit, the spots are neighbor
        if abs(i - j) <= 1:
            adj_mtx[i, j] = 1
# set the diagonal of the adj_mtx to 0
np.fill_diagonal(adj_mtx, 0)

GAT_model = GAT(in_feature_dim=100, out_feature_dim=10, heads_num=10, 
            concat=True, activation="LeakeyLU", dropout_prob=0.6, 
            add_skip_conn=True, bias=True, log_attention_weights=False,
            linear_proj=True)

toy_data = torch.tensor(toy_data.values, dtype=torch.float32)
adj_mtx = torch.tensor(adj_mtx, dtype=torch.float32)
out_nodes_feature, adj_mtx = GAT_model(toy_data, adj_mtx)
# %%
adj_mtx.shape, toy_data.shape, adj_mtx.nonzero(as_tuple=False).t().shape, out_nodes_feature.shape
# %%

# %%

# %%
