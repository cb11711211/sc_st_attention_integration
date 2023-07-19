import torch
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self, 
                 num_layers, 
                 num_heads_per_layer, 
                 num_features_per_layer, 
                 add_skip_connection=True,
                 bias=True, 
                 dropout=0.6, 
                 log_attention_weights=False):
        super().__init__()
        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = []
        for i in range(num_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_heads=num_heads_per_layer[i + 1],
                concat= True if i < num_layers - 1 else False,
                activation=nn.ELU() if i < num_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
                layer_id=-1 if i == num_layers - 1 else i,
            )
            gat_layers.append(layer)
        
        self.gat_net = nn.Sequential(*gat_layers,)

    def forward(self, data):
        return self.gat_net(data)
    

class GATLayer(torch.nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1
    def __init__(self, 
                 num_in_features, 
                 num_out_features, 
                 num_heads, 
                 concat=True, 
                 activation=nn.ELU(),
                 dropout_prob=0.6, 
                 add_skip_connection=True, 
                 bias=True, 
                 log_attention_weights=False,
                 layer_id=0):
        super().__init__()

        self.num_in_features = num_in_features
        self.num_heads = num_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.layer_id = layer_id

        self.linear_proj = nn.Linear(num_in_features, num_heads * num_out_features, bias=False)

        self.scoring_fn_trg = nn.Parameter(torch.Tensor(1, num_heads, num_out_features))
        self.scoring_fn_src = nn.Parameter(torch.Tensor(1, num_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_trg)
        nn.init.xavier_uniform_(self.scoring_fn_src)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        """
        The skip connection, concatenation and addition of the bias term.
        Used only if add_skip_connection=True.
        
        """
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients
        
        # check if the out_nodes_features are contiguous or not,
        # if it is not, it could cause errors in concatenation
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        # shape: (N, NH, F_out) -> (N, NH*F_out)
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]: # if dim(F_in) == dim(F_out)
                # (N, F_in) -> (N, 1, F_in)
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # skip_proj: (N, F_in) -> (N, NH * F_out)
                # view: (N, NH * F_out) -> (N, NH, F_out)
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)

        # the last layer does not need concatenation of multi-heads
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        # add the bias means that each head has its own bias and the bias is not shared
        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
    
    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_nodes):
        """
        Softmax over the neighborhoods.
        scores_per_edge: (E, H)
        trg_index: (E)
        num_nodes: (N)
        """
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neighborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_nodes)

        attention_per_edge = exp_scores_per_edge[trg_index] / (neighborhood_aware_denominator + 1e-16)
        return attention_per_edge.unsqueeze(-1)
    
    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_nodes):
        """
        Sum edge scores for every trg node in a neighborhood-aware fashion.
        exp_scores_per_edge: (E, H)
        trg_index: (E)
        num_nodes: (N)
        """
        # broadcast from (E) to (E, H) so that we can add it with the neighbors' scores
        trg_index_broadcast = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        # shape: (E, H) -> (N, H)
        size = list(exp_scores_per_edge.shape)
        size[self.src_nodes_dim] = num_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        # add scores of all edges e that point to node n
        # (N, H) + （N, H) -> (N, H)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcast, exp_scores_per_edge)
        # shape: (N, H) -> (E, H)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)
    
    def explicit_broadcast(self, x, y):
        """
        Make two tensors x and y of different sizes broadcastable.
        """
        for _ in range(x.dim(), y.dim()):
            x = x.unsqueeze(-1)
        
        return x.expand_as(y)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_nodes):
        """
        Aggregate all neigbors' features and the source node's own features.
        nodes_features_proj_lifted_weighted: (E, H)
        edge_index: (2, E)
        in_nodes_features: (N, F_in)
        num_nodes: (N)
        """
        size = list(nodes_features_proj_lifted_weighted.shape) # (E, H)
        size[self.nodes_dim] = num_nodes # (N, H)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        # explicitly broadcast all source node features to its neighboring target nodes
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features
    
    def lift(self, score_src, score_trg, nodes_features_matrix_proj, edge_index):
        """
        Lifts the edges' features to the nodes' features.
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_src = score_src.index_select(self.nodes_dim, src_nodes_index)
        scores_trg = score_trg.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_src, scores_trg, nodes_features_matrix_proj_lifted
    
    def forward(self, data):
        # step1: Linear Projection + regularization

        in_nodes_features, edge_index = data
        # in_nodes_features = data[0]
        # edge_index = data[1].nonzero(as_tuple=False).t()
        num_nodes = in_nodes_features.shape[0]
        assert edge_index.shape[0] == 2, f"Expected edge index with shape=(2,E) got {edge_index.shape}"

        in_nodes_features = in_nodes_features.view(-1, self.num_in_features)
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # step2: Compute edge attention scores
        scores_src = (nodes_features_proj * self.scoring_fn_src).sum(dim=-1)
        scores_trg = (nodes_features_proj * self.scoring_fn_trg).sum(dim=-1)

        scores_src_lifted, scores_trg_lifted, nodes_features_proj_lifted = self.lift(scores_src, scores_trg, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_src_lifted + scores_trg_lifted)

        attention_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_nodes)
        attention_per_edge = self.dropout(attention_per_edge)

        # step3: Neighborhood aggregation
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attention_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_nodes)
        
        # step4: Residual/skip connections, concat and bias
        out_nodes_features = self.skip_concat_bias(attention_per_edge, in_nodes_features, out_nodes_features)

        self.attention_mask = attention_per_edge
        # out_nodes_features: (N, NH, F_out) or (N, F_out*NH) if concat=False
        if not self.layer_id == -1:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads, self.num_out_features)
        return (out_nodes_features, edge_index)
    
# For the layers in the same GAT block, we share the same attention weights and biases
# Explain each steps in the forward function:
# 1. Linear Projection: project the input features to the same dimension as the output features
# 2. Compute edge attention scores: compute the attention scores for each edge in the graph, the scores are computed based on the source and trg nodes' features
# 3. Neighborhood aggregation: aggregate the neighbors' features to update the trg nodes' features
# 4. Residual/skip connections, concat and bias: add the skip connection, concat the multi-heads' features and add the bias term