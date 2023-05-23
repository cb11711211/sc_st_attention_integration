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
                log_attention_weights=False):
        super(GAT, self).__init__()
        self.heads_num = heads_num
        self.out_feature_dim = out_feature_dim
        self.concat = concat
        self.add_skip_conn = add_skip_conn

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
        
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        
        if self.add_skip_conn:
            if out_nodes_features.shape[-1] == in_node_features.shape[-1]:
                out_nodes_features += in_node_features.unsqueeze(1)

    def forward(self, x, adj_mtx):
        # Step 1: linear projection + regularization
        cell_nums = x.shape[0]
        assert adj_mtx == (cell_nums, cell_nums), f"The shape of adj_mtx should be {cell_nums, cell_nums}!"

        in_nodes_feature = self.dropout(x)
        # (N, F_in) * (F_in, H*F_out) -> (N, H, F_out)
        nodes_feature_proj = self.W(in_nodes_feature).reshape(-1, self.heads_num, self.out_feature_dim)
        nodes_feature_proj = self.dropout(nodes_feature_proj)

        # Step 2: compute edge attention
        # apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # (N, H, F_out) * (1, H, F_out) -> (N, H, 1)
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
        # (H, N, N) * (H, N, F_out) -> (N, H, F_out)
        out_nodes_feature = torch.bmm(all_attention_coefficients, nodes_feature_proj.transpose(0, 1))
        out_nodes_feature = out_nodes_feature.permute(1, 0, 2)

        # Step 4: skip connection, concat and bias
        out_nodes_feature = self.skip_concat_bias(all_attention_coefficients, in_nodes_feature, out_nodes_feature)
        return (out_nodes_feature, adj_mtx)
            

