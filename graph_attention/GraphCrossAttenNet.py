import torch
import torch.nn as nn
from GAT import GATLayer

class GraphCrossAttenNet(nn.Module):
    """
    Here, we implement the graph cross attention network.
    We try to use the cross attention mechanism to integrate the multi-omics data.
    The graph cross attention network is composed of two parts:
        1. The graph attention network for each modality.
        2. The cross attention network for the multi-omics data.
    And the network is trained by the multi-task learning, including
        1. The latent space reconstruction loss for each modality, which is the MSE loss.
        2. The cross attention loss for the multi-omics data, which is the MSE loss.
    The latent space reconstruction loss is used to keep the modality-specific information.
    The cross attention loss is used to integrate the multi-omics data.
    And this network applies constrative self-supervised learning to learn the latent space.
    """
    def __init__(self,
                 prot_feature_dim,
                 rna_feature_dim,
                 num_layers,
                 num_heads_per_layer,
                 num_features_per_layer,
                 add_skip_connection=True,
                 bias=True,
                 dropout=0.6,
                 log_attention_weights=False):
        super().__init__()
        num_heads_per_layer = [1] + num_heads_per_layer
        """
        There are two paths in the graph cross attention network.
        The first path is the graph attention network for RNA data, while the 2nd path is for the multi-omics.
        For the first path only consider the co-expression between the RNA data itself. 
        The input of the graph attention layer is the output of the last graph attention layer.
        For the 2nd path, we consider the co-expression between the RNA data and the proteomics data. 
        The input of the graph cross attention layer is the concatenation of 
            the output of the last 1st path attention layer 
            and the output of the last 2nd path attention layer.
        """
        self.prot_feature_dim = prot_feature_dim
        self.rna_feature_dim = rna_feature_dim
        # The graph attention network for RNA data
        self.GAT_linear_proj = nn.Linear(rna_feature_dim, num_features_per_layer[0])
        self.GAT_reproj = nn.Linear(num_features_per_layer[0], rna_feature_dim)
        self.gat_net = nn.ModuleList()
        for i in range(num_layers):
            self.gat_net.append(
                GATLayer(
                    num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                    num_out_features=num_features_per_layer[i + 1],
                    num_heads=num_heads_per_layer[i + 1],
                    concat=True if i < num_layers - 1 else False, # concat=False for the last layer
                    activation=nn.ELU() if i < num_layers - 1 else None, # no nonlinearity for the last layer
                    dropout_prob=dropout, # dropout only for hidden layers
                    add_skip_connection=add_skip_connection,
                    bias=bias,
                    log_attention_weights=log_attention_weights))
        # The graph cross attention network for multi-omics data
        # Here, the input of the graph cross attention layer is the concatenation of
        # the output of the last 1st path attention layer and the output of the last 2nd path attention layer.
        self.cross_attn_linear_proj = nn.Linear(prot_feature_dim + rna_feature_dim, 
            (num_features_per_layer[0] + prot_feature_dim))
        self.cross_attn_reproj = nn.Linear((num_features_per_layer[0] + prot_feature_dim),
            prot_feature_dim + rna_feature_dim)
        self.cross_attn_net = nn.ModuleList()
        
        for i in range(num_layers):
            self.cross_attn_net.append(
                CrossAttnLayer(
                    num_in_features=(num_features_per_layer[i] + prot_feature_dim) * num_heads_per_layer[i],
                    num_out_features=num_features_per_layer[i + 1] + prot_feature_dim,
                    num_heads=num_heads_per_layer[i + 1],
                    concat=True if i < num_layers - 1 else False,
                    activation=nn.ELU() if i < num_layers - 1 else None,
                    dropout_prob=dropout,
                    add_skip_connection=add_skip_connection,
                    bias=bias,
                    log_attention_weights=log_attention_weights))

    def forward(self, data):
        self_layer_attention_weights = []
        cross_layer_attention_weights = []
        # shape: rna_data = (N, F_rna), prot_data = (N, F_prot), adj_mtx = (N, N)
        rna_data, prot_data, edge_index = data
        # shape: concat_data = (N, F_rna + F_prot)
        concat_data = torch.cat((rna_data, prot_data), dim=1)
        for i in range(len(self.gat_net)):
            if i == 0:
                # The input of the first layer of the GAT should be linearly projected to the dim of input features.
                rna_data = self.GAT_linear_proj(rna_data)
                concat_data = self.cross_attn_linear_proj(concat_data)

                # shape of rna_data = (N, F_rna)
                # shape of concat_data = (N, F_rna + F_prot)

                gat_input = (rna_data, edge_index)
                cross_attn_input = (concat_data, edge_index)

                gat_output, _ = self.gat_net[i](gat_input)
                cross_attn_output, _ = self.cross_attn_net[i](cross_attn_input)
            else:
                gat_input = (gat_output, edge_index)
                cross_attn_input_from_rna = gat_output
                cross_attn_input_from_mo = cross_attn_output

                # shape of gat_output = (N, F_rna)
                # shape of cross_attn_output = (N, F_rna + F_prot)

                # The input of teh cross attention network is the average of the two parts
                cross_attn_input = torch.zeros_like(cross_attn_input_from_mo) # shape: (N, F_rna+F_prot)
                
                # Here is very important, we should use the attention from the previous layer to mask the input of the cross attention layer.
                # Get the number of head in this layer

                # before concat, the shape of the cross_attn_input is (N, F_rna+F_prot) ? (N, NH, F_rna+F_prot)
                # after concat, the shape of the cross_attn_input is (N, NH, F_rna+F_prot)

                num_heads = self.cross_attn_net[i-1].num_heads
                cross_attn_input_from_rna = cross_attn_input_from_rna.view(-1, 
                                                num_heads, int(cross_attn_input_from_rna.shape[1] / num_heads))

                # cross_attn_input_from_rna shape: (N, NH, F_rna)
                # cross_attn_input_from_mo shape: (N, NH, F_rna+F_prot)
                # cross_attn_input shape: (N, NH, F_rna+F_prot)
                cross_attn_input[:, :, :cross_attn_input_from_rna.shape[1]] = cross_attn_input_from_rna + cross_attn_input_from_mo[:, :, :cross_attn_input_from_rna.shape[1]]
                cross_attn_input[:, :, cross_attn_input_from_rna.shape[1]:] = cross_attn_input_from_mo[:, :, cross_attn_input_from_rna.shape[1]:]
                cross_attn_input = (cross_attn_input, edge_index)

                gat_output, _ = self.gat_net[i](gat_input)
                cross_attn_output, _ = self.cross_attn_net[i](cross_attn_input)

            self_layer_attention_weights.append(self.gat_net[i].attention_mask)
            cross_layer_attention_weights.append(self.cross_attn_net[i].attention_mask)
        # return (self_layer_attention_weights, cross_layer_attention_weights)
        gat_output = self.GAT_reproj(gat_output)
        cross_attn_output = self.cross_attn_reproj(cross_attn_output)
        return (gat_output, cross_attn_output)


class CrossAttnLayer(GATLayer):
    """
    Here, we implement the cross attention layer.
    The input of the layer contains two parts:
        1. The graph attention network output of the transcriptomics
        2. The graph attention network output of the last cross-attention layer
    And in the same time, the cross-attention layer input also has the skip connection to the previous layer's input.
    To sum up, the input of the cross-attention layer is the concatenation of the above three parts.
    Return the last layer's attention weights.
    """
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
                 log_attention_weights=False):
        
        super().__init__(num_in_features, num_out_features, num_heads, concat, 
                activation, dropout_prob, add_skip_connection, bias, log_attention_weights)
        
    def forward(self, data):
        # The data is the concatenation of the output of the last 1st path self-attention layer and the 2nd path cross-attention layer.
        # But the inital first layer of cross-attention layer is based on the concatenation of the proteomics and transcriptomics data.
        # The cross_attn_data is the concatenation of the output of the last layer of the 1st path and the output of the last layer of the 2nd path.
        cross_attn_data, edge_index = data
        # edge_index = adj_mtx.nonzero().t()
        num_nodes = cross_attn_data.shape[self.nodes_dim]
        
        in_nodes_features = self.dropout(cross_attn_data)
        in_nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)
        in_nodes_features_proj = self.dropout(in_nodes_features_proj)

        # step2: compute edge attention scores
        # shape: (E, H, F_out) * (1, H, F_out) = (E, H, 1)
        scores_src = (in_nodes_features_proj * self.scoring_fn_src).sum(dim=-1)
        scores_trg = (in_nodes_features_proj * self.scoring_fn_trg).sum(dim=-1)

        scores_src_lifted, scores_trg_lifted, nodes_features_proj_lifted = self.lift(scores_src, scores_trg, in_nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_src_lifted + scores_trg_lifted)

        attention_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.src_nodes_dim], num_nodes)
        attention_per_edge = self.dropout(attention_per_edge)

        # step3: Neighborhood aggregation
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attention_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_nodes)

        # step4: residual connection, concat and bias
        out_nodes_features = self.skip_concat_bias(attention_per_edge, in_nodes_features, out_nodes_features)

        self.attention_mask = attention_per_edge

        # reshape the out_nodes_features to the shape of (N, H, F_out)
        out_nodes_features = out_nodes_features.view(-1, self.num_heads, self.num_out_features)

        return (out_nodes_features, edge_index)

