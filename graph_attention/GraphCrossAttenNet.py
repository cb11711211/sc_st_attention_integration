import torch
import torch.nn as nn

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
                 num_layers,
                 num_heads_per_layer,
                 num_features_per_layer,
                 add_skip_connection=True,
                 bias=True,
                 dropout=0.6,
                 log_attention_weights=False):
        super().__init__()
        num_heads_per_layer = [1] + num_heads_per_layer

        cross_attn_layers = []
        for i in range(num_layers):
            layer = CrossAttnLayer(

            )
    

    def forward(self, data):
        return self.gat_net(data)
    

class CrossAttnLayer(nn.Module):
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
        super().__init__()

        self.num_heads = num_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        