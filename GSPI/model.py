import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear
from utils import permute_node

class GraphAttnBlock(nn.Module):
    """Graph attention block. Contains two GATConv layers and skip connection."""
    def __init__(self, channel_in, channel_out, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv((-1, -1), channel_in, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin1 = Linear(-1, channel_in * heads)
        self.ln = nn.LayerNorm(channel_in * heads)
        self.conv2 = GATConv((-1, -1), channel_out, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin2 = Linear(-1, channel_out * heads)

    def forward(self, x, edge_index, return_attention_weights=False):
        # An issue indicates that if the return_attention_weights is set to a value,
        # whether is True or False, the attention weights will be returned
        if return_attention_weights == False:
            x = self.conv1(x, edge_index) + self.lin1(x)
            x = self.ln(x)
            x = x.relu()
            x = self.conv2(x, edge_index) + self.lin2(x)
            return x
        else:
            x = self.conv1(x, edge_index, return_attention_weights) + self.lin1(x)
            x = self.ln(x)
            x = x.relu()
            x = self.conv2(x, edge_index, return_attention_weights) + self.lin2(x)
            return x


class GraphCrossAttn(nn.Module):
    def __init__(self, rna_input_dim, prot_input_dim, hidden_dim, embedding_dim, heads=4, num_blocks=2, dropout=0.2):
        super().__init__()
        self.rna_input_dim = rna_input_dim
        self.prot_input_dim = prot_input_dim
        # encoding
        self.rna_embedding = Linear(rna_input_dim, embedding_dim)
        self.prot_embedding = Linear(prot_input_dim, embedding_dim)
        self.cross_attn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.cross_attn_blocks.append(
                    GraphAttnBlock(
                        embedding_dim, 
                        hidden_dim, 
                        heads=heads, 
                        dropout=dropout
                    )
                )
            else:
                self.cross_attn_blocks.append(
                    GraphAttnBlock(
                        hidden_dim, 
                        hidden_dim, 
                        heads=heads, 
                        dropout=dropout
                        )
                    )
        # aggregating the cross attention heads
        self.cross_attn_agg = Linear(-1, hidden_dim)
        # decoding
        self.rna_decoding = Linear(hidden_dim, embedding_dim)
        self.prot_decoding = Linear(hidden_dim, embedding_dim)
        self.rna_recon = Linear(embedding_dim, rna_input_dim)
        self.prot_recon = Linear(embedding_dim, prot_input_dim)

    def regularization_loss(self):
        reg_loss = 0
        reg_loss += torch.norm(self.rna_embedding.weight, 2)
        reg_loss += torch.norm(self.prot_embedding.weight, 2)
        return reg_loss
        
    def forward(self, data, preserve_prob=0.5, permute=False, return_attention_weights=False):
        rna_embedding = self.rna_embedding(data.x[:, :self.rna_input_dim])
        prot_embedding = self.prot_embedding(data.x[:, self.rna_input_dim:])
        x_input = torch.cat([rna_embedding, prot_embedding], dim=1)

        # mask prediction task for the input
        for block in self.cross_attn_blocks:
            x = block(x_input, data.edge_index, return_attention_weights)
        x = self.cross_attn_agg(x)
        embedding = x.relu()

        # randomly shuffle the edges to get contrastive graph
        if permute:
            edge_index_perm = permute_node(data.edge_index, preserve_rate=preserve_prob)
            for block in self.cross_attn_blocks:
                x_perm = block(x_input, edge_index_perm)
            x_perm = self.cross_attn_agg(x_perm)
            embedding_perm = x_perm.relu()
            
        rna_embedding = self.rna_decoding(embedding)
        prot_embedding = self.prot_decoding(embedding)
        rna_recon = self.rna_recon(rna_embedding)
        prot_recon = self.prot_recon(prot_embedding)

        results = {
            "rna_recon": rna_recon,
            "prot_recon": prot_recon,
            "embedding": embedding
        }
        if permute:
            results["embedding_perm"] = embedding_perm

        return results


class GraphCrossAttn_spatial_encoding(GraphCrossAttn):
    def __init__(self, spatial_encoder_dim, rna_input_dim, prot_input_dim, hidden_dim, embedding_dim, heads=4, num_blocks=2, dropout=0.2):
        super().__init__(rna_input_dim, prot_input_dim, hidden_dim, embedding_dim, heads, num_blocks, dropout)
        self.spatial_encoder_dim = spatial_encoder_dim

        # encoding for spatial coordinates
        self.spatial_encoder = Linear(spatial_encoder_dim, embedding_dim)
        self.cross_attn_blocks_spatial = nn.ModuleList()
        
        for i in range(num_blocks):
            if i == 0:
                self.cross_attn_blocks_spatial.append(
                    GraphAttnBlock(
                        embedding_dim, 
                        hidden_dim, 
                        heads=heads, 
                        dropout=dropout
                    )
                )
            else:
                self.cross_attn_blocks_spatial.append(
                    GraphAttnBlock(
                        hidden_dim, 
                        hidden_dim, 
                        heads=heads, 
                        dropout=dropout
                        )
                    )
                
        # aggregating the cross attention heads
        self.cross_attn_agg_spatial = Linear(-1, hidden_dim)

        # decoding for spatial coordinates
        self.spatial_decoding = Linear(hidden_dim, embedding_dim)
        self.spatial_recon = Linear(embedding_dim, spatial_encoder_dim)

    def forward(self, data, preserve_prob=0.5, permute=False):
        rna_embedding = self.rna_embedding(data.x[:, :self.rna_input_dim])
        prot_embedding = self.prot_embedding(
            data.x[:, self.rna_input_dim:self.rna_input_dim + 
                   self.prot_input_dim])
        spatial_embedding = self.spatial_encoder(data.x[:, -self.spatial_encoder_dim:])
        x_input = torch.cat([rna_embedding, prot_embedding, spatial_embedding], dim=1)

        # mask prediction task for the input
        for block in self.cross_attn_blocks:
            x = block(x_input, data.edge_index)
        x = self.cross_attn_agg(x)
        embedding = x.relu()

        # randomly shuffle the edges to get contrastive graph
        if permute:
            edge_index_perm = permute_node(data.edge_index, preserve_rate=preserve_prob)
            for block in self.cross_attn_blocks:
                x_perm = block(x_input, edge_index_perm)
            x_perm = self.cross_attn_agg(x_perm)
            embedding_perm = x_perm.relu()

        rna_embedding = self.rna_decoding(embedding)
        rna_recon = self.rna_recon(rna_embedding)
        prot_embedding = self.prot_decoding(embedding)
        prot_recon = self.prot_recon(prot_embedding)
        spatial_embedding = self.spatial_decoding(embedding)
        spatial_recon = self.spatial_recon(spatial_embedding)
        if permute:
            return rna_recon, prot_recon, spatial_recon, embedding, embedding_perm
        return rna_recon, prot_recon, spatial_recon, embedding
        


class GraRPINet(nn.Module):
    def __init__(self, input_dim_rna, input_dim_prot, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.rna_embedding = Linear(input_dim_rna, hidden_channels)
        self.prot_embedding = Linear(input_dim_prot, hidden_channels)
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels * heads)
        self.conv2 = GATConv((-1, -1), out_channels, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels * heads)

    def forward(self, x, edge_index, input_dim_rna):
        x_rna = self.rna_embedding(x[:, :input_dim_rna])
        x_prot = self.prot_embedding(x[:, input_dim_rna:])
        x = torch.cat([x_rna, x_prot], dim=1)
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

