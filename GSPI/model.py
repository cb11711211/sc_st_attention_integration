import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear

class GraphAttnBlock(nn.Module):
    """Graph attention block. Contains two GATConv layers and skip connection."""
    def __init__(self, channel_in, channel_out, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv((-1, -1), channel_in, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin1 = Linear(-1, channel_in * 4)
        self.ln = nn.LayerNorm(channel_in * 4)
        self.conv2 = GATConv((-1, -1), channel_out, heads=heads, dropout=0.2, add_self_loops=False)
        self.lin2 = Linear(-1, channel_out * 4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = self.ln(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
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
        
    def forward(self, data):
        rna_embedding = self.rna_embedding(data.x[:, :self.rna_input_dim])
        prot_embedding = self.prot_embedding(data.x[:, self.rna_input_dim:])
        x_input = torch.cat([rna_embedding, prot_embedding], dim=1)

        raw_x_input = x_input.clone()
        for block in self.cross_attn_blocks:
            x = block(raw_x_input, data.edge_index)
        x = self.cross_attn_agg(x)
        embedding = x.relu()

        # # randomly shuffle the edges to get contrastive graph
        # edge_index_perm = data.edge_index[:, torch.randperm(data.edge_index.shape[1])]
        # for block in self.cross_attn_blocks:
        #     x_perm = block(x_input, edge_index_perm)
        # x_perm = self.cross_attn_agg(x_perm)
        # embedding_perm = x_perm.relu()

        rna_embedding = self.rna_decoding(embedding)
        prot_embedding = self.prot_decoding(embedding)
        rna_recon = self.rna_recon(rna_embedding)
        prot_recon = self.prot_recon(prot_embedding)
        return rna_recon, prot_recon, embedding


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

