import os
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NegativeSampling
from sklearn.neighbors import NearestNeighbors

from model import GraphCrossAttn, GraphCrossAttn_spatial_encoding

class Trainer():
    def __init__(
        self, 
        data: Data, 
        rna_input_dim: int, 
        prot_input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        heads: int,
        num_blocks: int,
        batch_size: int,
        device: torch.device,
        model_choice: str="Graph Cross Attention", 
        spatial_encoder_dim: int=2,
        epochs: int=100,
        mask_ratio: float=0.5,
        permute: bool=False,
        preserve_rate: float=0.5,
        num_splits: int=5,
        alpha: float=0.499,
        beta: float=0.499,
    ):
        self.data = data
        self.model_choice = model_choice
        self.epochs = epochs
        self.rna_input_dim = rna_input_dim
        self.prot_input_dim = prot_input_dim
        self.spatial_encoder_dim = spatial_encoder_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.device = device
        self.mask_ratio = mask_ratio
        self.permute = permute
        self.preserve_rate = preserve_rate
        self.num_splits = num_splits
        self.alpha = alpha
        self.beta = beta
        self.train_losses = []
        self.val_losses = []
        self.best_split = None

    def mask_input(self, batch, mask_ratio):
        """Random mask the input data"""
        batch = batch.to(self.device)
        batch.input = batch.x.clone()
        mask = torch.rand(batch.x.shape).to(self.device) < mask_ratio
        batch.input[mask] = 0
        batch.value_mask = mask
        return batch

    def masked_value_loss(self, pred, target, mask):
        """Define the loss function for masked value prediction"""
        loss = F.mse_loss(pred[mask], target[mask])
        return loss / mask.sum()

    def contrastive_loss(self, embedding, embedding_perm):
        """Define the loss function for contrastive learning"""
        embedding = F.log_softmax(embedding, dim=1)
        embedding_perm = F.softmax(embedding_perm, dim=1)
        def KL_divergence(p, q):
            return torch.sum(p * torch.log(p + 1e-6 / q + 1e-6))
        KL_div = KL_divergence(embedding, embedding_perm).mean()
        return KL_div
        

    def _train_step(self, loader, alpha=0.4, beta=0.1, split=0):
        self.model.train()
        total_loss = 0
        for batch in loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device) # input batch data
            masked_batch = self.mask_input(batch, mask_ratio=self.mask_ratio)
            train_mask = batch.train_mask[:, split]

            if self.permute:
                if self.model_choice == "Spatial Graph Cross Attention":
                    rna_recon, prot_recon, spatial_recon, embedding, embedding_perm = self.model(
                        masked_batch, 
                        permute=True,
                        preserve_prob=self.preserve_rate
                        )
                else:
                    rna_recon, prot_recon, embedding, embedding_perm = self.model(
                        masked_batch, 
                        permute=True,
                        preserve_prob=self.preserve_rate
                        )
                rna_recon = rna_recon[train_mask]
                prot_recon = prot_recon[train_mask]
                embedding = embedding[train_mask]
                embedding_perm = embedding_perm[train_mask]
                if self.model_choice == "Spatial Graph Cross Attention":
                    loss = alpha * self.masked_value_loss(
                        rna_recon,
                        masked_batch.x[train_mask, :self.rna_input_dim],
                        masked_batch.value_mask[train_mask, 
                                                :self.rna_input_dim]
                    ) + beta * self.masked_value_loss(
                        prot_recon,
                        masked_batch.x[train_mask, 
                                        self.rna_input_dim:self.rna_input_dim+
                                        self.prot_input_dim],
                        masked_batch.value_mask[train_mask, 
                                                self.rna_input_dim:self.rna_input_dim+
                                                self.prot_input_dim]
                    ) + (1 - alpha - beta) * self.contrastive_loss(
                        embedding, embedding_perm                    
                    )
                else:
                    loss = alpha * self.masked_value_loss(
                        rna_recon, 
                        masked_batch.x[train_mask, :self.rna_input_dim],
                        masked_batch.value_mask[train_mask, :self.rna_input_dim]
                    ) + beta * self.masked_value_loss(
                        prot_recon, 
                        masked_batch.x[train_mask, self.rna_input_dim:],
                        masked_batch.value_mask[train_mask, self.rna_input_dim:]
                    ) + (1 - alpha - beta) * self.contrastive_loss(embedding, embedding_perm)
            else:
                if self.model_choice == "Spatial Graph Cross Attention":
                    rna_recon, prot_recon, spatial_recon, embedding = self.model(
                        masked_batch, 
                        permute=False,
                        preserve_prob=self.preserve_rate
                        )
                    spatial_recon = spatial_recon[train_mask]
                else:
                    rna_recon, prot_recon, embedding = self.model(
                        masked_batch, 
                        permute=False,
                        preserve_prob=self.preserve_rate
                        )
                rna_recon = rna_recon[train_mask]
                prot_recon = prot_recon[train_mask]
                embedding = embedding[train_mask]
                if self.model_choice == "Spatial Graph Cross Attention":
                    loss = alpha * self.masked_value_loss(
                        rna_recon,
                        masked_batch.x[train_mask, :self.rna_input_dim],
                        masked_batch.value_mask[train_mask, 
                                                :self.rna_input_dim]
                    ) + beta * self.masked_value_loss(
                        prot_recon,
                        masked_batch.x[train_mask, 
                                        self.rna_input_dim:self.rna_input_dim + 
                                        self.prot_input_dim],
                        masked_batch.value_mask[train_mask, 
                                                self.rna_input_dim:self.rna_input_dim +
                                                self.prot_input_dim]
                    ) + (1 - alpha - beta) * self.contrastive_loss(
                        embedding, embedding                    
                    )
                else:
                    loss = alpha * self.masked_value_loss(
                            rna_recon, 
                            masked_batch.x[train_mask, :self.rna_input_dim],
                            masked_batch.value_mask[train_mask, :self.rna_input_dim]
                        ) + (1 - alpha) * self.masked_value_loss(
                            prot_recon, 
                            masked_batch.x[train_mask, self.rna_input_dim:],
                            masked_batch.value_mask[train_mask, self.rna_input_dim:]
                        )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _val_step(self, loader, alpha=0.5, beta=0.4, split=0):
        self.model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            masked_batch = self.mask_input(batch, mask_ratio=self.mask_ratio)
            val_mask = batch.val_mask[:, split]
            with torch.no_grad():
                if self.permute:
                    if self.model_choice == "Spatial Graph Cross Attention":
                        rna_recon, prot_recon, spatial_recon, embedding, embedding_perm = self.model(
                            masked_batch, 
                            permute=self.permute,
                            preserve_prob=self.preserve_rate
                            )
                        spatial_recon = spatial_recon[val_mask]
                    else:
                        rna_recon, prot_recon, embedding, embedding_perm = self.model(
                            masked_batch, 
                            permute=self.permute,
                            preserve_prob=self.preserve_rate
                            )
                    rna_recon = rna_recon[val_mask]
                    prot_recon = prot_recon[val_mask]
                    embedding = embedding[val_mask]
                    embedding_perm = embedding_perm[val_mask]
                    if self.model_choice == "Spatial Graph Cross Attention":
                        loss = alpha * self.masked_value_loss(
                            rna_recon,
                            masked_batch.x[val_mask, :self.rna_input_dim],
                            masked_batch.value_mask[val_mask, 
                                                    :self.rna_input_dim]
                        ) + beta * self.masked_value_loss(
                            prot_recon,
                            masked_batch.x[val_mask, 
                                            self.rna_input_dim:self.rna_input_dim +
                                            self.prot_input_dim],
                            masked_batch.value_mask[val_mask, 
                                                    self.rna_input_dim:self.rna_input_dim+
                                                    self.prot_input_dim]
                        ) + (1 - alpha - beta) * self.contrastive_loss(
                            embedding, embedding_perm                    
                        )
                    else:
                        loss = alpha * self.masked_value_loss(
                            rna_recon, 
                            masked_batch.x[val_mask, :self.rna_input_dim],
                            masked_batch.value_mask[val_mask, :self.rna_input_dim]
                        ) + beta * self.masked_value_loss(
                            prot_recon, 
                            masked_batch.x[val_mask, self.rna_input_dim:],
                            masked_batch.value_mask[val_mask, self.rna_input_dim:]
                        ) + (1 - alpha - beta) * self.contrastive_loss(embedding, embedding_perm)
                else:
                    if self.model_choice == "Spatial Graph Cross Attention":
                        rna_recon, prot_recon, spatial_recon, embedding = self.model(
                            masked_batch, 
                            permute=self.permute,
                            preserve_prob=self.preserve_rate
                            )
                        spatial_recon = spatial_recon[val_mask]
                    else:
                        rna_recon, prot_recon, _ = self.model(
                            masked_batch, permute=self.permute
                            )
                    rna_recon = rna_recon[val_mask]
                    prot_recon = prot_recon[val_mask]
                    if self.model_choice == "Spatial Graph Cross Attention":
                        loss = alpha * self.masked_value_loss(
                            rna_recon,
                            masked_batch.x[val_mask, :self.rna_input_dim],
                            masked_batch.value_mask[val_mask, 
                                                    :self.rna_input_dim]
                        ) + beta * self.masked_value_loss(
                            prot_recon,
                            masked_batch.x[val_mask, 
                                            self.rna_input_dim:self.rna_input_dim + 
                                            self.prot_input_dim],
                            masked_batch.value_mask[val_mask, 
                                                    self.rna_input_dim:self.rna_input_dim +
                                                    self.prot_input_dim]
                        ) + (1 - alpha - beta) * self.contrastive_loss(
                            embedding, embedding                    
                        )
                    else:
                        loss = alpha * self.masked_value_loss(
                            rna_recon, 
                            masked_batch.x[val_mask, :self.rna_input_dim],
                            masked_batch.value_mask[val_mask, :self.rna_input_dim]
                        ) + (1 - alpha) * self.masked_value_loss(
                            prot_recon, 
                            masked_batch.x[val_mask, self.rna_input_dim:],
                            masked_batch.value_mask[val_mask, self.rna_input_dim:]
                        )
            total_loss += loss.item()
        return total_loss

    def setup(self, split=0):
        if self.model_choice == "Graph Cross Attention":
            self.model = GraphCrossAttn(
                rna_input_dim=self.rna_input_dim,
                prot_input_dim=self.prot_input_dim,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                heads=self.heads,
                num_blocks=self.num_blocks,
            ).to(self.device)
        elif self.model_choice == "Spatial Graph Cross Attention":
            self.model = GraphCrossAttn_spatial_encoding(
                spatial_encoder_dim=self.spatial_encoder_dim,
                rna_input_dim=self.rna_input_dim,
                prot_input_dim=self.prot_input_dim,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                heads=self.heads,
                num_blocks=self.num_blocks,
            ).to(self.device)

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask[:, split],
            num_neighbors=[4,3],
            batch_size=128,
            replace=False,
            shuffle=False,
        )

        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask[:, split],
            num_neighbors=[4,3],
            batch_size=128,
            replace=False,
            shuffle=False,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

    def train(self):
        train_history_for_splits = []
        val_history_for_splits = []
        self.setup(split=0)
        lowest_loss = torch.inf
        for split in range(self.num_splits):
            self.setup(split=split)
            train_losses, val_losses = [], []
            for epoch in range(self.epochs):
                train_loss = self._train_step(
                    self.train_loader, 
                    alpha=self.alpha, 
                    beta=self.beta, 
                    split=split
                    )
                val_loss = self._val_step(
                    self.val_loader, 
                    alpha=self.alpha, 
                    beta=self.beta, 
                    split=split
                    )
                self.scheduler.step(val_loss)
                print(f"Epoch {epoch + 1}/{self.epochs} train_loss: {train_loss:.5f} val_loss: {val_loss:.5f}")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                self.best_model = self.model
                self.best_split = split
                
            train_history_for_splits.append(train_losses)
            val_history_for_splits.append(val_losses)
        print(f"Best model saved at split {self.best_split}")
        # save the best model parameters
        save_dict = {
            "model": self.best_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(save_dict, "save_model/best_model.pt")
        return train_history_for_splits, val_history_for_splits

    def fine_tune(
            self, 
            mode: str="inference",
            epochs: int=10,
            model_name: str="best_model.pt", 
        ):
        """
        Loading the pre-trained best model and fine-tune it for downstream tasks.
        """
        model = self.model
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        cpt = torch.load(f"./save_model/{model_name}")
        # load parameters from the pre-trained model
        for name, _ in model.named_children():
            model[name].load_state_dict(cpt["model"][name])
        optim.load_state_dict(cpt["optimizer"])
        model = model.to(self.device)
        self.setup(split=self.best_split)
        lowest_loss = torch.inf
        # fine-tune the model
            

    def ddp_run(rank: int, world_size: int, dataset: None, model: nn.Module):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = '10086'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        data = dataset
        train_index = data.train_mask.nonzero().view(-1)
        train_index = train_index.split(train_index.size(0) // world_size)[rank]

        train_loader = NeighborLoader(
            data,
            input_nodes=train_index,
            num_neighbors=[25, 10],
            batch_size=128,
            num_workers=4,
            shuffle=True,
        )

        if rank == 0:
            val_index = data.val_mask.nonzero().view(-1)
            val_loader = NeighborLoader(
                data, 
                input_nodes=val_index,
                batch_size=128,
                num_workers=4,
                shuffle=False,
            )
        torch.manual_seed(0)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, 11):
            model.train()
            for batch in train_loader:
                batch = batch.to(rank)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = F.cross_entropy(out, batch.x[:batch.batch_size])
                loss.backward()
                optimizer.step()








# model.load_state_dict(cpt["model"])
# optim.load_state_dict(cpt["optimizer"])
# model.rna_embedding.load_state_dict(cpt["model"]["rna_embedding"])
# model.prot_embedding.load_state_dict(cpt["model"]["prot_embedding"])
# model.cross_attn_blocks.load_state_dict(cpt["model"]["cross_attn_blocks"])
# model.cross_attn_agg.load_state_dict(cpt["model"]["cross_attn_agg"])
# model.rna_decoding.load_state_dict(cpt["model"]["rna_decoding"])
# model.prot_decoding.load_state_dict(cpt["model"]["prot_decoding"])
