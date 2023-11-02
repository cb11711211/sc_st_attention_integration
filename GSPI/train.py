import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

class Trainer():
    def __init__(
        self, 
        data: Data, 
        model: nn.Module, 
        rna_input_dim: int, 
        prot_input_dim: int,
        device: torch.device,
        epochs: int=100,
    ):
        self.data = data
        self.model = model
        self.epochs = epochs
        self.rna_input_dim = rna_input_dim
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def mask_input(self, batch, mask_ratio):
        """Random mask the input data"""
        batch = batch.to(self.device)
        batch.input = batch.x.clone()
        mask = torch.rand(batch.x.shape) < mask_ratio
        batch.input[mask] = 0
        return batch

    def masked_value_loss(self, pred, target):
        """Define the loss function for masked value prediction"""
        mask = target != 0
        loss = F.mse_loss(pred[mask], target[mask])
        return loss / mask.sum()

    def _train_step(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            rna_recon, prot_recon, embedding = self.model(batch)
            loss = self.masked_value_loss(
                    rna_recon, batch.x[:, :self.rna_input_dim]
                    ) + self.masked_value_loss(
                        prot_recon, batch.x[:, self.rna_input_dim:]
                    )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _val_step(self, loader):
        self.model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            rna_recon, prot_recon, embedding = self.model(batch)
            loss = self.masked_value_loss(
                rna_recon, batch.x[:, :self.rna_input_dim]
                ) + self.masked_value_loss(
                    prot_recon, batch.x[:, self.rna_input_dim:]
                    )
            total_loss += loss.item()
        return total_loss

    def setup(self):
        train_index = self.data.train_mask.nonzero().view(-1)
        val_index = self.data.val_mask.nonzero().view(-1)

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=[2,1],
            batch_size=128,
            replace=False,
            shuffle=False,
        )

        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[2,1],
            batch_size=128,
            replace=False,
            shuffle=False,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            train_loss = self._train_step(self.train_loader)
            val_loss = self._val_step(self.val_loader)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}/{self.epochs} train_loss: {train_loss:.5f} val_loss: {val_loss:.5f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return train_losses, val_losses