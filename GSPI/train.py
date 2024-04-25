import os
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from muon import MuData
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import distributed as dist
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model import GraphCrossAttn, GraphCrossAttn_spatial_encoding
from dataset import create_graphData

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
        lr: float,
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
        lambda_reg: float=1e-5,
        GAT_encoding: bool=False
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
        self.lr = lr
        self.device = device
        self.mask_ratio = mask_ratio
        self.permute = permute
        self.preserve_rate = preserve_rate
        self.num_splits = num_splits
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.GAT_encoding = GAT_encoding
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
        def MSE_loss(pred, target):
            return torch.sum((pred - target) ** 2)
        loss = MSE_loss(pred[mask], target[mask]) / mask.sum() 
        return loss

    def contrastive_loss(self, embedding, embedding_perm):
        """Define the loss function for contrastive learning"""
        embedding = F.log_softmax(embedding, dim=1)
        embedding_perm = F.softmax(embedding_perm, dim=1)
        def KL_divergence(p, q):
            return torch.sum(p * torch.log(p + 1e-6 / q + 1e-6))
        KL_div = KL_divergence(embedding, embedding_perm).mean()
        return KL_div

    def calculate_loss(self, masked_batch, train_mask):
        """Calculate the loss for the model"""
        results = self.model(
            masked_batch, 
            permute=self.permute, 
            preserve_prob=self.preserve_rate,
            return_attention_weights=self.return_attention_weights
            )
        rna_recon = results["rna_recon"][train_mask]
        prot_recon = results["prot_recon"][train_mask]
        embedding = results["embedding"][train_mask]
        if self.return_attention_weights:
            self.attention_weights = results["attention_weights"]

        loss = self.alpha * self.masked_value_loss(
            rna_recon,
            masked_batch.x[train_mask, :self.rna_input_dim],
            masked_batch.value_mask[train_mask, :self.rna_input_dim]
        ) + (1 - self.alpha) * self.masked_value_loss(
            prot_recon,
            masked_batch.x[train_mask, self.rna_input_dim:],
            masked_batch.value_mask[train_mask, self.rna_input_dim:]
        )
        
        if self.permute:
            embedding_perm = results["embedding_perm"][train_mask]
            loss += (1 - self.alpha - self.beta) * self.contrastive_loss(embedding, embedding_perm)

        return loss

    def _train_step(self, loader, split=0):
        self.model.train()
        total_loss = 0
        for batch in loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device) # input batch data
            masked_batch = self.mask_input(batch, mask_ratio=self.mask_ratio)
            train_mask = batch.train_mask[:, split]
            loss = self.calculate_loss(masked_batch, train_mask)
            # Regularized loss
            reg_loss = self.model.regularization_loss()
            loss = loss + self.lambda_reg * reg_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _val_step(self, loader, split=0):
        self.model.eval()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            masked_batch = self.mask_input(batch, mask_ratio=self.mask_ratio)
            val_mask = batch.val_mask[:, split]
            with torch.no_grad():
                loss = self.calculate_loss(masked_batch, val_mask)
                reg_loss = self.model.regularization_loss()
                loss = loss + self.lambda_reg * reg_loss
            total_loss += loss.item()
        return total_loss
    

    def _inference_step(self, loader, model, optim, alpha=0.4, beta=0.1, split=0):
        """
        Try to infer the spatial coordinates of the cells, given the pre-trained model.
        Input data: 
            1. Randomly mask 50% of the input data.
            2. Use the pre-trained model to infer the spatial coordinates.
            3. Iteratively update the model and using the inferred model to 
                impute the mask value.
            4. Decided to use which imputed value as the final value based on
                the imputation loss. Which means the imputed value with lower
                imputation loss will be used. This step could be done by Gaussian
                Process Regression.
        """
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            masked_batch = self.mask_input(batch, mask_ratio=self.mask_ratio)
            train_mask = batch.train_mask
            optim.zero_grad()
            loss = self.calculate_loss(masked_batch, train_mask)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        return total_loss

    def setup(self, split=None, mode="pre-train"):
        if self.model_choice == "Graph Cross Attention":
            self.model = GraphCrossAttn(
                rna_input_dim=self.rna_input_dim,
                prot_input_dim=self.prot_input_dim,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                heads=self.heads,
                num_blocks=self.num_blocks,
                GAT_encoding=self.GAT_encoding,
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

        if self.num_splits == 1:
            self.data.train_mask = self.data.train_mask.squeeze().unsqueeze(1)
            self.data.val_mask = self.data.val_mask.squeeze().unsqueeze(1)
            # print(self.data.train_mask.shape)

        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask[:, split],
            num_neighbors=[4,3],
            batch_size=self.batch_size,
            replace=False,
            shuffle=False,
        )

        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask[:, split],
            num_neighbors=[4,3],
            batch_size=self.batch_size,
            replace=False,
            shuffle=False,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

    def train(self, plot_loss=False, *args):
        train_history_for_splits = []
        val_history_for_splits = []
        # self.setup(split=0)
        lowest_loss = torch.inf
        for split in range(self.num_splits):
            self.setup(split=split)
            train_losses, val_losses = [], []
            for epoch in range(self.epochs):
                train_loss = self._train_step(
                    self.train_loader, 
                    split=split
                    )
                val_loss = self._val_step(
                    self.val_loader, 
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
        if plot_loss:
            self.plot_losses_curve(train_history_for_splits, val_history_for_splits)

    def fine_tune(
            self,
            model: nn.Module,
            epochs: int=10,
            model_name: str="best_model.pt",
            save_name: str="tmp",
        ):
        """
        Loading the pre-trained best model and fine-tune it for downstream tasks.
        """
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        cpt = torch.load(f"../save_model/{model_name}")
        # load parameters from the pre-trained model
        # for name, _ in model.named_children():
        #     model[name].load_state_dict(cpt["model"][name])
        model.load_state_dict(cpt["model"])
        optim.load_state_dict(cpt["optimizer"])
        model = model.to(self.device)
        self.setup()
        lowest_loss = torch.inf
        loss_history = []
        # fine-tune the model
        for epoch in range(epochs):
            inference_loss = self._inference_step(self.train_loader, model, optim)
            loss_history.append(inference_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {inference_loss}")
            if inference_loss < lowest_loss:
                lowest_loss = inference_loss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()
                }, f"../save_model/best_{save_name}")
        return loss_history            
    
    def get_embedding(
            self,
            graphData: Data
        ):
        """
        Generate the embedding for multi-omics data.
        Params:
            graphData: torch_geometric.data.Data
        """
        self.best_model.eval()
        with torch.no_grad():
            results = self.best_model(graphData)
        return results["embedding"].cpu().numpy()

    def get_attention_weights(
        self,
        graphData: Data
        ):
        """
        Generate the attention weights for attention weights
        Params:
            graphData: torch_geometric.data.Data
        """
        self.bets_model.eval()
        with torch.no_grad():
            results = self.best_model(graphData)
        return results["attention_weights"].cpu().numpy()


    def plot_losses_curve(
            self, 
            train_losses, 
            val_losses, 
            title="Training Losses Curve"
        ):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(train_losses[0], label="train")
        ax.plot(val_losses[0], label="validation")
        ax.title.set_text(f"{title}")
        ax.set_ylabel("Loss", fontsize=20)
        ax.set_xlabel("Epoch", fontsize=20)
        # ax.set_xticks(range(0, len(train_losses[0]), 5))
        # ax.set_yticks(range(0, 100, 10))
        plt.legend(prop={'size': 16, 'weight': 'normal'}, handlelength=3)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.show()


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