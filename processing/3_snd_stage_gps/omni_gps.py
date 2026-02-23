import torch
import torch.nn as nn
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TransformerConv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
import het_graph_builder as hgb
import train_flow as tf
import utils

from loss_metrics import get_loss_fn


class OmniPathPerturbationDataset(Dataset):
    def __init__(
        self, adata_full, adata_split, omnipath_builder, split="train", scale_factor=1.0
    ):
        super().__init__()

        # Filter for the split
        self.adata = adata_split[adata_split.obs["split"] == split].copy()
        self.split = split
        self.scale_factor = scale_factor

        # Get OmniPath graph structure
        self.edge_index, self.edge_type, self.edge_attr = (
            omnipath_builder.build_unified_graph()
        )
        self.gene_to_idx = omnipath_builder.gene_to_idx

        # Get control expression (from full dataset before split)
        control_data = adata_full[adata_full.obs["control"]].copy()
        print(f"[Dataset-{split}] Found {control_data.n_obs} control samples")

        if control_data.n_obs == 0:
            raise ValueError(f"No control samples found!")

        control_mean = control_data.X.mean(axis=0)
        if hasattr(control_mean, "A1"):
            control_mean = control_mean.A1
        elif hasattr(control_mean, "toarray"):
            control_mean = control_mean.toarray().flatten()
        else:
            control_mean = np.asarray(control_mean).flatten()

        self.control_expr = torch.tensor(control_mean, dtype=torch.float)

    def len(self):
        return self.adata.n_obs

    def get(self, idx):
        obs = self.adata.obs.iloc[idx]
        perturbed_genes = obs["perturbed_genes"]

        n_genes = len(self.gene_to_idx)
        is_perturbed = torch.zeros(n_genes, dtype=torch.float)

        for gene in perturbed_genes:
            if gene in self.gene_to_idx:
                is_perturbed[self.gene_to_idx[gene]] = 1.0

        node_features = torch.stack([self.control_expr, is_perturbed], dim=1)

        expr_data = self.adata.X[idx]
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray().flatten()
        else:
            expr_data = np.asarray(expr_data).flatten()

        perturbed_expr = torch.tensor(expr_data, dtype=torch.float)
        delta_expr = (perturbed_expr - self.control_expr) * self.scale_factor

        data = Data(
            x=node_features.float(),
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_attr=self.edge_attr,
            y=delta_expr.float(),
            num_nodes=n_genes,
        )

        return data


class StructuralBias(nn.Module):
    def __init__(self, k_max=5, num_heads=4, num_edge_types=0) -> None:
        super().__init__()

        self.k_max = k_max
        self.num_heads = num_heads

        num_bins = k_max + 2
        self.dist_bias = nn.Embedding(num_bins, num_heads)

        if num_edge_types > 0:
            self.edge_type_bias = nn.Embedding(num_edge_types, num_heads)
        else:
            self.edge_type_bias = None

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for i in range(self.k_max + 2):
                if i == self.k_max + 1:
                    self.dist_bias.weight[i] = -5.0
                else:
                    self.dist_bias.weight[i] = 2.0 - (i * 3.0 / self.k_max)

    def forward(self, dist_matrix, edge_type_matrix=None):
        B = self.dist_bias(dist_matrix).permute(2, 0, 1)
        if self.edge_type_bias is not None and edge_type_matrix is not None:
            B_edge = self.edge_type_bias(edge_type_matrix).permute(2, 0, 1)
            B = B + B_edge
        return B


# (inspirde by : https://papers.miccai.org/miccai-2025/paper/2868_paper.pdf assosciated code can be found at: https://github.com/shb2908/BIIGMA-Net)
class GuidedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

    def forward(self, x, bias_matrix):
        # N = x.size(0) not used anymore, maybe will need later again
        x_seq = x.unsqueeze(1)
        attn_mask = bias_matrix
        out, _ = self.mha(query=x_seq, key=x_seq, value=x_seq, attn_mask=attn_mask)

        return out.squeeze(1)


class OmniPathGuidedGPSLayer(nn.Module):
    def __init__(
        self, hidden_dim, num_edge_types=4, num_heads=8, dropout=0.1, edge_dim=6
    ):
        super().__init__()

        self.edge_type_mpnns = nn.ModuleList(
            [
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    edge_dim=edge_dim,
                )
                for _ in range(num_edge_types)
            ]
        )

        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))

        self.attention = GuidedMultiHeadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        self.norm_mpnn = nn.BatchNorm1d(hidden_dim)
        self.norm_attn = nn.BatchNorm1d(hidden_dim)
        self.norm_ffn = nn.BatchNorm1d(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_type, bias_matrix, batch=None):
        x_mpnn_list = []
        present_types = []

        for type_id in range(len(self.edge_type_mpnns)):
            type_mask = edge_type == type_id
            if type_mask.sum() > 0:
                type_edge_index = edge_index[:, type_mask]
                type_edge_attr = edge_attr[type_mask]
                x_type = self.edge_type_mpnns[type_id](
                    x, type_edge_index, type_edge_attr
                )
                x_mpnn_list.append(x_type)
                present_types.append(type_id)

        if x_mpnn_list:
            x_mpnn = torch.stack(x_mpnn_list)  # [num_present, N, D]
            present_indices = torch.tensor(present_types, device=x.device)
            weights = F.softmax(
                torch.index_select(self.edge_type_weights, 0, present_indices), dim=0
            )
            x_mpnn = (x_mpnn * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            x_mpnn = torch.zeros_like(x)

        x_mpnn = self.dropout(x_mpnn)

        x_attn = self.attention(x, bias_matrix)
        x_attn = self.dropout(x_attn)

        x_ffn = self.ffn(torch.cat([x_mpnn, x_attn], dim=1))
        x = x + x_ffn

        return x


class OmniPathGuidedGPSEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        num_edge_types=4,
        dropout=0.1,
        edge_dim=6,
        k_max=5,
        pe_dim=0,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(input_dim + pe_dim, hidden_dim)
        self.struct_bias = StructuralBias(
            k_max=k_max, num_heads=num_heads, num_edge_types=num_edge_types
        )

        self.layers = nn.ModuleList(
            [
                OmniPathGuidedGPSLayer(
                    hidden_dim=hidden_dim,
                    num_edge_types=num_edge_types,
                    num_heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_type,
        dist_matrix,
        edge_type_matrix=None,
        pe=None,
    ):
        if pe is not None:
            x = torch.cat([x, pe], dim=-1)

        h = self.node_encoder(x)
        h_input = h.clone()
        bias = self.struct_bias(dist_matrix, edge_type_matrix)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_type, bias)

        return h, h_input


class PerturbationExtractor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_input, pert_index):
        h_pert_set = h_input[pert_index]
        if h_pert_set.dim() == 1:
            h_pert_set = h_pert_set.unsqueeze(0) 

        if h_pert_set.size(0) == 1:
            pooled = h_pert_set.squeeze(0)
        else:
            scores = torch.matmul(self.attn_proj(h_pert_set), self.attn_query)
            weights = F.softmax(scores, dim=0)
            pooled = (weights.unsqueeze(-1) * h_pert_set).sum(dim=0)

        return self.mlp(pooled)


class CascadeLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, use_bias=False):
        super().__init__()

        self.use_bias = use_bias
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, Z, H, bias_matrix=None):
        Z_seq = Z.unsqueeze(1)
        H_seq = H.unsqueeze(1)

        attn_out, _ = self.cross_attn(
            query=Z_seq,
            key=H_seq,
            value=H_seq,
            attn_mask=bias_matrix if self.use_bias else None,
        )

        attn_out = attn_out.squeeze(1)

        Z = self.norm1(Z + attn_out)

        Z = self.norm2(Z + self.ffn(Z))

        return Z


class CascadeDecoder(nn.Module):
    def __init__(
        self, hidden_dim, num_layers=3, num_heads=8, dropout=0.1, use_bias=False
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CascadeLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_bias=use_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, H, h_pert, pert_index, bias_matrix=None):
        Z = H.clone()

        Z[pert_index] = Z[pert_index] + h_pert

        for layer in self.layers:
            Z = layer(Z, H, bias_matrix)

        return Z


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, Z, H, h_pert, raw_expr):
        h_pert_broadcast = h_pert.unsqueeze(0).expand(H.size(0), -1)
        raw_expr = raw_expr.unsqueeze(-1)
        combined = torch.cat([Z, H, h_pert_broadcast, raw_expr], dim=-1)

        return self.mlp(combined).squeeze(-1)


class COMPASSModelV2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        encoder_layers=6,
        decoder_layers=3,
        num_heads=8,
        num_edge_type=4,
        dropout=0.1,
        edge_dim=6,
        k_max=5,
        pe_dim=0,
        decoder_use_bias=False,
    ):
        super().__init__()

        self.encoder = OmniPathGuidedGPSEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            num_edge_types=num_edge_type+1,
            dropout=dropout,
            edge_dim=edge_dim,
            k_max=k_max,
            pe_dim=pe_dim,
        )

        self.pert_extractor = PerturbationExtractor(hidden_dim)

        self.decoder = CascadeDecoder(
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=decoder_use_bias,
        )

        self.prediction_head = PredictionHead(hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_type,
        dist_matrix,
        pert_index,
        edge_type_matrix=None,
        pe=None,
        bias_matrix=None,
    ):
        raw_expr = x[:, 0]
        
        H, h_inputs = self.encoder(
            x, edge_index, edge_attr, edge_type, dist_matrix, edge_type_matrix, pe
        )
        h_pert = self.pert_extractor(h_inputs, pert_index)
        Z = self.decoder(H, h_pert, pert_index, bias_matrix)

        delta = self.prediction_head(Z, H, h_pert, raw_expr)
        return delta


###### IMP

# One thing to notice in COMPASSModel.forward: the decoder currently doesn't receive h_pert directly. It only enters at the prediction head. If you want the cascade propagation itself to be conditioned on perturbation identity (which is biologically motivated — the cascade should originate from the perturbed gene), you could inject h_pert into Z(0)Z^{(0)}
# Z(0) by adding it to the perturbed gene's embedding before decoding. That's a natural extension once the base version is working.


####### TRAINING LOGIC #######


def extract_pert_idx(batch):
    pert_indicator = batch.x[:, 1]
    pert_index = torch.where(pert_indicator > 0.5)[0]

    if len(pert_index) == 0:
        # in case no pert
        pert_index = torch.tensor([0], device=batch.x.device)

    return pert_index


def train_epoch(
    model, loader, optimizer, device, g_weights, dist_matrix, edge_type_matrix=None, loss_fn=None
):
    if loss_fn is None:
        loss_fn = tf.weighted_mse

    model.train()

    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)

        pert_index = extract_pert_idx(batch)

        predictions = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_type=batch.edge_type,
            dist_matrix=dist_matrix,
            pert_index=pert_index,
            edge_type_matrix=edge_type_matrix,
        )

        loss = loss_fn(predictions, batch.y, g_weights)

        if torch.isnan(loss):
            print("\n!!!!!!!![ERROR] NaN loss detected!!!!!!!!!")
            print(
                f"  Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}"
            )
            print(f"  Perturbed genes: {pert_index.tolist()}")
            print(f"  Contains NaN in predictions: {torch.isnan(predictions).any()}")
            print("\n!!!!!!!![ERROR] NaN loss detected!!!!!!!!!")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.append(predictions.detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    if len(all_preds) == 0:
        # essentially a null state
        return {
            "loss": float("inf"),
            "mse": float("inf"),
            "mae": float("inf"),
            "correlation": 0.0,
            "direction_acc": 0.0,
        }

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = tf.compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def evaluate_compass(model, loader, device, dist_matrix, edge_type_matrix=None):
    model.eval()
    
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            pert_index = extract_pert_idx(batch)
            
            predictions = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_type=batch.edge_type,
                dist_matrix=dist_matrix,
                pert_index=pert_index,
                edge_type_matrix=edge_type_matrix,
            )
            
            loss = F.mse_loss(predictions, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = tf.compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader.dataset)
    
    return metrics


def train_compass_model(
    train_dataset,
    val_dataset,
    test_dataset,
    num_genes,
    num_edge_types=4,
    device="cuda",
    hidden_dim=256,
    encoder_layers=6,
    decoder_layers=3,
    num_heads=8,
    dropout=0.1,
    batch_size=1,
    num_epochs=200,
    lr=0.001,
    patience=10,
    use_gene_weights=True,
    gene_list=None,
    k_max=5,
    decoder_use_bias=False,
    loss_fn_name="mse"
):
    
    loss_fn = get_loss_fn(loss_fn_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    sample_data = train_dataset.get(0)

    edge_dim = (
        sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 1
    )
    input_dim = sample_data.x.shape[1]

    print(f"Detected edge_dim: {edge_dim}, input_dim: {input_dim}")

    dist_matrix = utils.compute_shortest_path_distances(
        edge_index=sample_data.edge_index,
        num_nodes=num_genes,
        k_max=k_max,
        directed=True,
    ).to(device)
    print(
        f"Distance matrix: {dist_matrix.shape}, "
        f"reachable pairs: {(dist_matrix <= k_max).sum().item()}/{num_genes**2}"
    )

    edge_type_matrix = None
    if num_edge_types > 0:
        edge_type_matrix = torch.zeros(
            num_genes, num_genes, dtype=torch.long, device=device
        )
        ei = sample_data.edge_index.to(device)
        et = sample_data.edge_type.to(device)
        edge_type_matrix[ei[0], ei[1]] = et + 1   

    model = COMPASSModelV2(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        num_edge_type=num_edge_types,
        dropout=dropout,
        edge_dim=edge_dim,
        k_max=k_max,
        decoder_use_bias=decoder_use_bias,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())

    print(
        f"\nModel params: {total_params:,} (encoder: {enc_params:,}, decoder: {dec_params:,})"
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    gene_weights = None
    if use_gene_weights:
        gene_weights = tf.compute_gene_weights(train_dataset).to(device)
        print(f"Gene weights shape: {gene_weights.shape}")

    best_val_loss = float("inf")
    patience_counter = 0
    min_epochs = 100  # Don't allow early stopping before this many epochs

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gene_weights,
            dist_matrix,
            edge_type_matrix,
            loss_fn=loss_fn
        )
        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"MAE: {train_metrics['mae']:.4f}, "
            f"Corr: {train_metrics['correlation']:.3f}, "
            f"Dir Acc: {train_metrics['direction_acc']:.3f}"
        )

        val_metrics = evaluate_compass(model, val_loader, device, dist_matrix, edge_type_matrix)
        print(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}, "
            f"Corr: {val_metrics['correlation']:.3f}, "
            f"Dir Acc: {val_metrics['direction_acc']:.3f}"
        )

        scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_compass_model.pt")
            print("best model saved!!!!....maybe")
        else:
            patience_counter += 1
            # Only check early stopping after minimum epochs
            if patience_counter >= patience and epoch >= min_epochs:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load("best_compass_model.pt"))
    test_metrics = evaluate_compass(model, test_loader, device, dist_matrix, edge_type_matrix)

    utils.plot_test_results(model, test_dataset, gene_list, device, "plots_comp", dist_matrix, edge_type_matrix)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (COMPASS)")
    print("=" * 60)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"Correlation: {test_metrics['correlation']:.3f}")
    print(f"Direction Accuracy: {test_metrics['direction_acc']:.3f}")
    print("=" * 60)

    return model, test_metrics


def grid_search_compass_model(
    train_dataset,
    val_dataset,
    test_dataset,
    num_genes,
    num_edge_types=4,
    device="cuda",
    param_grid=None,
    num_epochs=50,
    patience=10,
    batch_size=1,
    use_gene_weights=True,
    gene_list=None,
    output_dir="plots_comp_search",
):



    if param_grid is None:
        param_grid = {
            "hidden_dim": [128, 256],
            "encoder_layers": [4, 6],
            "decoder_layers": [2, 3],
            "num_heads": [8, 16],
            "dropout": [0.1, 0.2],
            "lr": [0.001, 0.0005],
            "k_max": [3, 5],
            "decoder_use_bias": [False],
            "loss_fn_name": ['mse', 'correlation_mse', 'autofocus_direction', 'top_k_focused']
        }

    from itertools import product
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"\n{'='*70}")
    print(f"COMPASS GRID SEARCH: {len(combinations)} configurations to evaluate")
    print(f"{'='*70}\n")

    results = []
    best_val_loss = float("inf")
    best_config = None
    best_model_path = None

    for i, combo in enumerate(combinations):
        config = dict(zip(param_names, combo))

        print(f"\n{'='*70}")
        print(f"Configuration {i+1}/{len(combinations)}")
        print(f"{'='*70}")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*70}\n")

        try:
            # Extract parameters
            hidden_dim = config.get("hidden_dim", 256)
            encoder_layers = config.get("encoder_layers", 6)
            decoder_layers = config.get("decoder_layers", 3)
            num_heads = config.get("num_heads", 8)
            dropout = config.get("dropout", 0.1)
            lr = config.get("lr", 0.001)
            k_max = config.get("k_max", 5)
            decoder_use_bias = config.get("decoder_use_bias", False)
            loss_fn_name = config.get("loss_fn_name", "mse")

            

            # Train model
            model, test_metrics = train_compass_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                num_genes=num_genes,
                num_edge_types=num_edge_types,
                device=device,
                hidden_dim=hidden_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                num_heads=num_heads,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                patience=patience,
                use_gene_weights=use_gene_weights,
                gene_list=gene_list,
                k_max=k_max,
                decoder_use_bias=decoder_use_bias,
                loss_fn_name=loss_fn_name
            )

            # Load validation metrics from best checkpoint
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            model.load_state_dict(torch.load("best_compass_model.pt"))
            
            # Create dist_matrix and edge_type_matrix for validation
            sample_data = train_dataset.get(0)
            eval_dist_matrix = utils.compute_shortest_path_distances(
                edge_index=sample_data.edge_index,
                num_nodes=num_genes,
                k_max=k_max,
                directed=True,
            ).to(device)
            
            eval_edge_type_matrix = None
            if num_edge_types > 0:
                eval_edge_type_matrix = torch.zeros(
                    num_genes, num_genes, dtype=torch.long, device=device
                )
                ei = sample_data.edge_index.to(device)
                et = sample_data.edge_type.to(device)
                eval_edge_type_matrix[ei[0], ei[1]] = et + 1
            
            val_metrics = evaluate_compass(model, val_loader, device, eval_dist_matrix, eval_edge_type_matrix)

            # Store results
            result = {
                "config_id": i,
                "hidden_dim": hidden_dim,
                "encoder_layers": encoder_layers,
                "decoder_layers": decoder_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "lr": lr,
                "k_max": k_max,
                "decoder_use_bias": decoder_use_bias,
                "loss_fn_name": loss_fn_name,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_correlation": val_metrics["correlation"],
                "val_direction_acc": val_metrics["direction_acc"],
                "test_loss": test_metrics["loss"],
                "test_mae": test_metrics["mae"],
                "test_correlation": test_metrics["correlation"],
                "test_direction_acc": test_metrics["direction_acc"],
                "num_params": sum(p.numel() for p in model.parameters()),
            }
            results.append(result)

            # Track best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_config = config.copy()
                best_model_path = f"best_compass_grid_search_config_{i}.pt"
                torch.save(model.state_dict(), output_path / best_model_path)
                print(f"\n🏆 New best configuration! Val Loss: {best_val_loss:.4f}")

            # Save intermediate results (overwrites same file for progress tracking)
            df_results = pd.DataFrame(results)
            df_results.to_csv(
                output_path / "compass_grid_search_intermediate.csv", index=False
            )

            print(f"\nConfig {i+1} Summary:")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test Correlation: {test_metrics['correlation']:.3f}")

        except Exception as e:
            print(f"\n❌ Error in configuration {i+1}: {str(e)}")
            import traceback

            traceback.print_exc()

            result = {
                "config_id": i,
                "hidden_dim": config.get("hidden_dim", None),
                "encoder_layers": config.get("encoder_layers", None),
                "decoder_layers": config.get("decoder_layers", None),
                "num_heads": config.get("num_heads", None),
                "dropout": config.get("dropout", None),
                "lr": config.get("lr", None),
                "k_max": config.get("k_max", None),
                "decoder_use_bias": config.get("decoder_use_bias", None),
                "error": str(e),
            }
            results.append(result)

    # Final summary
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"compass_grid_search_final_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)

    print(f"\n{'='*70}")
    print(f"COMPASS GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"\nBest Configuration (Val Loss: {best_val_loss:.4f}):")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"\nBest model saved to: {output_path / best_model_path}")
    print(f"{'='*70}\n")

    # Create visualization of results
    _plot_compass_grid_search_results(df_results, output_path)

    return df_results, best_config, best_model_path


def _plot_compass_grid_search_results(df_results, output_dir):
    """Create visualizations of COMPASS grid search results"""

    df_valid = df_results.dropna(subset=["val_loss"])

    if len(df_valid) == 0:
        print("No valid results to plot")
        return

    params = [
        "hidden_dim",
        "encoder_layers",
        "decoder_layers",
        "num_heads",
        "dropout",
        "lr",
        "k_max",
        "loss_fn_name"
    ]
    varied_params = [
        p for p in params if p in df_valid.columns and df_valid[p].nunique() > 1
    ]

    n_plots = len(varied_params) + 1 
    n_rows = (n_plots + 2) // 3 
    n_cols = min(3, n_plots)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle("COMPASS Grid Search Results", fontsize=16, fontweight="bold")

    for idx, param in enumerate(varied_params):
        ax = axes[idx]

        grouped = df_valid.groupby(param)["val_loss"].agg(["mean", "std", "min"])

        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=5,
            linewidth=2,
            markersize=8,
        )
        ax.scatter(
            grouped.index,
            grouped["min"],
            color="red",
            marker="*",
            s=200,
            label="Best",
            zorder=5,
        )

        ax.set_xlabel(param, fontsize=12, fontweight="bold")
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(f"Val Loss vs {param}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axes[len(varied_params)]
    scatter = ax.scatter(
        df_valid["val_loss"],
        df_valid["test_correlation"],
        c=df_valid["num_params"],
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
    )
    ax.set_xlabel("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Correlation", fontsize=12)
    ax.set_title("Test Correlation vs Val Loss", fontsize=13)
    plt.colorbar(scatter, ax=ax, label="# Parameters")
    ax.grid(True, alpha=0.3)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        output_dir / "compass_grid_search_visualization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    df_ranked = df_valid.sort_values("val_loss").head(10)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    table_data.append(
        [
            "Rank",
            "Hidden",
            "Enc Layers",
            "Dec Layers",
            "Heads",
            "Dropout",
            "LR",
            "k_max",
            "Val Loss",
            "Test Corr",
            "Test MAE",
            "Loss Fn Name"
        ]
    )

    for rank, (_, row) in enumerate(df_ranked.iterrows(), 1):
        table_data.append(
            [
                rank,
                int(row["hidden_dim"]),
                int(row["encoder_layers"]),
                int(row["decoder_layers"]),
                int(row["num_heads"]),
                f"{row['dropout']:.2f}",
                f"{row['lr']:.4f}",
                int(row["k_max"]),
                f"{row['val_loss']:.4f}",
                f"{row['test_correlation']:.3f}",
                f"{row['test_mae']:.4f}",
                f"{row["loss_fn_name"]}"
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.07, 0.09, 0.11, 0.11, 0.09, 0.1, 0.1, 0.09, 0.11, 0.11, 0.11, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best row
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor("#90EE90")

    plt.title(
        "Top 10 COMPASS Configurations by Validation Loss",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(
        output_dir / "compass_grid_search_rankings.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")
    print(f"  - compass_grid_search_visualization.png")
    print(f"  - compass_grid_search_rankings.png")
