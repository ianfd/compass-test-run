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


def create_train_val_test_split(adata, test_strategy='mixed'):
    adata_pert = adata[~adata.obs['control']].copy()
    split_info = {}

    if test_strategy == 'mixed':
        train_idx, temp_idx = train_test_split(
            np.arange(len(adata_pert)),
            test_size=0.3,
            random_state=42
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42
        )

        split_info = {
            'strategy': 'mixed',
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    elif test_strategy == 'unseen_single':
        single_pert = adata_pert[adata_pert.obs['n_perturbed'] == 1].copy()

        unique_genes = list(set([
            genes[0] for genes in single_pert.obs['perturbed_genes']
            if len(genes) == 1
        ]))

        train_genes, temp_genes = train_test_split(
            unique_genes,
            test_size=0.3,
            random_state=32
        )

        val_genes, test_genes = train_test_split(
            temp_genes,
            test_size=0.5,
            random_state=42
        )

        def get_split(genes_list):
            if len(genes_list) == 1:
                gene = genes_list[0]
                if gene in train_genes:
                    return 'train'
                elif gene in val_genes:
                    return 'val'
                elif gene in test_genes:
                    return 'test'
            return None

        single_pert.obs['split'] = single_pert.obs['perturbed_genes'].apply(get_split)
        single_pert = single_pert[single_pert.obs['split'].notna()].copy()

        split_info = {
            'strategy': 'unseen_single',
            'train': np.where(single_pert.obs['split'] == 'train')[0],
            'val': np.where(single_pert.obs['split'] == 'val')[0],
            'test': np.where(single_pert.obs['split'] == 'test')[0],
            'train_genes': train_genes,
            'val_genes': val_genes,
            'test_genes': test_genes
        }

    elif test_strategy == 'combo_split':
        single_pert = adata_pert[adata_pert.obs['n_perturbed'] == 1].copy()
        combo_pert = adata_pert[adata_pert.obs['n_perturbed'] == 2].copy()

        train_idx, val_idx = train_test_split(
            np.arange(len(single_pert)),
            test_size=0.15,
            random_state=32
        )

        split_info = {
            'strategy': 'combo_split',
            'train': train_idx,
            'val': val_idx,
            'test': np.arange(len(combo_pert)),
            'note': 'Train on singles test on combo'
        }

        single_pert.obs['split'] = 'train'
        single_pert.obs.loc[single_pert.obs.index[val_idx], 'split'] = 'val'
        combo_pert.obs['split'] = 'test'
        adata_pert = sc.concat([single_pert, combo_pert])

        split_info['train'] = np.where(adata_pert.obs['split'] == 'train')[0]
        split_info['val'] = np.where(adata_pert.obs['split'] == 'val')[0]
        split_info['test'] = np.where(adata_pert.obs['split'] == 'test')[0]

    adata_pert.obs['split'] = 'train'
    adata_pert.obs.loc[adata_pert.obs.index[split_info['val']], 'split'] = 'val'
    adata_pert.obs.loc[adata_pert.obs.index[split_info['test']], 'split'] = 'test'

    print(f"\n=== {test_strategy.upper()} Split ===")
    print(f"Train: {len(split_info['train'])} perturbations")
    print(f"Val:   {len(split_info['val'])} perturbations")
    print(f"Test:  {len(split_info['test'])} perturbations")

    return adata_pert, split_info

def compute_metrics(predictions, targets, threshold=0.5, verbose=False):
    
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    
    if verbose:
        print(f"\n[Metrics] Prediction stats:")
        print(f"  Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        print(f"  Targets - min: {targets.min():.4f}, max: {targets.max():.4f}, mean: {targets.mean():.4f}, std: {targets.std():.4f}")
        print(f"  Pred abs>0.01: {(predictions.abs() > 0.01).sum()}/{len(predictions.flatten())}")
        print(f"  Target abs>0.01: {(targets.abs() > 0.01).sum()}/{len(targets.flatten())}")
    
    pred_mean = predictions.mean()
    target_mean = targets.mean()
    pred_std = predictions.std()
    target_std = targets.std()

    if pred_std > 0 and target_std > 0:
        correlation = (
            ((predictions - pred_mean) * (targets - target_mean)).mean()
            / (pred_std * target_std) 
        ).item()
    else:
        correlation = 0

    pred_sign = (predictions > threshold).float() - (predictions < -threshold).float()
    target_sign = (targets > threshold).float() - (targets < -threshold).float()
    direction_acc = (pred_sign == target_sign).float().mean().item()

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'direction_acc': direction_acc
    }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        predictions = model(batch)
        loss = F.mse_loss(predictions, batch.y)

        total_loss += loss.item() * batch.num_graphs
        all_preds.append(predictions.cpu())
        all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader.dataset)

    return metrics

def direction_magnitude_loss(prediction, target, lambda_dir=0.5, margin=0.1):
    mse = F.mse_loss(prediction, target)

    prediction_abs = torch.abs(prediction)
    target_abs = torch.abs(target)

    signs_match = (torch.sign(prediction) == torch.sign(target)).float()

    magnitude_ratio = torch.clamp(prediction_abs / (target_abs + 1e-8), 0, 2)
    magnitude_penalty = torch.abs(1.0-magnitude_ratio)

    dir_loss = (1.0 - signs_match) + 0.5 * magnitude_penalty

    return mse + lambda_dir * dir_loss.mean()


def compute_gene_weights(train_dataset):
    all_deltas = []
    for data in train_dataset:
        all_deltas.append(data.y)
    
    all_deltas = torch.stack(all_deltas)

    gene_var = all_deltas.var(dim=0)

    weights = torch.sqrt(gene_var + 1e-8)
    weights = weights / weights.mean()

    weights = torch.clamp(weights, 0.1, 10.0)

    return weights

def weighted_mse(pred, target, weights):
    if weights is None:
        return ((pred - target) ** 2).mean()
    return ((pred - target) ** 2 * weights).mean()

@torch.no_grad()
def get_all_predictions(model, dataset, device):
    """Get predictions for all samples in dataset"""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_perturbed_genes = []
    
    for data in loader:
        data = data.to(device)
        predictions = model(data)
        
        all_preds.append(predictions.cpu())
        all_targets.append(data.y.cpu())
        
        # Get perturbed genes for this sample
        obs = dataset.adata.obs.iloc[len(all_perturbed_genes)]
        all_perturbed_genes.append(obs['perturbed_genes'])
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    
    return all_preds, all_targets, all_perturbed_genes


@torch.no_grad()
def get_all_predictions_compass(model, dataset, device, dist_matrix, edge_type_matrix=None):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_perturbed_genes = []
    
    for data in loader:
        data = data.to(device)
        
        # Extract perturbation index
        pert_indicator = data.x[:, 1]
        pert_index = torch.where(pert_indicator > 0.5)[0]
        if len(pert_index) == 0:
            pert_index = torch.tensor([0], device=device)
        
        predictions = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_type=data.edge_type,
            dist_matrix=dist_matrix,
            pert_index=pert_index,
            edge_type_matrix=edge_type_matrix,
        )
        
        all_preds.append(predictions.cpu())
        all_targets.append(data.y.cpu())
        
        # Get perturbed genes for this sample
        obs = dataset.adata.obs.iloc[len(all_perturbed_genes)]
        all_perturbed_genes.append(obs['perturbed_genes'])
    
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    
    return all_preds, all_targets, all_perturbed_genes