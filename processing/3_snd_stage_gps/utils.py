import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import train_flow as tf

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

def plot_test_results(model, test_dataset, gene_names, device, output_dir='plots', dist_matrix=None, edge_type_matrix=None):
    """Generate all plots for test set predictions
    
    Args:
        model: Trained model (GPS or COMPASS)
        test_dataset: Test dataset
        gene_names: List of gene names
        device: Device to run on
        output_dir: Output directory for plots
        dist_matrix: Distance matrix (required for COMPASS models)
        edge_type_matrix: Edge type matrix (optional, for COMPASS models)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print("GENERATING TEST SET PLOTS")
    print(f"{'='*60}")
    
    # Get predictions - use appropriate function based on whether dist_matrix is provided
    if dist_matrix is not None:
        # COMPASS model
        preds, targets, perturbed_genes = tf.get_all_predictions_compass(
            model, test_dataset, device, dist_matrix, edge_type_matrix
        )
    else:
        # Standard GPS model
        preds, targets, perturbed_genes = tf.get_all_predictions(model, test_dataset, device)
    
    # Calculate overall metrics
    metrics = tf.compute_metrics(preds, targets)
    print(f"\nTest Set Metrics:")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  Direction Acc: {metrics['direction_acc']:.4f}")
    
    # 1. Overall correlation plot
    print("\n  1. Overall correlation plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_points = len(preds.flatten())
    if n_points > 50000:
        indices = np.random.choice(n_points, 50000, replace=False)
        preds_plot = preds.flatten()[indices].numpy()
        targets_plot = targets.flatten()[indices].numpy()
    else:
        preds_plot = preds.flatten().numpy()
        targets_plot = targets.flatten().numpy()
    
    hexbin = ax.hexbin(targets_plot, preds_plot, gridsize=50, cmap='viridis', mincnt=1)
    
    min_val = min(targets_plot.min(), preds_plot.min())
    max_val = max(targets_plot.max(), preds_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('True Δ Expression', fontsize=12)
    ax.set_ylabel('Predicted Δ Expression', fontsize=12)
    ax.set_title(f'Test Set: Overall Correlation\n' + 
                 f"Correlation: {metrics['correlation']:.3f}, MAE: {metrics['mae']:.4f}", 
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(hexbin, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_overall_correlation.png', bbox_inches='tight')
    plt.close()
    
    # 2. Per-perturbation metrics
    print("  2. Per-perturbation analysis...")
    n_perturbations = preds.shape[0]
    correlations = []
    maes = []
    pert_labels = []
    
    for i in range(n_perturbations):
        pred = preds[i]
        target = targets[i]
        
        m = tf.compute_metrics(pred.unsqueeze(0), target.unsqueeze(0))
        correlations.append(m['correlation'])
        maes.append(m['mae'])
        
        genes = perturbed_genes[i]
        if len(genes) == 1:
            pert_labels.append(genes[0])
        else:
            pert_labels.append(f"{'+'.join(genes[:2])}" + (f"+{len(genes)-2}more" if len(genes) > 2 else ""))
    
    df_metrics = pd.DataFrame({
        'Perturbation': pert_labels,
        'Correlation': correlations,
        'MAE': maes,
        'N_Genes': [len(g) for g in perturbed_genes]
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(correlations, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(correlations), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(correlations):.3f}')
    axes[0, 0].set_xlabel('Correlation', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Per-Perturbation Correlations', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    axes[0, 1].hist(maes, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(np.mean(maes), color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(maes):.4f}')
    axes[0, 1].set_xlabel('MAE', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Distribution of Per-Perturbation MAE', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    
    single_gene = df_metrics[df_metrics['N_Genes'] == 1]
    multi_gene = df_metrics[df_metrics['N_Genes'] > 1]
    
    axes[1, 0].scatter(single_gene['N_Genes'], single_gene['Correlation'], alpha=0.6, label='Single', s=50)
    if len(multi_gene) > 0:
        axes[1, 0].scatter(multi_gene['N_Genes'], multi_gene['Correlation'], alpha=0.6, 
                           label='Combo', s=50, color='red')
    axes[1, 0].set_xlabel('Number of Perturbed Genes', fontsize=11)
    axes[1, 0].set_ylabel('Correlation', fontsize=11)
    axes[1, 0].set_title('Correlation vs Perturbation Complexity', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    
    worst_10 = df_metrics.nsmallest(10, 'Correlation')
    axes[1, 1].barh(range(len(worst_10)), worst_10['Correlation'].values)
    axes[1, 1].set_yticks(range(len(worst_10)))
    axes[1, 1].set_yticklabels(worst_10['Perturbation'].values, fontsize=9)
    axes[1, 1].set_xlabel('Correlation', fontsize=11)
    axes[1, 1].set_title('Top 10 Worst Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=0.8)
    
    plt.suptitle('Test Set: Per-Perturbation Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'test_per_perturbation_metrics.png', bbox_inches='tight')
    plt.close()
    
    df_metrics.to_csv(output_dir / 'test_perturbation_metrics.csv', index=False)
    
    # 3. Best/worst perturbations detailed view
    print("  3. Best/worst perturbations...")
    n = 5
    best_indices = df_metrics.nlargest(n, 'Correlation').index.tolist()
    worst_indices = df_metrics.nsmallest(n, 'Correlation').index.tolist()
    
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    for i, idx in enumerate(best_indices):
        pred = preds[idx].numpy()
        target = targets[idx].numpy()
        genes = perturbed_genes[idx]
        corr = df_metrics.loc[idx, 'Correlation']
        
        axes[0, i].scatter(target, pred, alpha=0.5, s=10)
        axes[0, i].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', linewidth=1)
        axes[0, i].set_xlabel('True Δ', fontsize=9)
        axes[0, i].set_ylabel('Pred Δ', fontsize=9)
        
        gene_label = '+'.join(genes) if len(genes) <= 2 else f"{genes[0]}+{len(genes)-1}more"
        axes[0, i].set_title(f'{gene_label}\nCorr: {corr:.3f}', fontsize=10)
    
    for i, idx in enumerate(worst_indices):
        pred = preds[idx].numpy()
        target = targets[idx].numpy()
        genes = perturbed_genes[idx]
        corr = df_metrics.loc[idx, 'Correlation']
        
        axes[1, i].scatter(target, pred, alpha=0.5, s=10, color='red')
        axes[1, i].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', linewidth=1)
        axes[1, i].set_xlabel('True Δ', fontsize=9)
        axes[1, i].set_ylabel('Pred Δ', fontsize=9)
        
        gene_label = '+'.join(genes) if len(genes) <= 2 else f"{genes[0]}+{len(genes)-1}more"
        axes[1, i].set_title(f'{gene_label}\nCorr: {corr:.3f}', fontsize=10)
    
    axes[0, 0].text(-0.3, 0.5, 'BEST', transform=axes[0, 0].transAxes, 
                    fontsize=16, fontweight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.3, 0.5, 'WORST', transform=axes[1, 0].transAxes, 
                    fontsize=16, fontweight='bold', rotation=90, va='center')
    
    plt.suptitle('Test Set: Best vs Worst Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_best_worst_perturbations.png', bbox_inches='tight')
    plt.close()
    
    # 4. Gene-level analysis
    print("  4. Gene-level analysis...")
    n_genes = preds.shape[1]
    gene_maes = []
    gene_corrs = []
    gene_mean_abs_delta = []
    
    for gene_idx in range(n_genes):
        pred_gene = preds[:, gene_idx]
        target_gene = targets[:, gene_idx]
        
        mae = (pred_gene - target_gene).abs().mean().item()
        gene_maes.append(mae)
        
        if target_gene.std() > 1e-6:
            corr = np.corrcoef(pred_gene.numpy(), target_gene.numpy())[0, 1]
        else:
            corr = 0.0
        gene_corrs.append(corr)
        
        gene_mean_abs_delta.append(target_gene.abs().mean().item())
    
    df_genes = pd.DataFrame({
        'Gene': gene_names,
        'MAE': gene_maes,
        'Correlation': gene_corrs,
        'Mean_Abs_Delta': gene_mean_abs_delta
    })
    
    df_genes_filtered = df_genes[df_genes['Mean_Abs_Delta'] > 0.05].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df_genes['MAE'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('MAE', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Per-Gene MAE', fontsize=12, fontweight='bold')
    
    axes[0, 1].scatter(df_genes['Mean_Abs_Delta'], df_genes['MAE'], alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Mean |Δ| (True)', fontsize=11)
    axes[0, 1].set_ylabel('MAE', fontsize=11)
    axes[0, 1].set_title('Prediction Error vs Gene Variability', fontsize=12, fontweight='bold')
    
    top_n = 20
    if len(df_genes_filtered) > 0:
        worst_genes = df_genes_filtered.nlargest(min(top_n, len(df_genes_filtered)), 'MAE')
        axes[1, 0].barh(range(len(worst_genes)), worst_genes['MAE'].values, color='red', alpha=0.7)
        axes[1, 0].set_yticks(range(len(worst_genes)))
        axes[1, 0].set_yticklabels(worst_genes['Gene'].values, fontsize=8)
        axes[1, 0].set_xlabel('MAE', fontsize=11)
        axes[1, 0].set_title(f'Top {len(worst_genes)} Genes by Error (|Δ| > 0.05)', fontsize=12, fontweight='bold')
        
        best_genes = df_genes_filtered.nlargest(min(top_n, len(df_genes_filtered)), 'Correlation')
        axes[1, 1].barh(range(len(best_genes)), best_genes['Correlation'].values, color='green', alpha=0.7)
        axes[1, 1].set_yticks(range(len(best_genes)))
        axes[1, 1].set_yticklabels(best_genes['Gene'].values, fontsize=8)
        axes[1, 1].set_xlabel('Correlation', fontsize=11)
        axes[1, 1].set_title(f'Top {len(best_genes)} Genes by Correlation (|Δ| > 0.05)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Test Set: Gene-Level Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_gene_level_analysis.png', bbox_inches='tight')
    plt.close()
    
    df_genes.to_csv(output_dir / 'test_gene_metrics.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    return metrics

def compute_shortest_path_distances(edge_index, num_nodes, k_max=5, directed=True):
    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    adj = csr_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(num_nodes, num_nodes)
    )
    
    dist = shortest_path(adj, directed=directed, unweighted=True)
    dist = np.clip(dist, 0, k_max + 1)
    dist[np.isinf(dist)] = k_max + 1
    
    return torch.tensor(dist, dtype=torch.long)

def full_diagnostic(train_dataset, val_dataset, test_dataset):
    
    def analyze_split(dataset, name):
        all_deltas = []
        for data in dataset:
            all_deltas.append(data.y)
        
        all_deltas = torch.cat(all_deltas)
        
        print(f"\n{'='*60}")
        print(f"{name} SET - Delta Expression Analysis")
        print(f"{'='*60}")
        print(f"# Perturbations: {len(dataset)}")
        print(f"# Genes: {all_deltas.shape[1] if len(all_deltas.shape) > 1 else 'N/A'}")
        print(f"\nOverall Statistics:")
        print(f"  Mean:  {all_deltas.mean():.6f}")
        print(f"  Std:   {all_deltas.std():.6f}")
        print(f"  Min:   {all_deltas.min():.6f}")
        print(f"  Max:   {all_deltas.max():.6f}")
        
        print(f"\nValue Distribution:")
        print(f"  |Δ| < 0.01:  {(all_deltas.abs() < 0.01).float().mean()*100:.1f}%")
        print(f"  |Δ| < 0.05:  {(all_deltas.abs() < 0.05).float().mean()*100:.1f}%")
        print(f"  |Δ| < 0.10:  {(all_deltas.abs() < 0.10).float().mean()*100:.1f}%")
        print(f"  |Δ| > 0.20:  {(all_deltas.abs() > 0.20).float().mean()*100:.1f}%")
        print(f"  |Δ| > 0.50:  {(all_deltas.abs() > 0.50).float().mean()*100:.1f}%")
        print(f"  |Δ| > 1.00:  {(all_deltas.abs() > 1.00).float().mean()*100:.1f}%")
        
        # Per-gene variance
        if len(all_deltas.shape) > 1:
            per_gene_std = all_deltas.std(dim=0)
            print(f"\nPer-Gene Variability:")
            print(f"  # genes with std < 0.01: {(per_gene_std < 0.01).sum()}/{len(per_gene_std)}")
            print(f"  # genes with std < 0.05: {(per_gene_std < 0.05).sum()}/{len(per_gene_std)}")
            print(f"  # genes with std > 0.20: {(per_gene_std > 0.20).sum()}/{len(per_gene_std)}")
            print(f"  Mean gene std: {per_gene_std.mean():.6f}")
            print(f"  Max gene std:  {per_gene_std.max():.6f}")
    
    analyze_split(train_dataset, "TRAIN")
    analyze_split(val_dataset, "VAL")
    analyze_split(test_dataset, "TEST")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print("If most |Δ| < 0.10, your targets are too small for the model to learn magnitude.")
    print("If many genes have std < 0.05, you need per-gene weighting.")
    print(f"{'='*60}\n")

def parse_perturbations(condition):
    if 'ctrl' in condition.lower():
        return []
    
    condition_clean = condition.replace('+ctrl', '') # should be cleared already, but just in case

    genes = [g.strip() for g in condition_clean.split('+') if g.strip()]
    return genes