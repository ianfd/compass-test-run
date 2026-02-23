import anndata as ad
import pandas as pd
import torch
import sys
import os
import json
from itertools import product
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append("/local/26-01_compass-proj/processing/3_snd_stage_gps")

import omni_gps
import het_graph_builder as hgb
import train_flow as tf
import utils


# ---------------------------------------------------------------------------
# Worker function executed in each spawned subprocess (one per GPU)
# ---------------------------------------------------------------------------

def _grid_search_worker(
    rank,
    num_gpus,
    combinations,
    param_names,
    train_dataset,
    val_dataset,
    test_dataset,
    gene_list,
    num_edge_types,
    num_epochs,
    output_dir,
    worker_results_dir,
):
    import torch
    from torch_geometric.loader import DataLoader

    device = f"cuda:{rank}"

    worker_wd = Path(worker_results_dir) / f"worker_{rank}"
    worker_wd.mkdir(exist_ok=True, parents=True)
    orig_wd = os.getcwd()
    os.chdir(worker_wd)

    my_combos = [
        (i, combo)
        for i, combo in enumerate(combinations)
        if i % num_gpus == rank
    ]

    print(
        f"\n[GPU {rank}] Handling {len(my_combos)} / {len(combinations)} "
        f"configurations on {device}"
    )

    results = []

    for i, combo in my_combos:
        config = dict(zip(param_names, combo))

        print(f"\n[GPU {rank}] {'='*60}")
        print(f"[GPU {rank}] Config {i + 1}/{len(combinations)}")
        for k, v in config.items():
            print(f"[GPU {rank}]   {k}: {v}")
        print(f"[GPU {rank}] {'='*60}")

        try:
            model, test_metrics = omni_gps.train_compass_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                num_genes=len(gene_list),
                num_edge_types=num_edge_types,
                device=device,
                hidden_dim=config["hidden_dim"],
                encoder_layers=config["encoder_layers"],
                decoder_layers=config["decoder_layers"],
                num_heads=config["num_heads"],
                dropout=config["dropout"],
                batch_size=1,
                num_epochs=num_epochs,
                lr=config["lr"],
                patience=10,
                gene_list=gene_list,
                k_max=config["k_max"],
                decoder_use_bias=config["decoder_use_bias"],
                loss_fn_name=config["loss_fn_name"],
            )

            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            model.load_state_dict(torch.load("best_compass_model.pt", map_location=device))

            sample_data = train_dataset.get(0)
            k_max = config["k_max"]

            eval_dist_matrix = utils.compute_shortest_path_distances(
                edge_index=sample_data.edge_index,
                num_nodes=len(gene_list),
                k_max=k_max,
                directed=True,
            ).to(device)

            eval_edge_type_matrix = None
            if num_edge_types > 0:
                eval_edge_type_matrix = torch.zeros(
                    len(gene_list), len(gene_list), dtype=torch.long, device=device
                )
                ei = sample_data.edge_index.to(device)
                et = sample_data.edge_type.to(device)
                eval_edge_type_matrix[ei[0], ei[1]] = et + 1

            val_metrics = omni_gps.evaluate_compass(
                model, val_loader, device, eval_dist_matrix, eval_edge_type_matrix
            )

            best_path = Path(output_dir) / f"best_compass_grid_search_config_{i}.pt"
            torch.save(model.state_dict(), best_path)

            result = {
                "config_id": i,
                **config,
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

            print(f"[GPU {rank}] Config {i + 1} — val_loss: {val_metrics['loss']:.4f}, "
                  f"test_corr: {test_metrics['correlation']:.3f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                "config_id": i,
                **{k: config.get(k) for k in param_names},
                "error": str(e),
            }

        results.append(result)

    os.chdir(orig_wd)
    rank_results_file = Path(worker_results_dir) / f"results_rank_{rank}.json"
    with open(rank_results_file, "w") as fh:
        json.dump(results, fh)

    print(f"\n[GPU {rank}] Finished — {len(results)} configs written to {rank_results_file}")


def plot_grid_search_results(df_results, param_names, output_dir):
    """Generate comprehensive plots for grid search results"""
    
    output_dir = Path(output_dir)
    
    # Filter out failed runs
    df_valid = df_results.dropna(subset=["val_loss"]).copy()
    
    if len(df_valid) == 0:
        print("\nNo valid results to plot - all configurations failed.")
        return
    
    print(f"\n{'='*60}")
    print(f"GENERATING GRID SEARCH VISUALIZATION")
    print(f"{'='*60}")
    print(f"Valid configurations: {len(df_valid)} / {len(df_results)}")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Overall metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Val loss distribution
    axes[0, 0].hist(df_valid['val_loss'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df_valid['val_loss'].min(), color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {df_valid["val_loss"].min():.4f}')
    axes[0, 0].set_xlabel('Validation Loss', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Distribution of Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    # Test correlation distribution
    axes[0, 1].hist(df_valid['test_correlation'], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(df_valid['test_correlation'].max(), color='r', linestyle='--', linewidth=2,
                       label=f'Best: {df_valid["test_correlation"].max():.4f}')
    axes[0, 1].set_xlabel('Test Correlation', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Distribution of Test Correlation', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    
    # Test MAE distribution
    axes[1, 0].hist(df_valid['test_mae'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(df_valid['test_mae'].min(), color='r', linestyle='--', linewidth=2,
                       label=f'Best: {df_valid["test_mae"].min():.4f}')
    axes[1, 0].set_xlabel('Test MAE', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Distribution of Test MAE', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    
    # Direction accuracy distribution
    axes[1, 1].hist(df_valid['test_direction_acc'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(df_valid['test_direction_acc'].max(), color='r', linestyle='--', linewidth=2,
                       label=f'Best: {df_valid["test_direction_acc"].max():.4f}')
    axes[1, 1].set_xlabel('Test Direction Accuracy', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Distribution of Test Direction Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    
    plt.suptitle('Grid Search: Overall Metrics Distribution', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_search_metrics_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ Saved metrics distribution plot")
    
    # 2. Hyperparameter impact analysis
    categorical_params = [p for p in param_names if p in ['loss_fn_name', 'decoder_use_bias']]
    numerical_params = [p for p in param_names if p not in categorical_params]
    
    if numerical_params:
        n_params = len(numerical_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, param in enumerate(numerical_params):
            if param in df_valid.columns:
                # Group by parameter value and compute mean test correlation
                param_impact = df_valid.groupby(param).agg({
                    'test_correlation': ['mean', 'std', 'count'],
                    'val_loss': 'mean'
                }).reset_index()
                
                ax = axes[idx]
                ax2 = ax.twinx()
                
                # Plot correlation
                ax.errorbar(param_impact[param], 
                           param_impact[('test_correlation', 'mean')],
                           yerr=param_impact[('test_correlation', 'std')],
                           marker='o', linestyle='-', linewidth=2, markersize=8,
                           color='green', label='Test Correlation', capsize=5)
                
                # Plot val loss
                ax2.plot(param_impact[param],
                        param_impact[('val_loss', 'mean')],
                        marker='s', linestyle='--', linewidth=2, markersize=8,
                        color='steelblue', label='Val Loss')
                
                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11)
                ax.set_ylabel('Test Correlation', fontsize=11, color='green')
                ax2.set_ylabel('Validation Loss', fontsize=11, color='steelblue')
                ax.tick_params(axis='y', labelcolor='green')
                ax2.tick_params(axis='y', labelcolor='steelblue')
                ax.set_title(f'Impact of {param.replace("_", " ").title()}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numerical_params), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Grid Search: Hyperparameter Impact on Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'grid_search_hyperparameter_impact.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("  ✓ Saved hyperparameter impact plot")
    
    # 3. Loss function comparison
    if 'loss_fn_name' in df_valid.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        loss_fn_stats = df_valid.groupby('loss_fn_name').agg({
            'test_correlation': ['mean', 'std'],
            'test_mae': ['mean', 'std'],
            'val_loss': ['mean', 'std']
        }).reset_index()
        
        # Test correlation by loss function
        ax = axes[0]
        x_pos = np.arange(len(loss_fn_stats))
        ax.bar(x_pos, loss_fn_stats[('test_correlation', 'mean')], 
               yerr=loss_fn_stats[('test_correlation', 'std')],
               capsize=5, alpha=0.7, color='green', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_fn_stats['loss_fn_name'], rotation=45, ha='right')
        ax.set_ylabel('Test Correlation', fontsize=11)
        ax.set_title('Test Correlation by Loss Function', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Val loss by loss function
        ax = axes[1]
        ax.bar(x_pos, loss_fn_stats[('val_loss', 'mean')],
               yerr=loss_fn_stats[('val_loss', 'std')],
               capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(loss_fn_stats['loss_fn_name'], rotation=45, ha='right')
        ax.set_ylabel('Validation Loss', fontsize=11)
        ax.set_title('Validation Loss by Loss Function', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Grid Search: Loss Function Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'grid_search_loss_function_comparison.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("  ✓ Saved loss function comparison plot")
    
    # 4. Top configurations comparison
    n_top = min(10, len(df_valid))
    top_configs = df_valid.nsmallest(n_top, 'val_loss')
    
    fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.4)))
    
    y_pos = np.arange(len(top_configs))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n_top))
    
    bars = ax.barh(y_pos, top_configs['test_correlation'], color=colors, edgecolor='black')
    
    # Add config labels
    labels = []
    for _, row in top_configs.iterrows():
        label = f"Config {int(row['config_id'])}: "
        if 'loss_fn_name' in row:
            label += f"{row['loss_fn_name'][:10]}, "
        label += f"h={int(row['hidden_dim'])}, "
        label += f"enc={int(row['encoder_layers'])}, "
        label += f"dec={int(row['decoder_layers'])}"
        labels.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Test Correlation', fontsize=11)
    ax.set_title(f'Top {n_top} Configurations by Validation Loss', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_configs['test_correlation'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_search_top_configurations.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ Saved top configurations plot")
    
    # 5. Metrics correlation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Val loss vs test correlation
    ax = axes[0]
    scatter = ax.scatter(df_valid['val_loss'], df_valid['test_correlation'],
                        c=df_valid['test_mae'], cmap='coolwarm', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Validation Loss', fontsize=11)
    ax.set_ylabel('Test Correlation', fontsize=11)
    ax.set_title('Validation Loss vs Test Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Test MAE')
    
    # Test correlation vs direction accuracy
    ax = axes[1]
    scatter = ax.scatter(df_valid['test_correlation'], df_valid['test_direction_acc'],
                        c=df_valid['val_loss'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Test Correlation', fontsize=11)
    ax.set_ylabel('Test Direction Accuracy', fontsize=11)
    ax.set_title('Test Correlation vs Direction Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Val Loss')
    
    plt.suptitle('Grid Search: Metrics Relationships', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_search_metrics_correlation.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ Saved metrics correlation plot")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")


def main_train_run_omni_search():

    print("\n 1 Loading data")

    adata = ad.read_zarr(
        "/local/26-01_compass-proj/data/pp/norman_pseudobulked_z_scale.zarr"
    )

    print("\n 2/5 Building Pyg graph")

    adata.X = adata.layers["mean"]
    adata.var_names = adata.var["gene_name"].values

    adata.obs["perturbed_genes"] = adata.obs["condition_cleaned"].apply(
        utils.parse_perturbations
    )
    adata.obs["n_perturbed"] = adata.obs["perturbed_genes"].apply(len)
    adata.obs["control"] = adata.obs["n_perturbed"] == 0

    adata_split, split_info = tf.create_train_val_test_split(
        adata, test_strategy="mixed"
    )

    print("\n" + "=" * 60)
    print("BUILDING OMNIPATH NETWORK")
    print("=" * 60)

    gene_list = adata.var_names.tolist()

    omnipath_builder = hgb.OmniPathGRNBuilder(gene_list)

    # Load interaction layers
    print("\nLoading OmniPath interaction layers...")
    #omnipath_builder = omnipath_builder.load_transcriptional_regulation()
    #omnipath_builder = omnipath_builder.load_signaling_network() # Currently disabled to omnipath not avail
    #omnipath_builder = omnipath_builder.load_ppi_network() # Currently disabled to omnipath not avail
    omnipath_builder = omnipath_builder.debug_load_local_go()
    omnipath_builder = omnipath_builder.remove_duplicates()

    # Get statistics
    omnipath_builder.get_statistics()

    # 3. Create datasets
    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    train_dataset = omni_gps.OmniPathPerturbationDataset(
        adata_full=adata,
        adata_split=adata_split,
        omnipath_builder=omnipath_builder,
        split="train",
    )

    val_dataset = omni_gps.OmniPathPerturbationDataset(
        adata_full=adata,
        adata_split=adata_split,
        omnipath_builder=omnipath_builder,
        split="val",
    )

    test_dataset = omni_gps.OmniPathPerturbationDataset(
        adata_full=adata,
        adata_split=adata_split,
        omnipath_builder=omnipath_builder,
        split="test",
    )

    utils.full_diagnostic(train_dataset, val_dataset, test_dataset)

    print("\n 5/5 Training GPS model (multi-GPU grid search)")

    num_edge_types = len(omnipath_builder.edges.keys())

    param_grid = {
        "hidden_dim": [256],
        "encoder_layers": [3],
        "decoder_layers": [4],
        "num_heads": [16],
        "dropout": [0.1],
        "lr": [0.001],
        "k_max": [3],
        "decoder_use_bias": [False],
        "loss_fn_name": ['autofocus_direction']
    }

    num_epochs = 100
    output_dir = Path("plots_compass_v2_search")
    output_dir.mkdir(exist_ok=True, parents=True)
    worker_results_dir = output_dir / "worker_results"
    worker_results_dir.mkdir(exist_ok=True)

    param_names = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    num_gpus = min(torch.cuda.device_count(), 6)
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")

    print(f"\nDispatching {len(combinations)} configurations across {num_gpus} GPU(s)")

    import torch.multiprocessing as mp
    mp.spawn(
        _grid_search_worker,
        args=(
            num_gpus,
            combinations,
            param_names,
            train_dataset,
            val_dataset,
            test_dataset,
            gene_list,
            num_edge_types,
            num_epochs,
            str(output_dir.absolute()),
            str(worker_results_dir.absolute()),
        ),
        nprocs=num_gpus,
        join=True,
    )

    all_results = []
    for rank in range(num_gpus):
        rank_file = worker_results_dir / f"results_rank_{rank}.json"
        with open(rank_file) as fh:
            all_results.extend(json.load(fh))

    df_results = (
        pd.DataFrame(all_results)
        .sort_values("config_id")
        .reset_index(drop=True)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv = output_dir / f"compass_grid_search_final_{timestamp}.csv"
    df_results.to_csv(final_csv, index=False)

    # Generate visualization plots
    plot_grid_search_results(df_results, param_names, output_dir)

    valid = df_results.dropna(subset=["val_loss"])
    if not valid.empty:
        best_row = valid.sort_values("val_loss").iloc[0]
        best_config = {k: best_row[k] for k in param_names}
        print(f"\n{'='*60}")
        print(f"GRID SEARCH COMPLETE — best val_loss: {best_row['val_loss']:.4f}")
        print(f"Best config: {best_config}")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total configurations tested: {len(df_results)}")
        print(f"Successful configurations: {len(valid)}")
        print(f"Failed configurations: {len(df_results) - len(valid)}")
        print(f"\nBest metrics:")
        print(f"  Val Loss:         {valid['val_loss'].min():.4f}")
        print(f"  Test Correlation: {valid['test_correlation'].max():.4f}")
        print(f"  Test MAE:         {valid['test_mae'].min():.4f}")
        print(f"  Test Dir Acc:     {valid['test_direction_acc'].max():.4f}")
        print(f"\nMean metrics:")
        print(f"  Val Loss:         {valid['val_loss'].mean():.4f} ± {valid['val_loss'].std():.4f}")
        print(f"  Test Correlation: {valid['test_correlation'].mean():.4f} ± {valid['test_correlation'].std():.4f}")
        print(f"  Test MAE:         {valid['test_mae'].mean():.4f} ± {valid['test_mae'].std():.4f}")
        print(f"  Test Dir Acc:     {valid['test_direction_acc'].mean():.4f} ± {valid['test_direction_acc'].std():.4f}")
        print(f"{'='*60}")
    else:
        print("\nGrid search complete — all configurations failed.")

    print(f"\nFull results saved to: {final_csv}")
    print("GPS analysis complete")


if __name__ == "__main__":
    # main_train_run_omni()
    main_train_run_omni_search()
