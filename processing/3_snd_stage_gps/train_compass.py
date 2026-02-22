import anndata as ad
import pandas as pd
import torch
import sys
import os
import json
from itertools import product
from pathlib import Path
from datetime import datetime

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

def main_train_run_omni_search():

    print("\n 1 Loading data")

    adata = ad.read_zarr(
        "/local/26-01_compass-proj/data/pp/norman_pseudobulked_log_norm.zarr"
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
    omnipath_builder = omnipath_builder.load_signaling_network()
    omnipath_builder = omnipath_builder.load_ppi_network()
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
        "hidden_dim": [128, 256],
        "encoder_layers": [6],
        "decoder_layers": [3],
        "num_heads": [8, 16],
        "dropout": [0.1, 0.2],
        "lr": [0.0005],
        "k_max": [5],
        "decoder_use_bias": [False],
        "loss_fn_name": ['autofocus_direction', 'mse', 'correlation_mse', 'top_k_focused']
    }

    num_epochs = 50
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

    valid = df_results.dropna(subset=["val_loss"])
    if not valid.empty:
        best_row = valid.sort_values("val_loss").iloc[0]
        best_config = {k: best_row[k] for k in param_names}
        print(f"\n{'='*60}")
        print(f"GRID SEARCH COMPLETE — best val_loss: {best_row['val_loss']:.4f}")
        print(f"Best config: {best_config}")
    else:
        print("\nGrid search complete — all configurations failed.")

    print(f"Full results saved to: {final_csv}")
    print("\n GPS analysis complete")


if __name__ == "__main__":
    # main_train_run_omni()
    main_train_run_omni_search()
