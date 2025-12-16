"""
eval_all_checkpoints.py

Evaluate all st-Swin-UNet model checkpoints in a directory on the *test* dataset
and save metrics (loss, acc, prec, rec, f1, csi) for each epoch into a single
CSV file.

Adapted from transformer_cnn_model/eval_all_checkpoints.py for st-Swin-UNet.

Usage example (from repo root)
------------------------------
    conda activate braided

    # Use all defaults from config.py
    python -m swin-unet.eval_all_checkpoints

    # Or override some options from the command line
    python -m swin-unet.eval_all_checkpoints \
        --checkpoint-dir swin-unet/checkpoints \
        --checkpoint-pattern "stswin_tiny_epoch*.pt" \
        --output-csv swin-unet/scores/test_metrics_all_epochs_stswin_tiny.csv

Command-line options
--------------------
    --checkpoint-dir PATH
        Directory containing checkpoint .pt files.
        Default: eval_cfg.checkpoint_dir (from config).

    --checkpoint-pattern GLOB
        Glob pattern to match checkpoint files (e.g. "stswin_tiny_epoch*.pt").
        Default: eval_cfg.checkpoint_pattern (from config).

    --output-csv PATH
        Output CSV file path where metrics for all epochs will be written.
        Default: eval_cfg.scores_csv (from config).

    --cpu
        Force evaluation on CPU even if CUDA is available.
        By default, the first CUDA device ("cuda:0") is used if available.

    --dir-folders PATH
        Root directory of the satellite dataset.
        Default: data_cfg.dir_folders (from config).

    --cache-dir PATH
        Directory where cached tensors are stored/loaded.
        Default: data_cfg.cache_dir (from config).

    --variant STR
        Model variant to use: "tiny" or "small".
        Default: model_cfg.variant (from config).

    --temporal-aggregation STR
        Temporal aggregation method: "concat_proj", "learned_weighted_sum", or "mean".
        Default: model_cfg.temporal_aggregation (from config).
"""

import argparse
import re
from pathlib import Path
import csv
import sys

import torch

# Add swin-unet directory to path
sys.path.insert(0, str(Path(__file__).parent))
from st_swin_unet_model import create_swin_unet_tiny, create_swin_unet_small

# Import shared utilities
from transformer_cnn_model.train_eval_functions.train_eval import validation_unet
from transformer_cnn_model.preprocessing.load_data import build_dataloaders

# Import local config
from config import data_cfg, model_cfg, eval_cfg, train_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all st-Swin-UNet checkpoints on the test set and write metrics to CSV."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(eval_cfg.checkpoint_dir),
        help="Directory containing checkpoint .pt files "
             f"(default: {eval_cfg.checkpoint_dir}).",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default=eval_cfg.checkpoint_pattern,
        help="Glob pattern to match checkpoint files "
             f"(default: {eval_cfg.checkpoint_pattern}).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(eval_cfg.scores_csv),
        help="Path to output CSV file with metrics per epoch.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--dir-folders",
        type=str,
        default=data_cfg.dir_folders,
        help="Root folder of the satellite dataset "
             f"(default: {data_cfg.dir_folders}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(data_cfg.cache_dir),
        help="Directory where cached tensors are stored "
             f"(default: {data_cfg.cache_dir}).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=model_cfg.variant,
        choices=["tiny", "small"],
        help="Model variant: 'tiny' or 'small' "
             f"(default: {model_cfg.variant}).",
    )
    parser.add_argument(
        "--temporal-aggregation",
        type=str,
        default=model_cfg.temporal_aggregation,
        choices=["concat_proj", "learned_weighted_sum", "mean"],
        help="Temporal aggregation method "
             f"(default: {model_cfg.temporal_aggregation}).",
    )
    return parser.parse_args()


def extract_epoch_from_name(path: Path) -> int:
    """
    Extracts the epoch number from filenames like 'stswin_tiny_epoch010.pt'.

    Returns an integer epoch, or -1 if no epoch could be parsed.
    """
    m = re.search(r"epoch(\d+)", path.name)
    if m:
        return int(m.group(1))
    return -1


def create_model(variant: str, temporal_aggregation: str, in_chans: int):
    """
    Factory function to create st-Swin-UNet model based on variant.

    Args:
        variant: "tiny" or "small"
        temporal_aggregation: "concat_proj", "learned_weighted_sum", or "mean"
        in_chans: Number of temporal frames (inferred from data)

    Returns:
        StSwinUnet model
    """
    if variant == "tiny":
        return create_swin_unet_tiny(
            in_chans=in_chans,
            temporal_aggregation=temporal_aggregation
        )
    elif variant == "small":
        return create_swin_unet_small(
            in_chans=in_chans,
            temporal_aggregation=temporal_aggregation
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'tiny' or 'small'.")


def main():
    args = parse_args()

    # -----------------------
    # 1. Find checkpoints
    # -----------------------
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist: {ckpt_dir}")

    checkpoint_paths = sorted(
        ckpt_dir.glob(args.checkpoint_pattern),
        key=extract_epoch_from_name,
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints matching pattern '{args.checkpoint_pattern}' "
            f"found in {ckpt_dir}"
        )

    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print("  -", p.name)

    # -----------------------
    # 2. Device configuration
    # -----------------------
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        print("\nUsing CPU")
        pin_memory = False

    # -----------------------
    # 3. Build DataLoaders once (reuse cache)
    # -----------------------
    print("\nBuilding dataloaders for evaluation (using cache if available)...")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=0,  # evaluation is OK with 0, can increase if desired
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=args.dir_folders,
        device="cpu",  # data lives on CPU; moved to GPU in validation_unet
        use_cache=data_cfg.use_cache,
        cache_dir=cache_dir,
    )

    # Peek at test loader to infer T (time steps)
    x_sample, y_sample = next(iter(test_loader))
    B, T, H, W = x_sample.shape
    print(f"\nSample test batch shape: x={x_sample.shape}, y={y_sample.shape}")
    print(f"Inferred T (time steps / input channels) = {T}")

    # -----------------------
    # 4. Prepare model once (we'll reload weights per checkpoint)
    # -----------------------
    print(f"\nCreating st-Swin-UNet model:")
    print(f"  Variant: {args.variant}")
    print(f"  Temporal aggregation: {args.temporal_aggregation}")
    print(f"  Input channels: {T}")

    model = create_model(
        variant=args.variant,
        temporal_aggregation=args.temporal_aggregation,
        in_chans=T
    )

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # -----------------------
    # 5. Evaluate each checkpoint on test set
    # -----------------------
    rows = []
    print("\nEvaluating checkpoints on test set...")

    for ckpt_path in checkpoint_paths:
        epoch = extract_epoch_from_name(ckpt_path)
        print(f"\n=== Evaluating {ckpt_path.name} (epoch {epoch}) ===")

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        test_losses, acc, prec, rec, f1, csi = validation_unet(
            model,
            test_loader,
            nonwater=train_cfg.nonwater_label,
            water=train_cfg.water_label,
            device=str(device),
            loss_f=train_cfg.loss_f,
            water_threshold=train_cfg.water_threshold,
        )

        mean_test_loss = float(torch.tensor(test_losses).mean())

        print(
            f"Test loss={mean_test_loss:.6f}, "
            f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, "
            f"f1={f1:.4f}, csi={csi:.4f}"
        )

        rows.append({
            "epoch": epoch,
            "checkpoint": ckpt_path.name,
            "test_loss": mean_test_loss,
            "test_acc": float(acc),
            "test_prec": float(prec),
            "test_rec": float(rec),
            "test_f1": float(f1),
            "test_csi": float(csi),
        })

    # Sort again by epoch (just in case)
    rows.sort(key=lambda r: r["epoch"])

    # -----------------------
    # 6. Write CSV
    # -----------------------
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "checkpoint",
        "test_loss",
        "test_acc",
        "test_prec",
        "test_rec",
        "test_f1",
        "test_csi",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved test metrics for {len(rows)} checkpoints to {out_path}")


if __name__ == "__main__":
    main()
