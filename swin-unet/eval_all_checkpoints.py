"""
eval_all_checkpoints.py

Evaluate all st-Swin-UNet model checkpoints in a directory on the validation or test dataset
and save metrics (loss, acc, prec, rec, f1, csi) for each epoch into a single CSV file.

IMPORTANT: For proper methodology, evaluate on VALIDATION set to select best checkpoint,
then evaluate that checkpoint ONCE on test set using final_test_eval.py

Adapted from transformer_cnn_model/eval_all_checkpoints.py for st-Swin-UNet.

Usage example (from repo root)
------------------------------
    conda activate braided

    # Evaluate on VALIDATION set (recommended for model selection)
    python -m swin-unet.eval_all_checkpoints --split val

    # Or evaluate on test set (only for final reporting)
    python -m swin-unet.eval_all_checkpoints --split test

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
        default=None,
        help="Directory containing checkpoint .pt files "
             "(default: auto-generated based on variant).",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default=None,
        help="Glob pattern to match checkpoint files "
             "(default: auto-generated based on variant).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file with metrics per epoch "
             "(default: auto-generated based on variant).",
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
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=data_cfg.temporal_frames,
        choices=[4, 9],
        help=f"Number of input temporal frames: 4 or 9 years (default: {data_cfg.temporal_frames})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on: 'val' for validation (recommended for model selection), "
             "'test' for final evaluation (default: val)",
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

    # Update data config with temporal frames
    data_cfg.temporal_frames = args.temporal_frames
    data_cfg.year_target = args.temporal_frames + 1

    # Update eval config for the specified variant and temporal frames
    eval_cfg.update_for_variant(args.variant, args.temporal_frames)

    # Use variant-specific defaults if not specified
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(eval_cfg.checkpoint_dir)
    if args.checkpoint_pattern is None:
        args.checkpoint_pattern = eval_cfg.checkpoint_pattern
    if args.output_csv is None:
        # Auto-generate CSV name based on split
        model_id = f"{args.variant}_{args.temporal_frames}y"
        scores_dir = Path("swin-unet/scores")
        args.output_csv = str(scores_dir / f"{args.split}_metrics_all_epochs_stswin_{model_id}.csv")

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

    # Select loader based on split
    if args.split == "val":
        eval_loader = val_loader
        split_name = "validation"
    else:
        eval_loader = test_loader
        split_name = "test"

    # Peek at loader to infer T (time steps)
    x_sample, y_sample = next(iter(eval_loader))
    B, T, H, W = x_sample.shape
    print(f"\nSample {split_name} batch shape: x={x_sample.shape}, y={y_sample.shape}")
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
    # 5. Evaluate each checkpoint on selected split
    # -----------------------
    rows = []
    print(f"\nEvaluating checkpoints on {split_name} set...")

    for ckpt_path in checkpoint_paths:
        epoch = extract_epoch_from_name(ckpt_path)
        print(f"\n=== Evaluating {ckpt_path.name} (epoch {epoch}) ===")

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        losses, acc, prec, rec, f1, csi = validation_unet(
            model,
            eval_loader,
            nonwater=train_cfg.nonwater_label,
            water=train_cfg.water_label,
            device=str(device),
            loss_f=train_cfg.loss_f,
            water_threshold=train_cfg.water_threshold,
        )

        mean_loss = float(torch.tensor(losses).mean())

        print(
            f"{split_name.capitalize()} loss={mean_loss:.6f}, "
            f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, "
            f"f1={f1:.4f}, csi={csi:.4f}"
        )

        rows.append({
            "epoch": epoch,
            "checkpoint": ckpt_path.name,
            "loss": mean_loss,
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "csi": float(csi),
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
        "loss",
        "acc",
        "prec",
        "rec",
        "f1",
        "csi",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {split_name} metrics for {len(rows)} checkpoints to {out_path}")


if __name__ == "__main__":
    main()
