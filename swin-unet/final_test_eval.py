"""
final_test_eval.py

Identifies the best checkpoint based on VALIDATION F1 score, then evaluates it
ONCE on the test set to get an unbiased performance estimate.

This follows proper ML methodology:
1. Use validation metrics to select best model
2. Evaluate selected model once on test set
3. Report test performance as final result

Usage example (from repo root)
------------------------------
    # For tiny 4y model
    python -m swin-unet.final_test_eval --variant tiny --temporal-frames 4

    # For small 4y model
    python -m swin-unet.final_test_eval --variant small --temporal-frames 4

    # For tiny 9y model
    python -m swin-unet.final_test_eval --variant tiny --temporal-frames 9

Prerequisites
-------------
You must first run eval_all_checkpoints.py on the VALIDATION set:
    python -m swin-unet.eval_all_checkpoints --variant tiny --temporal-frames 4 --split val
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

# Add swin-unet directory to path
sys.path.insert(0, str(Path(__file__).parent))
from st_swin_unet_model import create_swin_unet_tiny, create_swin_unet_small

# Import shared utilities
from transformer_cnn_model.train_eval_functions.train_eval import validation_unet
from transformer_cnn_model.preprocessing.load_data import build_dataloaders

# Import local config
from config import data_cfg, model_cfg, train_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoint (selected from validation metrics) on test set."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=model_cfg.variant,
        choices=["tiny", "small"],
        help=f"Model variant: 'tiny' or 'small' (default: {model_cfg.variant})",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=data_cfg.temporal_frames,
        choices=[4, 9],
        help=f"Number of input temporal frames: 4 or 9 years (default: {data_cfg.temporal_frames})",
    )
    parser.add_argument(
        "--temporal-aggregation",
        type=str,
        default=model_cfg.temporal_aggregation,
        choices=["concat_proj", "learned_weighted_sum", "mean"],
        help=f"Temporal aggregation method (default: {model_cfg.temporal_aggregation})",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Path to validation metrics CSV (default: auto-generated based on variant)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing checkpoint files (default: auto-generated based on variant)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "csi", "acc", "loss"],
        help="Metric to use for selecting best checkpoint (default: f1)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available",
    )
    return parser.parse_args()


def create_model(variant: str, temporal_aggregation: str, in_chans: int):
    """Create st-Swin-UNet model based on variant."""
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
        raise ValueError(f"Unknown variant: {variant}")


def main():
    args = parse_args()

    # Update data config
    data_cfg.temporal_frames = args.temporal_frames
    data_cfg.year_target = args.temporal_frames + 1

    # Model identifier
    model_id = f"{args.variant}_{args.temporal_frames}y"

    # Set defaults if not specified
    if args.val_csv is None:
        args.val_csv = f"swin-unet/scores/val_metrics_all_epochs_stswin_{model_id}.csv"

    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"swin-unet/checkpoints_{model_id}"

    # -----------------------
    # 1. Load validation metrics and find best checkpoint
    # -----------------------
    val_csv_path = Path(args.val_csv)
    if not val_csv_path.exists():
        raise FileNotFoundError(
            f"Validation metrics CSV not found: {val_csv_path}\n"
            f"Please run: python -m swin-unet.eval_all_checkpoints "
            f"--variant {args.variant} --temporal-frames {args.temporal_frames} --split val"
        )

    print(f"Loading validation metrics from: {val_csv_path}")
    df = pd.read_csv(val_csv_path)

    # Find best epoch based on selected metric
    if args.metric == "loss":
        # Lower is better for loss
        best_idx = df["loss"].idxmin()
        best_value = df.loc[best_idx, "loss"]
        comparison = "lowest"
    else:
        # Higher is better for f1, csi, acc
        best_idx = df[args.metric].idxmax()
        best_value = df.loc[best_idx, args.metric]
        comparison = "highest"

    best_epoch = df.loc[best_idx, "epoch"]
    best_checkpoint_name = df.loc[best_idx, "checkpoint"]

    print(f"\n{'='*60}")
    print(f"Best checkpoint selected based on VALIDATION {args.metric.upper()}")
    print(f"{'='*60}")
    print(f"Epoch: {best_epoch}")
    print(f"Checkpoint: {best_checkpoint_name}")
    print(f"Validation {args.metric.upper()}: {best_value:.6f} ({comparison} among {len(df)} checkpoints)")
    print(f"{'='*60}\n")

    # -----------------------
    # 2. Load best checkpoint
    # -----------------------
    checkpoint_path = Path(args.checkpoint_dir) / best_checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    # Device configuration
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        pin_memory = False

    # Build dataloaders
    print("\nBuilding dataloaders (using cache)...")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # Infer T from data
    x_sample, _ = next(iter(test_loader))
    _, T, _, _ = x_sample.shape
    print(f"Detected T (temporal frames) = {T}")

    # Create model
    print(f"\nCreating model:")
    print(f"  Variant: {args.variant}")
    print(f"  Temporal aggregation: {args.temporal_aggregation}")
    print(f"  Input channels: {T}")

    model = create_model(
        variant=args.variant,
        temporal_aggregation=args.temporal_aggregation,
        in_chans=T
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # -----------------------
    # 3. Evaluate ONCE on test set
    # -----------------------
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION (unbiased estimate)")
    print(f"{'='*60}\n")

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

    print(f"Model: st-Swin-UNet ({args.variant}_{args.temporal_frames}y)")
    print(f"Best checkpoint: epoch {best_epoch} (selected by validation {args.metric})")
    print(f"\nTest set results:")
    print(f"  Loss:      {mean_test_loss:.6f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  CSI/IoU:   {csi:.4f}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")

    # Save results to file
    results_dir = Path("swin-unet/scores")
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / f"final_test_results_stswin_{model_id}.txt"

    with open(results_file, "w") as f:
        f.write(f"st-Swin-UNet Final Test Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model variant: {args.variant}\n")
        f.write(f"Temporal frames: {args.temporal_frames}\n")
        f.write(f"Temporal aggregation: {args.temporal_aggregation}\n")
        f.write(f"\nBest checkpoint: epoch {best_epoch}\n")
        f.write(f"Selected by: validation {args.metric} = {best_value:.6f}\n")
        f.write(f"\nTest set performance (unbiased estimate):\n")
        f.write(f"  Loss:      {mean_test_loss:.6f}\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n")
        f.write(f"  CSI/IoU:   {csi:.4f}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
