"""
visualize_predictions.py

Visualize model predictions on test samples by showing:
  - Input temporal sequence (first, middle, and last frames)
  - Ground truth target
  - Model prediction
  - Difference/error map

Usage example (from repo root)
------------------------------
    conda activate braided

    # Use best checkpoint from training
    python -m swin-unet.visualize_predictions --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt

    # Specify number of samples to visualize
    python -m swin-unet.visualize_predictions --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt --num-samples 10

    # Use a specific split (train/val/test)
    python -m swin-unet.visualize_predictions --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt --split test
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add swin-unet directory to path
sys.path.insert(0, str(Path(__file__).parent))
from st_swin_unet_model import create_swin_unet_tiny, create_swin_unet_small

# Import shared utilities
from transformer_cnn_model.preprocessing.load_data import build_dataloaders

# Import local config
from config import data_cfg, model_cfg, train_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize st-Swin-UNet predictions on test samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to visualize (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(data_cfg.plots_dir),
        help="Directory to save visualizations (default: from config)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=model_cfg.variant,
        choices=["tiny", "small"],
        help="Model variant (default: from config)",
    )
    parser.add_argument(
        "--temporal-aggregation",
        type=str,
        default=model_cfg.temporal_aggregation,
        choices=["concat_proj", "learned_weighted_sum", "mean"],
        help="Temporal aggregation method (default: from config)",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=data_cfg.temporal_frames,
        choices=[4, 9],
        help=f"Number of input temporal frames: 4 or 9 years (default: {data_cfg.temporal_frames})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for binary classification (default: 0.5)",
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


def visualize_prediction(input_seq, target, prediction, threshold=0.5,
                         sample_idx=0, save_path=None, variant=None):
    """
    Create a comprehensive visualization of model prediction.

    Args:
        input_seq: (T, H, W) temporal input sequence
        target: (H, W) ground truth
        prediction: (H, W) model prediction (probabilities)
        threshold: classification threshold
        sample_idx: sample number for title
        save_path: path to save figure
        variant: model variant name for title (e.g., "tiny" or "small")
    """
    T, H, W = input_seq.shape

    # Select frames to show: first, middle, last
    frame_indices = [0, T // 2, T - 1]

    # Create figure with 2 rows
    # Row 1: Input frames
    # Row 2: Target, Prediction, Difference
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Show selected input frames
    for col, frame_idx in enumerate(frame_indices):
        ax = axes[0, col]
        im = ax.imshow(input_seq[frame_idx], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Input Frame {frame_idx + 1}/{T}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 1, Col 4: Show temporal mean
    ax = axes[0, 3]
    temporal_mean = input_seq.mean(axis=0)
    im = ax.imshow(temporal_mean, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Temporal Mean\n({T} frames)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2, Col 0: Ground truth
    ax = axes[1, 0]
    im = ax.imshow(target, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2, Col 1: Prediction (probabilities)
    ax = axes[1, 1]
    im = ax.imshow(prediction, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Prediction\n(probabilities)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2, Col 2: Binary prediction
    ax = axes[1, 2]
    binary_pred = (prediction > threshold).astype(float)
    im = ax.imshow(binary_pred, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Binary Prediction\n(threshold={threshold})', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2, Col 3: Error map
    ax = axes[1, 3]
    error = np.abs(target - binary_pred)
    im = ax.imshow(error, cmap='Reds', vmin=0, vmax=1)
    error_pct = (error.sum() / error.size) * 100
    ax.set_title(f'Error Map\n({error_pct:.2f}% error)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overall title
    variant_str = f" ({variant})" if variant else ""
    fig.suptitle(f'Sample {sample_idx + 1} - st-Swin-UNet{variant_str} Prediction Visualization',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    return fig


def compute_metrics(target, prediction, threshold=0.5):
    """Compute metrics for a single prediction."""
    binary_pred = (prediction > threshold).astype(float)

    # Convert to boolean for sklearn-like computation
    target_bool = target.astype(bool)
    pred_bool = binary_pred.astype(bool)

    # True positives, false positives, false negatives
    tp = np.logical_and(target_bool, pred_bool).sum()
    fp = np.logical_and(~target_bool, pred_bool).sum()
    fn = np.logical_and(target_bool, ~pred_bool).sum()
    tn = np.logical_and(~target_bool, ~pred_bool).sum()

    # Metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # CSI/IoU

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'error_rate': 1 - accuracy
    }


def main():
    args = parse_args()

    # Update data config with temporal frames
    data_cfg.temporal_frames = args.temporal_frames
    data_cfg.year_target = args.temporal_frames + 1

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
        batch_size=1,  # Process one at a time for visualization
        num_workers=0,
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # Select loader based on split
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    print(f"Using {args.split} split ({len(loader)} samples)")

    # Infer T from data
    x_sample, y_sample = next(iter(loader))
    _, T, H, W = x_sample.shape
    print(f"Input shape: {x_sample.shape}, Target shape: {y_sample.shape}")
    print(f"Detected T (time steps) = {T}")

    # Create and load model
    print(f"\nLoading model from {args.checkpoint}")
    model = create_model(
        variant=args.variant,
        temporal_aggregation=args.temporal_aggregation,
        in_chans=T
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Model identifier for directory organization
    model_id = f"stswin_{args.variant}_{args.temporal_frames}y"

    # Create model-specific output directory with predictions subfolder
    output_dir = Path(args.output_dir) / model_id / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize predictions
    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    print(f"Model variant: {args.variant}")
    print(f"Temporal frames: {args.temporal_frames} input years + 1 target")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {output_dir}")

    all_metrics = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            if idx >= args.num_samples:
                break

            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get prediction
            outputs = model(inputs)

            # Move to CPU and convert to numpy
            input_np = inputs[0].cpu().numpy()  # (T, H, W)
            target_np = targets[0].cpu().numpy()  # (H, W)
            pred_np = outputs[0, 0].cpu().numpy()  # (H, W)

            # Compute metrics
            metrics = compute_metrics(target_np, pred_np, threshold=args.threshold)
            all_metrics.append(metrics)

            print(f"\nSample {idx + 1}/{args.num_samples}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  IoU/CSI: {metrics['iou']:.4f}")

            # Create visualization (simpler filename since already in model-specific directory)
            save_path = output_dir / f"{args.split}_sample_{idx + 1:03d}.png"
            visualize_prediction(
                input_np,
                target_np,
                pred_np,
                threshold=args.threshold,
                sample_idx=idx,
                save_path=save_path,
                variant=f"{args.variant}_{args.temporal_frames}y"
            )
            plt.close()  # Close to save memory

    # Print average metrics
    print(f"\n{'='*60}")
    print(f"Average metrics across {len(all_metrics)} samples:")
    print(f"{'='*60}")
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'iou']:
        avg_value = np.mean([m[metric_name] for m in all_metrics])
        std_value = np.std([m[metric_name] for m in all_metrics])
        print(f"  {metric_name.capitalize():10s}: {avg_value:.4f} ± {std_value:.4f}")
    print(f"{'='*60}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
