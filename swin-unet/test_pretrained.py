"""
test_pretrained.py

Test pretrained st-Swin-UNet models on the test set.

This script loads pretrained checkpoints and evaluates them, reusing existing
evaluation functions for consistency.

Usage examples (from repo root)
-------------------------------
    # Test tiny variant (4-year input)
    python -m swin-unet.test_pretrained --variant tiny

    # Test small variant (4-year input)
    python -m swin-unet.test_pretrained --variant small

    # Use CPU instead of GPU
    python -m swin-unet.test_pretrained --variant tiny --cpu

Pretrained Models
-----------------
The pretrained checkpoints are stored in swin-unet/pretrained/:
- stswin_tiny_4y_best.pt: Tiny variant, epoch 33, validation F1=0.665
- stswin_small_4y_best.pt: Small variant, epoch 14, validation F1=0.668

These were selected based on validation F1 scores following proper ML methodology.
"""

import argparse
import sys
from pathlib import Path
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
        description="Test pretrained st-Swin-UNet model on test set."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        choices=["tiny", "small"],
        help="Model variant to test: 'tiny' or 'small' (default: tiny)",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=4,
        help="Number of input temporal frames (default: 4). Note: pretrained models are only available for 4-year input.",
    )
    parser.add_argument(
        "--temporal-aggregation",
        type=str,
        default=model_cfg.temporal_aggregation,
        help=f"Temporal aggregation method (default: {model_cfg.temporal_aggregation})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: auto-load from pretrained/)",
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

    # Determine checkpoint path
    if args.checkpoint is None:
        # Auto-load from pretrained directory
        checkpoint_name = f"stswin_{args.variant}_{args.temporal_frames}y_best.pt"
        checkpoint_path = Path("swin-unet/pretrained") / checkpoint_name
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable pretrained models:")
        pretrained_dir = Path("swin-unet/pretrained")
        if pretrained_dir.exists():
            for ckpt in pretrained_dir.glob("*.pt"):
                print(f"  - {ckpt.name}")
        else:
            print("  (No pretrained models found)")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Testing Pretrained st-Swin-UNet Model")
    print(f"{'='*60}")
    print(f"Variant: {args.variant}")
    print(f"Temporal frames: {args.temporal_frames}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # Device configuration
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        pin_memory = False

    # Build dataloaders (disable cache to avoid conflicts with different temporal_frames)
    print("\nBuilding dataloaders")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,  # Disable cache to avoid temporal_frames conflicts
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

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
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

    print(f"Results:")
    print(f"  Loss:      {mean_test_loss:.6f}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  CSI/IoU:   {csi:.4f}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
