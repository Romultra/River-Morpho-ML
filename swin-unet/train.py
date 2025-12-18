"""
train.py

Training script for st-Swin-UNet model using the JRC Global Surface Water dataset.
Follows the same pattern as transformer_cnn_model/train.py with adaptations for
the Swin Transformer architecture.

Usage:
    # Train tiny variant (default)
    python -m swin-unet.train

    # Train small variant with custom batch size
    python -m swin-unet.train --variant small --batch-size 4

    # Train with custom learning rate and epochs
    python -m swin-unet.train --lr 5e-5 --epochs 100

    # Use learning rate scheduler
    python -m swin-unet.train --use-scheduler
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add swin-unet directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from st_swin_unet_model import create_swin_unet_tiny, create_swin_unet_small

# Import shared utilities from transformer_cnn_model
from transformer_cnn_model.train_eval_functions.train_eval import training_unet, validation_unet
from transformer_cnn_model.preprocessing.load_data import build_dataloaders

# Import local config
from config import data_cfg, model_cfg, train_cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train st-Swin-UNet model for river morphology prediction."
    )

    # Model configuration
    parser.add_argument(
        "--variant",
        type=str,
        default=model_cfg.variant,
        choices=["tiny", "small"],
        help=f"Model variant: 'tiny' (6.8M params) or 'small' (11M params) (default: {model_cfg.variant})",
    )
    parser.add_argument(
        "--temporal-aggregation",
        type=str,
        default=model_cfg.temporal_aggregation,
        choices=["concat_proj", "learned_weighted_sum", "mean"],
        help=f"Temporal aggregation method (default: {model_cfg.temporal_aggregation})",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=data_cfg.temporal_frames,
        choices=[4, 9],
        help=f"Number of input temporal frames: 4 or 9 years (default: {data_cfg.temporal_frames})",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_cfg.num_epochs,
        help=f"Number of training epochs (default: {train_cfg.num_epochs})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=train_cfg.lr,
        help=f"Learning rate (default: {train_cfg.lr})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=train_cfg.weight_decay,
        help=f"Weight decay (default: {train_cfg.weight_decay})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=data_cfg.batch_size,
        help=f"Batch size (default: {data_cfg.batch_size})",
    )
    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        help="Use cosine annealing learning rate scheduler",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=train_cfg.warmup_epochs,
        help=f"Warmup epochs before scheduler starts (default: {train_cfg.warmup_epochs})",
    )

    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Custom checkpoint directory (default: auto-generated based on variant)",
    )

    # Hardware configuration
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if CUDA is available",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=data_cfg.num_workers,
        help=f"Number of data loading workers (default: {data_cfg.num_workers})",
    )

    return parser.parse_args()


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
    # Parse command-line arguments
    args = parse_args()

    # Update configs with command-line arguments
    model_cfg.variant = args.variant
    model_cfg.temporal_aggregation = args.temporal_aggregation
    data_cfg.temporal_frames = args.temporal_frames
    data_cfg.year_target = args.temporal_frames + 1  # Update year_target based on temporal_frames
    train_cfg.num_epochs = args.epochs
    train_cfg.lr = args.lr
    train_cfg.weight_decay = args.weight_decay
    data_cfg.batch_size = args.batch_size
    data_cfg.num_workers = args.num_workers
    train_cfg.use_scheduler = args.use_scheduler
    train_cfg.warmup_epochs = args.warmup_epochs

    # Set checkpoint directory based on variant and temporal frames if not specified
    model_id = f"{args.variant}_{args.temporal_frames}y"
    if args.checkpoint_dir:
        train_cfg.ckpt_dir = Path(args.checkpoint_dir)
    else:
        train_cfg.ckpt_dir = Path(f"swin-unet/checkpoints_{model_id}")

    # -----------------------
    # 1. Device configuration
    # -----------------------
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        print("Using CPU")
        pin_memory = False

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Model variant: {model_cfg.variant}")
    print(f"Temporal frames: {data_cfg.temporal_frames} input years + 1 target")
    print(f"Temporal aggregation: {model_cfg.temporal_aggregation}")
    print(f"Epochs: {train_cfg.num_epochs}")
    print(f"Learning rate: {train_cfg.lr}")
    print(f"Weight decay: {train_cfg.weight_decay}")
    print(f"Batch size: {data_cfg.batch_size}")
    print(f"Use scheduler: {train_cfg.use_scheduler}")
    print(f"Checkpoint dir: {train_cfg.ckpt_dir}")
    print(f"{'='*60}\n")

    # -----------------------
    # 2. Build DataLoaders
    # -----------------------
    print("Building dataloaders (this may take a while the first time)...")

    data_cfg.cache_dir.mkdir(exist_ok=True)
    train_cfg.ckpt_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=pin_memory,
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # Peek at one batch to infer T (temporal dimension)
    x_sample, y_sample = next(iter(train_loader))
    B, T, H, W = x_sample.shape
    print(f"Sample batch shape: x={x_sample.shape}, y={y_sample.shape}")
    print(f"Detected T (time steps) = {T}")

    # Update model config with inferred in_chans
    model_cfg.in_chans = T

    # -----------------------
    # 3. Instantiate the model
    # -----------------------
    print(f"\nCreating st-Swin-UNet model:")
    print(f"  Variant: {model_cfg.variant}")
    print(f"  Temporal aggregation: {model_cfg.temporal_aggregation}")
    print(f"  Input channels (temporal frames): {T}")

    model = create_model(
        variant=model_cfg.variant,
        temporal_aggregation=model_cfg.temporal_aggregation,
        in_chans=T
    )

    model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # -----------------------
    # 4. Optimizer
    # -----------------------
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay
    )
    print(f"\nOptimizer: AdamW (lr={train_cfg.lr}, weight_decay={train_cfg.weight_decay})")

    # -----------------------
    # 5. Optional: Learning rate scheduler
    # -----------------------
    scheduler = None
    if train_cfg.use_scheduler:
        if train_cfg.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.num_epochs - train_cfg.warmup_epochs,
                eta_min=train_cfg.lr * 0.01
            )
            print(f"Using CosineAnnealingLR scheduler with {train_cfg.warmup_epochs} warmup epochs")
        # Add other scheduler types here if needed

    # -----------------------
    # 6. Training loop
    # -----------------------
    print(f"\n{'='*60}")
    print(f"Starting training for {train_cfg.num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, train_cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{train_cfg.num_epochs} =====")

        # Checkpoint path
        ckpt_path = train_cfg.ckpt_dir / f"stswin_{model_id}_epoch{epoch:03d}.pt"

        # Training
        train_losses = training_unet(
            model,
            train_loader,
            optimizer,
            nonwater=train_cfg.nonwater_label,
            water=train_cfg.water_label,
            pixel_size=train_cfg.pixel_size,
            water_threshold=train_cfg.water_threshold,
            device=str(device),
            loss_f=train_cfg.loss_f,
            physics=train_cfg.physics,
            verbose=True,
        )

        mean_train_loss = float(torch.tensor(train_losses).mean())
        print(f"Train loss: {mean_train_loss:.6f}")

        # Validation
        val_losses, acc, prec, rec, f1, csi = validation_unet(
            model,
            val_loader,
            nonwater=train_cfg.nonwater_label,
            water=train_cfg.water_label,
            device=str(device),
            loss_f=train_cfg.loss_f,
            water_threshold=train_cfg.water_threshold,
        )

        mean_val_loss = float(torch.tensor(val_losses).mean())
        print(f"Val loss: {mean_val_loss:.6f}")
        print(
            f"Val metrics: "
            f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, csi={csi:.4f}"
        )

        # Learning rate scheduling (after warmup)
        if scheduler is not None and epoch > train_cfg.warmup_epochs:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint
        if epoch % train_cfg.save_every_n_epochs == 0:
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # -----------------------
    # 7. Final test evaluation
    # -----------------------
    print(f"\n{'='*60}")
    print("Final evaluation on test set")
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
    print(f"Test loss: {mean_test_loss:.6f}")
    print(
        f"Test metrics: "
        f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, csi={csi:.4f}"
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
