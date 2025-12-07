"""
eval_all_checkpoints.py

Evaluate *all* selected model checkpoints in a directory
on the test dataset and save metrics (loss, acc, prec, rec, f1, csi)
for each epoch into a single CSV file.

Usage example (from repo root):
    conda activate braided

    python -m transformer_cnn_model.eval_all_checkpoints \
        --checkpoint-dir transformer_cnn_model/checkpoints \
        --output-csv transformer_cnn_model/scores/test_metrics_all_epochs.csv

Options:
    --checkpoint-dir ...  # directory containing checkpoint .pt files
                            # (default: transformer_cnn_model/checkpoints)
    --checkpoint-pattern ...  # glob pattern to match checkpoint files
                            # (default: transformer_unet_epoch*.pt)
    --output-csv ...    # output CSV file path (default: transformer_cnn_model/test_metrics_all_epochs.csv)
    --cpu               # force CPU even if CUDA is available
    --batch-size 1      # batch size for evaluation
    --dir-folders ...   # dataset root (default: data/satellite/dataset_month3)
    --cache-dir ...     # cache directory (default: transformer_cnn_model/cache)
"""

import argparse
import re
from pathlib import Path
import csv

import torch

from transformer_cnn_model.model_architecture import TransformerUNet
from model.st_unet.st_unet import UNet3D
from transformer_cnn_model.train_eval_functions.train_eval import validation_unet
from transformer_cnn_model.preprocessing.load_data import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints on the test set and write metrics to CSV."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="transformer_cnn_model/checkpoints",
        help="Directory containing checkpoint .pt files "
             "(default: transformer_cnn_model/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="transformer_unet_epoch*.pt",
        help="Glob pattern to match checkpoint files "
             "(default: transformer_unet_epoch*.pt).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="transformer_cnn_model/test_metrics_all_epochs.csv",
        help="Path to output CSV file with metrics per epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--dir-folders",
        type=str,
        default="data/satellite/dataset_month3",
        help="Root folder of the satellite dataset "
             "(default: data/satellite/dataset_month3).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="transformer_cnn_model/cache",
        help="Directory where cached tensors are stored "
             "(default: transformer_cnn_model/cache).",
    )
    return parser.parse_args()


def extract_epoch_from_name(path: Path) -> int:
    """
    Extracts the epoch number from filenames like 'transformer_unet_epoch010.pt'.

    Returns an integer epoch, or -1 if no epoch could be parsed.
    """
    m = re.search(r"epoch(\d+)", path.name)
    if m:
        return int(m.group(1))
    return -1


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
        batch_size=args.batch_size,
        num_workers=0,            # evaluation is OK with 0, can increase if desired
        pin_memory=pin_memory,
        year_target=5,
        dir_folders=args.dir_folders,
        device="cpu",            # data lives on CPU; moved to GPU in validation_unet
        use_cache=True,
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
    # Using TransformerUNet 
    model = TransformerUNet(
        n_channels=T,                 # T inferred from data 
        n_classes=1,                  # binary prediction
        use_temporal_transformer=True,
        init_hid_dim=8,
        kernel_size=3,
        pooling="max",
        bilinear=False,
        drop_channels=False,
        p_drop=None,
        d_model=8,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
    )
    model.to(device)

    # # Alternatively, use UNet3D without transformer:
    # model = UNet3D(
    #     n_channels=T,
    #     n_classes=1,
    #     init_hid_dim=8,      
    #     kernel_size=3,
    #     pooling='max',
    #     bilinear=False,
    #     drop_channels=False,
    #     p_drop=None,
    # )
    # model.to(device)

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
            nonwater=0,
            water=1,
            device=str(device),
            loss_f="BCE",
            water_threshold=0.5,
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
