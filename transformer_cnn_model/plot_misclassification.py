"""
plot_misclassification.py

Generate a misclassification (TP / FP / FN / TN) map for a model checkpoint,
using parameters defined in config.py.

This script supports **two modes** of operation:

----------------------------------------------------------------------
1) YEAR-BASED MODE (recommended)
----------------------------------------------------------------------
You specify a target year and reach:

    --year YEAR
    --reach REACH
    --split {training, validation, testing}   (default: testing)

The script then:
    • Reads the model's required temporal depth T from config.py
      (T = year_target - 1).
    • Automatically loads the previous T years of images from disk.
    • Uses the target year as ground truth.
    • Runs the model and generates the misclassification map.

Example:
    python -m transformer_cnn_model.plot_misclassification \
        --year 2018 --reach 3 --split testing --save


----------------------------------------------------------------------
2) SAMPLE-BASED MODE (legacy, only for debugging)
----------------------------------------------------------------------
If --year is NOT provided, the script falls back to selecting
samples from the cached test dataset:

    --sample INDEX     (0-based index into the test dataset)

Example:
    python -m transformer_cnn_model.plot_misclassification --sample 3


----------------------------------------------------------------------
CHECKPOINT SELECTION
----------------------------------------------------------------------
By default the script loads the **latest checkpoint** found in:

    eval_cfg.checkpoint_dir
matching
    eval_cfg.checkpoint_pattern

You may load a specific epoch using:

    --epoch N


----------------------------------------------------------------------
ARGUMENT SUMMARY
----------------------------------------------------------------------
Year-based mode:
    --year YEAR            Target prediction year (activates year-based mode)
    --reach REACH          Reach ID (required when --year is used)
    --split SPLIT          Dataset split: training / validation / testing
    --epoch N              Load checkpoint for epoch N
    --threshold X          Water probability threshold (default from config)
    --save                 Save output instead of displaying

Sample-based mode:
    --sample INDEX         Test sample index (ignored if --year is provided)
    --epoch N              Load checkpoint for epoch N
    --threshold X          Water probability threshold
    --save                 Save output instead of displaying

Note:
    • Model architecture comes **only** from config.py (model_cfg.architecture)
      and cannot be overridden from CLI.
    • All paths (data, cache, plots, checkpoints) come from config.py.

"""

from pathlib import Path
from typing import Optional
import argparse
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from transformer_cnn_model.model_architecture import TransformerUNet
from model.st_unet.st_unet import UNet3D
from transformer_cnn_model.preprocessing.load_data import build_dataloaders
from transformer_cnn_model.preprocessing.dataset_generation import load_image_array
from preprocessing.satellite_analysis_pre import load_avg
from transformer_cnn_model.config import data_cfg, model_cfg, train_cfg, eval_cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------
# Build model given T and checkpoint
# ---------------------------------------------------------
def build_model(device: torch.device, T: int, ckpt_path: Path) -> torch.nn.Module:
    """Construct the model (from config) with n_channels = T and load weights."""
    model_type = model_cfg.architecture

    if model_type == "transunet":
        print("[INFO] Using TransformerUNet model")
        model = TransformerUNet(
            n_channels=T,
            n_classes=model_cfg.n_classes,
            use_temporal_transformer=model_cfg.use_temporal_transformer,
            init_hid_dim=model_cfg.init_hid_dim,
            kernel_size=model_cfg.kernel_size,
            pooling=model_cfg.pooling,
            bilinear=model_cfg.bilinear,
            drop_channels=model_cfg.drop_channels,
            p_drop=model_cfg.p_drop,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
        )

    elif model_type == "unet":
        print("[INFO] Using TransformerUNet model WITHOUT transformer (pure CNN)")
        model = TransformerUNet(
            n_channels=T,
            n_classes=model_cfg.n_classes,
            use_temporal_transformer=False,
            init_hid_dim=model_cfg.init_hid_dim,
            kernel_size=model_cfg.kernel_size,
            pooling=model_cfg.pooling,
            bilinear=model_cfg.bilinear,
            drop_channels=model_cfg.drop_channels,
            p_drop=model_cfg.p_drop,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            num_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
        )

    elif model_type == "unet3d":
        print("[INFO] Using UNet3D (no transformer)")
        model = UNet3D(
            n_channels=T,
            n_classes=model_cfg.n_classes,
            init_hid_dim=model_cfg.init_hid_dim,
            kernel_size=model_cfg.kernel_size,
            pooling=model_cfg.pooling,
            bilinear=model_cfg.bilinear,
            drop_channels=model_cfg.drop_channels,
            p_drop=model_cfg.p_drop,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    return model


# ---------------------------------------------------------
# Build input/target for a specific year/reach/split
# ---------------------------------------------------------
def build_sequence_for_target_year(
    split: str,          # "training", "validation", "testing"
    reach: int,          # e.g. 5
    target_year: int,    # e.g. 2018
    year_target: int,    # e.g. 5 or 10 (from config)
    nodata_value: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build input sequence (T = year_target-1) and target image for a specific
    (split, reach, target_year).

    Uses:
        - data_cfg.dir_folders as the dataset_monthX directory
        - same naming pattern as training:
          '{year}_{MM}_01_{split}_r{reach}.tif'
        - load_image_array + load_avg to fill nodata.
    """
    # Infer month from data_cfg.dir_folders (expects 'dataset_month3' etc.)
    m = re.search(r"dataset_month(\d+)", str(data_cfg.dir_folders))
    if m:
        month = int(m.group(1))
    else:
        raise ValueError(
            f"Could not infer month from data_cfg.dir_folders={data_cfg.dir_folders}"
        )

    dataset_dir = Path(data_cfg.dir_folders)
    # default collection name (can be added to config later if needed)
    collection = getattr(data_cfg, "collection", "JRC_GSW1_4_MonthlyHistory")

    folder = dataset_dir / f"{collection}_{split}_r{reach}"
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found for split/reach: {folder}")

    # Years: previous (year_target-1) + current target_year
    years = [target_year - (year_target - 1) + k for k in range(year_target)]

    images = []
    for y in years:
        fname = f"{y}_{month:02d}_01_{split}_r{reach}.tif"
        path = folder / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

        img = load_image_array(str(path), scaled_classes=True)
        avg = load_avg(
            split,
            reach,
            y,
            dir_averages="data/satellite/averages",
        )
        img = np.where(img == nodata_value, avg, img)
        images.append(img)

    images = np.stack(images, axis=0)  # (year_target, H, W)
    input_np = images[:-1]             # first T = year_target-1 → input
    target_np = images[-1]             # last year → ground truth

    return input_np, target_np


# ---------------------------------------------------------
# Misclassification computation
# ---------------------------------------------------------
def compute_misclassification_map(
    pred_probs: torch.Tensor,
    target: torch.Tensor,
    water_threshold: float = 0.5,
):
    """Return a map: 0=TN, 1=FP, 2=FN, 3=TP."""
    if pred_probs.ndim == 3:
        pred_probs = pred_probs.squeeze(0)

    pred_bin = (pred_probs >= water_threshold).float()

    mis_map = torch.zeros_like(target, dtype=torch.int8)
    water = 1.0
    nonwater = 0.0

    tp = (pred_bin == water) & (target == water)
    fp = (pred_bin == water) & (target == nonwater)
    tn = (pred_bin == nonwater) & (target == nonwater)
    fn = (pred_bin == nonwater) & (target == water)

    mis_map[tn] = 0
    mis_map[fp] = 1
    mis_map[fn] = 2
    mis_map[tp] = 3

    return mis_map


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_misclassification_map(
    mis_map: torch.Tensor,
    save_path: Optional[Path],
    title: str = "Misclassification map",
):
    """Plot discrete misclassification map."""
    mis_np = mis_map.cpu().numpy()

    colors = [
        "#808080",  # 0 = TN
        "#ff0000",  # 1 = FP
        "#0000ff",  # 2 = FN
        "#ffffff",  # 3 = TP
    ]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 10))
    plt.imshow(mis_np, cmap=cmap, norm=norm)
    plt.title(title)
    plt.axis("off")

    import matplotlib.patches as mpatches
    labels = [
        "TN (correct non-water)",
        "FP (pred water, gt non-water)",
        "FN (pred non-water, gt water)",
        "TP (correct water)",
    ]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot misclassification map using either year-based or sample-based selection."
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch whose checkpoint should be loaded. "
             "If omitted, loads the latest checkpoint in eval_cfg.checkpoint_dir.",
    )

    # Year-based selection
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Target year for prediction. If provided, year-based mode is used.",
    )
    parser.add_argument(
        "--reach",
        type=int,
        default=None,
        help="Reach ID (required in year-based mode).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testing",
        choices=["training", "validation", "testing"],
        help="Dataset split for year-based mode (default: testing).",
    )

    # Sample-based selection (legacy)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Index of the test sample to visualize (only used if --year is not given).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=train_cfg.water_threshold,
        help=f"Water probability threshold (default: {train_cfg.water_threshold}).",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the misclassification map instead of displaying it.",
    )

    return parser.parse_args()


def resolve_checkpoint(epoch: Optional[int]) -> Path:
    ckpt_dir = Path(eval_cfg.checkpoint_dir)
    pattern = eval_cfg.checkpoint_pattern

    ckpt_paths = sorted(ckpt_dir.glob(pattern))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir} matching pattern {pattern}"
        )

    if epoch is not None:
        for p in ckpt_paths:
            if f"epoch{epoch:03d}" in p.name:
                return p
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

    return ckpt_paths[-1]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    ckpt_path = resolve_checkpoint(args.epoch)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    water_threshold = args.threshold
    model_type = model_cfg.architecture

    # -------- YEAR-BASED MODE --------
    if args.year is not None:
        if args.reach is None:
            raise ValueError("--reach must be specified when using --year.")

        print(
            f"[INFO] Year-based mode: split={args.split}, reach={args.reach}, "
            f"target_year={args.year}, year_target={data_cfg.year_target}"
        )

        input_np, target_np = build_sequence_for_target_year(
            split=args.split,
            reach=args.reach,
            target_year=args.year,
            year_target=data_cfg.year_target,
            nodata_value=-1,
        )

        # (1, T, H, W)
        x = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).to(device)
        y = torch.tensor(target_np, dtype=torch.float32).to(device)

        T = x.shape[1]
        model = build_model(device, T, ckpt_path)

        with torch.no_grad():
            preds = model(x).squeeze(0)

        mis_map = compute_misclassification_map(preds, y, water_threshold)

        if args.save:
            out_name = (
                f"misclass_year{args.year}_reach{args.reach}_"
                f"{args.split}_{model_type}.png"
            )
            out_path = data_cfg.plots_dir / out_name
        else:
            out_path = None

        title = (
            f"Misclassification – year {args.year}, reach {args.reach}, "
            f"split {args.split}"
        )
        plot_misclassification_map(mis_map, save_path=out_path, title=title)
        return

    # -------- SAMPLE-BASED MODE (legacy) --------
    print("[INFO] Sample-based mode (no --year provided).")
    sample_idx = args.sample

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # infer T from test loader
    x_sample, _ = next(iter(test_loader))
    _, T, _, _ = x_sample.shape
    print(f"[INFO] Detected T = {T} from test_loader")

    model = build_model(device, T, ckpt_path)

    test_dataset = test_loader.dataset
    if sample_idx < 0 or sample_idx >= len(test_dataset):
        raise ValueError(
            f"sample_idx {sample_idx} out of range (0–{len(test_dataset)-1})"
        )

    x, y = test_dataset[sample_idx]
    x = x.unsqueeze(0).to(device)
    y = y.to(device)

    with torch.no_grad():
        preds = model(x).squeeze(0)

    mis_map = compute_misclassification_map(preds, y, water_threshold)

    if args.save:
        out_name = f"misclass_sample{sample_idx:03d}_{model_type}.png"
        out_path = data_cfg.plots_dir / out_name
    else:
        out_path = None

    plot_misclassification_map(
        mis_map,
        save_path=out_path,
        title=f"Misclassification map – sample {sample_idx}",
    )


if __name__ == "__main__":
    main()
