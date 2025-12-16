"""
plot_input_output.py

Visualize model inputs and outputs for a given checkpoint.

This script plots:
  • the input temporal sequence (previous T years)
  • the model prediction for the target year
  • the ground-truth target image

It supports TWO modes of operation:

----------------------------------------------------------------------
1) YEAR-BASED MODE (recommended)
----------------------------------------------------------------------
Select a target year and reach:

    --year YEAR
    --reach REACH
    --split {training, validation, testing}   (default: testing)

The script then:
  • Uses year_target from config.py to build the input sequence:
        T = year_target - 1
  • Loads the previous T years as input and the target year as ground truth
  • Runs the model and visualizes outputs

Example:
    python -m transformer_cnn_model.plot_input_output \
        --year 2018 \
        --reach 1 \
        --split testing \
        --epoch 18 \
        --save

Prediction/Target only (skip input sequence; recommended when T is large, e.g. 9):
    python -m transformer_cnn_model.plot_input_output \
        --year 2018 \
        --reach 1 \
        --split testing \
        --epoch 18 \
        --pred-only \
        --save


----------------------------------------------------------------------
2) SAMPLE-BASED MODE (legacy / debugging)
----------------------------------------------------------------------
If --year is NOT provided, a cached test sample is used:

    --sample INDEX     (0-based index)

Example:
    python -m transformer_cnn_model.plot_input_output \
        --sample 3 \
        --epoch 18


----------------------------------------------------------------------
CHECKPOINT SELECTION
----------------------------------------------------------------------
By default the script loads the latest checkpoint found in:

    eval_cfg.checkpoint_dir

matching:

    eval_cfg.checkpoint_pattern

You may load a specific epoch with:

    --epoch N


----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------
If --save is specified, figures are written to:

    data_cfg.plots_dir

If --pred-only is used, the saved file name gets a "_predonly" suffix
to avoid overwriting the full-layout plot.
"""

from pathlib import Path
from typing import Optional
import argparse
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap

from transformer_cnn_model.model_architecture import TransformerUNet
from model.st_unet.st_unet import UNet3D
from transformer_cnn_model.preprocessing.load_data import build_dataloaders
from transformer_cnn_model.preprocessing.dataset_generation import load_image_array
from preprocessing.satellite_analysis_pre import load_avg
from transformer_cnn_model.config import data_cfg, model_cfg, train_cfg, eval_cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------
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
# Returns: input_np (T,H,W), target_np (H,W)
# ---------------------------------------------------------
def build_sequence_for_target_year(
    split: str,
    reach: int,
    target_year: int,
    year_target: int,
    nodata_value: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build input sequence (T = year_target-1) and target image for a specific
    (split, reach, target_year).

    Uses:
      - data_cfg.dir_folders as the dataset_monthX directory
      - naming pattern:
        '{year}_{MM}_01_{split}_r{reach}.tif'
      - load_image_array + load_avg to fill nodata.
    """
    m = re.search(r"dataset_month(\d+)", str(data_cfg.dir_folders))
    if m:
        month = int(m.group(1))
    else:
        raise ValueError(
            f"Could not infer month from data_cfg.dir_folders={data_cfg.dir_folders}"
        )

    dataset_dir = Path(data_cfg.dir_folders)
    collection = getattr(data_cfg, "collection", "JRC_GSW1_4_MonthlyHistory")

    folder = dataset_dir / f"{collection}_{split}_r{reach}"
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found for split/reach: {folder}")

    years = [target_year - (year_target - 1) + k for k in range(year_target)]

    images = []
    for y in years:
        fname = f"{y}_{month:02d}_01_{split}_r{reach}.tif"
        path = folder / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

        img = load_image_array(str(path), scaled_classes=True)
        avg = load_avg(split, reach, y, dir_averages="data/satellite/averages")
        img = np.where(img == nodata_value, avg, img)
        images.append(img)

    images = np.stack(images, axis=0)  # (year_target, H, W)
    input_np = images[:-1]             # (T, H, W)
    target_np = images[-1]             # (H, W)
    return input_np, target_np


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
_GREY_WHITE = ListedColormap([(0.6, 0.6, 0.6), (1.0, 1.0, 1.0)])


def plot_inputs_pred_target(
    inputs_np: np.ndarray,     # (T,H,W)
    pred_probs_np: np.ndarray, # (H,W)
    target_np: np.ndarray,     # (H,W)
    threshold: float,
    save_path: Optional[Path],
    title: str,
):
    """
    Full layout for T=4 only:
      Row 1: 4 input frames
      Gap band: bracket + label
      Row 2: Prediction + Target
    """
    T = inputs_np.shape[0]
    if T != 4:
        raise ValueError(
            f"plot_inputs_pred_target expects T=4, got T={T}. "
            "Use --pred-only to skip plotting inputs for large T."
        )

    pred_bin = (pred_probs_np >= threshold).astype(np.float32)

    fig = plt.figure(figsize=(14, 9))

    # ----- spacing / layout params (figure fraction) -----
    # Title
    fig.suptitle(title, x=0.5, y=0.985, ha="center", fontsize=22)

    # Margins
    left_margin = 0.06
    right_margin = 0.06

    # Rows
    inputs_row_h = 0.34
    bottom_row_h = 0.38
    gap_band_h = 0.10
    label_to_bottom_gap = 0.06

    # Vertical placement (you may tune inputs_row_y0 if needed)
    inputs_row_y0 = 0.54
    gap_band_y0 = inputs_row_y0 - gap_band_h
    bottom_row_y0 = gap_band_y0 - label_to_bottom_gap - bottom_row_h

    # Inputs row: tight grouping
    gap_x_inputs = 0.012
    inputs_group_x0 = left_margin
    inputs_group_x1 = 1.0 - right_margin
    inputs_group_w = inputs_group_x1 - inputs_group_x0
    input_w = (inputs_group_w - 3 * gap_x_inputs) / 4.0

    # Bottom row: centered under inputs group
    bottom_panel_w = 0.23
    gap_x_bottom = 0.05
    bottom_group_w = 2 * bottom_panel_w + gap_x_bottom
    center_x = (inputs_group_x0 + inputs_group_x1) / 2.0
    bottom_group_x0 = center_x - bottom_group_w / 2.0

    # ----- Row 1: inputs -----
    input_axes = []
    for i in range(4):
        x0 = inputs_group_x0 + i * (input_w + gap_x_inputs)
        ax = fig.add_axes([x0, inputs_row_y0, input_w, inputs_row_h])
        ax.imshow(inputs_np[i], cmap=_GREY_WHITE, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"Input t={i}", fontsize=16, pad=10)
        ax.axis("off")
        input_axes.append(ax)

    # ----- Row 2: Prediction + Target -----
    ax_pred = fig.add_axes([bottom_group_x0, bottom_row_y0, bottom_panel_w, bottom_row_h])
    ax_tgt  = fig.add_axes([bottom_group_x0 + bottom_panel_w + gap_x_bottom, bottom_row_y0, bottom_panel_w, bottom_row_h])

    ax_pred.imshow(pred_bin, cmap=_GREY_WHITE, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_pred.set_title("Prediction", fontsize=18, pad=10)
    ax_pred.axis("off")

    ax_tgt.imshow(target_np, cmap=_GREY_WHITE, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_tgt.set_title("Target", fontsize=18, pad=10)
    ax_tgt.axis("off")

    # ----- Bracket + label in gap band -----
    left = input_axes[0].get_position().x0
    right = input_axes[-1].get_position().x1

    bracket_y = gap_band_y0 + 0.07
    tick = 0.018

    fig.add_artist(mlines.Line2D([left, right], [bracket_y, bracket_y],
                                 transform=fig.transFigure, lw=2.2, color="black"))
    fig.add_artist(mlines.Line2D([left, left], [bracket_y, bracket_y + tick],
                                 transform=fig.transFigure, lw=2.2, color="black"))
    fig.add_artist(mlines.Line2D([right, right], [bracket_y, bracket_y + tick],
                                 transform=fig.transFigure, lw=2.2, color="black"))

    label_y = bracket_y - 0.03
    fig.text((left + right) / 2.0, label_y, "Input sequence",
             ha="center", va="top", fontsize=18)

    # ----- Save/show -----
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pred_target_only(
    pred_probs_np: np.ndarray, # (H,W)
    target_np: np.ndarray,     # (H,W)
    threshold: float,
    save_path: Optional[Path],
    title: str,
):
    """Compact plot: Prediction + Target only (no input sequence)."""
    pred_bin = (pred_probs_np >= threshold).astype(np.float32)

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title, x=0.5, y=0.98, ha="center", fontsize=20)

    # Manual axes (stable spacing)
    ax_pred = fig.add_axes([0.08, 0.12, 0.40, 0.72])
    ax_tgt  = fig.add_axes([0.52, 0.12, 0.40, 0.72])

    ax_pred.imshow(pred_bin, cmap=_GREY_WHITE, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_pred.set_title("Prediction", fontsize=18, pad=10)
    ax_pred.axis("off")

    ax_tgt.imshow(target_np, cmap=_GREY_WHITE, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_tgt.set_title("Target", fontsize=18, pad=10)
    ax_tgt.axis("off")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot inputs + prediction + target (or prediction+target only)."
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch whose checkpoint should be loaded. If omitted, loads the latest checkpoint.",
    )

    # Year-based selection
    parser.add_argument("--year", type=int, default=None, help="Target year (activates year-based mode).")
    parser.add_argument("--reach", type=int, default=None, help="Reach ID (required if --year is used).")
    parser.add_argument(
        "--split",
        type=str,
        default="testing",
        choices=["training", "validation", "testing"],
        help="Dataset split for year-based mode (default: testing).",
    )

    # Sample-based selection
    parser.add_argument("--sample", type=int, default=0, help="Test sample index (used if --year not given).")

    parser.add_argument(
        "--threshold",
        type=float,
        default=train_cfg.water_threshold,
        help=f"Water probability threshold (default: {train_cfg.water_threshold}).",
    )

    parser.add_argument(
        "--pred-only",
        action="store_true",
        help="Plot only Prediction and Target (skip input sequence). Useful when T is large.",
    )

    parser.add_argument("--save", action="store_true", help="Save output instead of displaying it.")
    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    ckpt_path = resolve_checkpoint(args.epoch)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    threshold = args.threshold
    model_type = model_cfg.architecture

    # -------- YEAR-BASED MODE --------
    if args.year is not None:
        if args.reach is None:
            raise ValueError("--reach must be specified when using --year.")

        input_np, target_np = build_sequence_for_target_year(
            split=args.split,
            reach=args.reach,
            target_year=args.year,
            year_target=data_cfg.year_target,
            nodata_value=-1,
        )

        # (1, T, H, W)
        x = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).to(device)

        T = x.shape[1]
        model = build_model(device, T, ckpt_path)

        with torch.no_grad():
            pred = model(x)
        pred = pred.squeeze().detach().cpu().numpy()  # (H, W) (after squeeze)

        title = f"Prediction – year {args.year}, reach {args.reach}, split {args.split} ({model_type})"

        if args.save:
            suffix = "_predonly" if args.pred_only else ""
            out_name = f"prediction_year{args.year}_reach{args.reach}_{args.split}_{model_type}{suffix}.png"
            out_path = data_cfg.plots_dir / out_name
        else:
            out_path = None

        if args.pred_only:
            plot_pred_target_only(
                pred_probs_np=pred,
                target_np=target_np,
                threshold=threshold,
                save_path=out_path,
                title=title,
            )
        else:
            plot_inputs_pred_target(
                inputs_np=input_np,
                pred_probs_np=pred,
                target_np=target_np,
                threshold=threshold,
                save_path=out_path,
                title=title,
            )
        return

    # -------- SAMPLE-BASED MODE --------
    print("[INFO] Sample-based mode (no --year provided).")
    sample_idx = args.sample

    _, _, test_loader = build_dataloaders(
        batch_size=data_cfg.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        year_target=data_cfg.year_target,
        dir_folders=data_cfg.dir_folders,
        device="cpu",
        use_cache=data_cfg.use_cache,
        cache_dir=data_cfg.cache_dir,
    )

    # infer T from loader
    x_sample, _ = next(iter(test_loader))
    _, T, _, _ = x_sample.shape
    print(f"[INFO] Detected T = {T} from test_loader")

    model = build_model(device, T, ckpt_path)

    test_dataset = test_loader.dataset
    if sample_idx < 0 or sample_idx >= len(test_dataset):
        raise ValueError(f"sample_idx {sample_idx} out of range (0–{len(test_dataset)-1})")

    x, y = test_dataset[sample_idx]     # x: (T,H,W), y: (H,W)
    x = x.unsqueeze(0).to(device)       # (1,T,H,W)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    with torch.no_grad():
        pred = model(x)
    pred = pred.squeeze().detach().cpu().numpy()  # (H,W)

    title = f"Prediction – sample {sample_idx:03d} ({model_type})"

    if args.save:
        suffix = "_predonly" if args.pred_only else ""
        out_name = f"prediction_sample{sample_idx:03d}_{model_type}{suffix}.png"
        out_path = data_cfg.plots_dir / out_name
    else:
        out_path = None

    if args.pred_only:
        plot_pred_target_only(
            pred_probs_np=pred,
            target_np=y_np,
            threshold=threshold,
            save_path=out_path,
            title=title,
        )
    else:
        plot_inputs_pred_target(
            inputs_np=x.squeeze(0).detach().cpu().numpy(),
            pred_probs_np=pred,
            target_np=y_np,
            threshold=threshold,
            save_path=out_path,
            title=title,
        )


if __name__ == "__main__":
    main()
