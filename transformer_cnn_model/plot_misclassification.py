"""
plot_misclassification.py

Load a selected checkpoint and generate a misclassification map for a chosen
test sample (TP / FP / FN / TN).
Usage example (from repo root):
    conda activate braided
    python -m transformer_cnn_model.plot_misclassification \
        --checkpoint transformer_cnn_model/checkpoints_transformer_unet/transformer_unet_epoch018.pt \
        --sample 3 \
        --model transunet \
        --save

Options:
    --checkpoint ...  # path to checkpoint .pt file
    --sample 0        # index of test sample to visualize
    --model ...       # which model to load: transunet / unet / unet3d
    --threshold 0.5   # water probability threshold
    --save            # save output instead of displaying
"""

from pathlib import Path
from typing import Optional
import argparse

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from transformer_cnn_model.model_architecture import TransformerUNet
from model.st_unet.st_unet import UNet3D
from transformer_cnn_model.preprocessing.load_data import build_dataloaders

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------
# Build model (based on checkpoint + T inferred from data)
# ---------------------------------------------------------
def build_model_from_loader(
    device: torch.device,
    test_loader,
    ckpt_path: Path,
    model_type: str = "transunet"
):
    """Infer T from loader and construct the appropriate model."""

    x_sample, _ = next(iter(test_loader))
    _, T, _, _ = x_sample.shape
    print(f"[INFO] Detected T = {T}")

    # ---- Select which model to instantiate ----
    if model_type == "transunet":
        print("[INFO] Using TransformerUNet model")
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
    
    elif model_type == "unet":
        print("[INFO] Using TransformerUNet model without transformer")
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

    elif model_type == "unet3d":
        print("[INFO] Using UNet3D (no transformer)")
        model = UNet3D(
            n_channels=T,
            n_classes=1,
            init_hid_dim=8,
            kernel_size=3,
            pooling='max',
            bilinear=False,
            drop_channels=False,
            p_drop=None,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ---- Load weights ----
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    return model


# ---------------------------------------------------------
# Misclassification computation
# ---------------------------------------------------------
def compute_misclassification_map(
    pred_probs: torch.Tensor,
    target: torch.Tensor,
    water_threshold: float = 0.5
):
    """Return a map: 0=TN, 1=FP, 2=FN, 3=TP"""
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
    title: str = "Misclassification map"
):
    """Plot discrete misclassification map."""
    mis_np = mis_map.cpu().numpy()

    colors = [
        "#808080",  # 0 = TN
        "#ff0000",  # 1 = FP (red)
        "#0000ff",  # 2 = FN (blue)
        "#ffffff",  # 3 = TP (white)
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
        "TP (correct water)"
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
# Main (argparse)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot misclassification map for a model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Index of test sample to visualize")
    parser.add_argument("--model", type=str, default="transunet",
                        choices=["transunet", "unet", "unet3d"],
                        help="Which model to load")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Water probability threshold")
    parser.add_argument("--save", action="store_true",
                        help="Save output instead of displaying")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    sample_idx = args.sample
    model_type = args.model
    water_threshold = args.threshold

    # ---- Device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Data ----
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=1,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        year_target=5,
        dir_folders="data/satellite/dataset_month3",
        device="cpu",
        use_cache=True,
        cache_dir="transformer_cnn_model/cache",
    )

    # ---- Model ----
    model = build_model_from_loader(device, test_loader, ckpt_path, model_type=model_type)

    # ---- Sample ----
    test_dataset = test_loader.dataset
    if sample_idx < 0 or sample_idx >= len(test_dataset):
        raise ValueError(f"sample_idx {sample_idx} out of range (0–{len(test_dataset)-1})")

    x, y = test_dataset[sample_idx]
    x = x.unsqueeze(0).to(device)
    y = y.to(device)

    with torch.no_grad():
        preds = model(x).squeeze(0)

    mis_map = compute_misclassification_map(preds, y, water_threshold)

    # ---- Save or show ----
    if args.save:
        out_name = f"misclass_sample{sample_idx:03d}_{model_type}.png"
        out_path = Path("transformer_cnn_model/plots") / out_name
    else:
        out_path = None

    plot_misclassification_map(
        mis_map,
        save_path=out_path,
        title=f"Misclassification map – sample {sample_idx}"
    )


if __name__ == "__main__":
    main()
