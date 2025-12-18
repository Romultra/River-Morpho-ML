"""
plot_score.py

Load the CSV with metrics for all epochs (validation or test) and:
  - print the best epoch for several metrics (F1, CSI, loss, precision, recall, accuracy)
  - save plots of loss vs epoch and F1/CSI vs epoch.

IMPORTANT: For proper methodology, use VALIDATION metrics to select best checkpoint.
Only plot test metrics if you're visualizing the final selected model's performance.

Adapted from transformer_cnn_model/plot_score.py for st-Swin-UNet.

Usage example (from repo root)
------------------------------
    # Plot VALIDATION metrics (recommended for model selection)
    python -m swin-unet.plot_score --variant tiny --temporal-frames 4 --split val

    # Plot test metrics (only for final reporting)
    python -m swin-unet.plot_score --variant tiny --temporal-frames 4 --split test

Outputs
-------
  - Console printout of best epochs for each metric
  - CSV summary: plots/stswin_{variant}_{temporal}y/best_epoch_summary.csv
  - Loss plot: plots/stswin_{variant}_{temporal}y/{split}_loss.png
  - F1/CSI plot: plots/stswin_{variant}_{temporal}y/{split}_f1_csi.png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add swin-unet directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import data_cfg, eval_cfg, model_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot test metrics from CSV and identify best epochs."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=model_cfg.variant,
        choices=["tiny", "small"],
        help=f"Model variant to plot (default: {model_cfg.variant})",
    )
    parser.add_argument(
        "--temporal-frames",
        type=int,
        default=data_cfg.temporal_frames,
        choices=[4, 9],
        help=f"Number of input temporal frames: 4 or 9 years (default: {data_cfg.temporal_frames})",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to metrics CSV file (default: auto-generated based on variant and temporal frames)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(data_cfg.plots_dir),
        help=f"Directory to save plots (default: {data_cfg.plots_dir})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to plot: 'val' for validation (recommended), 'test' for final results (default: val)",
    )
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Update eval config for the specified variant and temporal frames
eval_cfg.update_for_variant(args.variant, args.temporal_frames)

# Path to CSV with metrics for all epochs
if args.csv_path:
    csv_path = args.csv_path
else:
    # Auto-generate CSV name based on split
    model_id = f"{args.variant}_{args.temporal_frames}y"
    scores_dir = Path("swin-unet/scores")
    csv_path = scores_dir / f"{args.split}_metrics_all_epochs_stswin_{model_id}.csv"

# Model identifier for titles / filenames
model_name = f"stswin_{args.variant}_{args.temporal_frames}y"  # e.g. "stswin_tiny_4y" or "stswin_small_9y"

# Directory where plots will be saved (model-specific subdirectory)
plots_dir = Path(args.output_dir) / model_name
plots_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
print(f"Loading metrics from {csv_path}")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} epochs of data\n")

# Split name for labels
split_name = "Validation" if args.split == "val" else "Test"

# --------------------------------------------------------------------
# Helper function to print best row for any metric
# --------------------------------------------------------------------
def print_best(df, metric, higher_is_better=True):
    if higher_is_better:
        idx = df[metric].idxmax()
    else:
        idx = df[metric].idxmin()

    row = df.loc[idx]
    print(
        f"Best {metric} : epoch {int(row['epoch'])} "
        f"({metric} = {row[metric]:.6f}, "
        f"loss = {row['loss']:.6f}, "
        f"F1 = {row['f1']:.4f}, "
        f"CSI = {row['csi']:.4f}, "
        f"prec = {row['prec']:.4f}, "
        f"rec = {row['rec']:.4f}, "
        f"acc = {row['acc']:.4f})"
    )
    return row


print(f"==================== BEST EPOCHS ({split_name.upper()}) ====================")

# Highest F1
best_f1 = print_best(df, "f1")

# Highest CSI
best_csi = print_best(df, "csi")

# Lowest loss
best_loss = print_best(df, "loss", higher_is_better=False)

# Highest precision
best_prec = print_best(df, "prec")

# Highest recall
best_rec = print_best(df, "rec")

# Highest accuracy
best_acc = print_best(df, "acc")

print("=" * 60 + "\n")

# Optional summary table
summary_df = pd.DataFrame({
    "metric": ["F1", "CSI", "Loss", "Precision", "Recall", "Accuracy"],
    "epoch": [
        int(best_f1["epoch"]),
        int(best_csi["epoch"]),
        int(best_loss["epoch"]),
        int(best_prec["epoch"]),
        int(best_rec["epoch"]),
        int(best_acc["epoch"]),
    ],
    "value": [
        best_f1["f1"],
        best_csi["csi"],
        best_loss["loss"],
        best_prec["prec"],
        best_rec["rec"],
        best_acc["acc"],
    ]
})
summary_path = plots_dir / f"best_epoch_summary_{args.split}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary of best epochs to {summary_path}")

# --------------------------------------------------------------------
# Plot loss vs epoch
# --------------------------------------------------------------------
fig_loss = plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["loss"], marker="o", linewidth=2, markersize=6)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel(f"{split_name} Loss", fontsize=12)
plt.title(f"{split_name} Loss vs Epoch (st-Swin-UNet {args.variant}, {args.temporal_frames}y)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)

loss_plot_path = plots_dir / f"{args.split}_loss.png"
fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
print(f"Saved loss plot to {loss_plot_path}")

# --------------------------------------------------------------------
# Plot F1, CSI, Precision, and Recall vs epoch
# --------------------------------------------------------------------
fig_scores = plt.figure(figsize=(12, 7))
plt.plot(df["epoch"], df["f1"], marker="o", linewidth=2, markersize=5, label="F1", color='#1f77b4')
plt.plot(df["epoch"], df["csi"], marker="s", linewidth=2, markersize=5, label="CSI", color='#ff7f0e')
plt.plot(df["epoch"], df["prec"], marker="^", linewidth=2, markersize=5, label="Precision", color='#2ca02c')
plt.plot(df["epoch"], df["rec"], marker="v", linewidth=2, markersize=5, label="Recall", color='#d62728')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title(f"{split_name} Metrics vs Epoch (st-Swin-UNet {args.variant}, {args.temporal_frames}y)", fontsize=14, fontweight="bold")
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)

scores_plot_path = plots_dir / f"{args.split}_metrics.png"
fig_scores.savefig(scores_plot_path, dpi=300, bbox_inches="tight")
print(f"Saved metrics plot to {scores_plot_path}")

print("\nDone! Close the plot windows to exit.")
plt.show()
