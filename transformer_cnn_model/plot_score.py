import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to CSV with metrics for all epochs
csv_path = "transformer_cnn_model/scores/test_metrics_all_epochs_transunet.csv"

# Directory where plots will be saved
plots_dir = Path("transformer_cnn_model/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Load metrics
df = pd.read_csv(csv_path)

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
        f"loss = {row['test_loss']:.6f}, "
        f"F1 = {row['test_f1']:.4f}, "
        f"CSI = {row['test_csi']:.4f}, "
        f"prec = {row['test_prec']:.4f}, "
        f"rec = {row['test_rec']:.4f}, "
        f"acc = {row['test_acc']:.4f})"
    )
    return row


print("\n==================== BEST EPOCHS ====================")

# Highest F1
best_f1 = print_best(df, "test_f1")

# Highest CSI
best_csi = print_best(df, "test_csi")

# Lowest loss
best_loss = print_best(df, "test_loss", higher_is_better=False)

# Highest precision
best_prec = print_best(df, "test_prec")

# Highest recall
best_rec = print_best(df, "test_rec")

# Highest accuracy
best_acc = print_best(df, "test_acc")

print("=====================================================\n")

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
        best_f1["test_f1"],
        best_csi["test_csi"],
        best_loss["test_loss"],
        best_prec["test_prec"],
        best_rec["test_rec"],
        best_acc["test_acc"],
    ]
})

summary_path = plots_dir / "best_epoch_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary of best epochs to {summary_path}")

# --------------------------------------------------------------------
# Plot test loss vs epoch
# --------------------------------------------------------------------
fig_loss = plt.figure()
plt.plot(df["epoch"], df["test_loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Test loss")
plt.title("Test loss vs Epoch (TransformerUNet)")
plt.grid(True)

loss_plot_path = plots_dir / "test_loss_transunet.png"
fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
print(f"Saved loss plot to {loss_plot_path}")

# --------------------------------------------------------------------
# Plot F1 and CSI vs epoch
# --------------------------------------------------------------------
fig_scores = plt.figure()
plt.plot(df["epoch"], df["test_f1"], marker="o", label="F1")
plt.plot(df["epoch"], df["test_csi"], marker="o", label="CSI")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("F1 and CSI vs Epoch (TransformerUNet)")
plt.legend()
plt.grid(True)

scores_plot_path = plots_dir / "test_f1_csi_transunet.png"
fig_scores.savefig(scores_plot_path, dpi=300, bbox_inches="tight")
print(f"Saved F1/CSI plot to {scores_plot_path}")

plt.show()
