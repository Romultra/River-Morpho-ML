import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("transformer_cnn_model/scores/test_metrics_all_epochs_unet3d.csv")

plt.figure()
plt.plot(df["epoch"], df["test_loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Test loss")
plt.grid(True)

plt.figure()
plt.plot(df["epoch"], df["test_f1"], marker="o", label="F1")
plt.plot(df["epoch"], df["test_csi"], marker="o", label="CSI")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.show()