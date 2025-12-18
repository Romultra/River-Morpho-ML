# st-Swin-UNet Training Infrastructure

Complete training, evaluation, and visualization infrastructure for the **Spatio-Temporal Swin-UNet** (st-Swin-UNet) model for river morphology prediction from satellite imagery.

## Overview

st-Swin-UNet is a Swin Transformer-based U-Net architecture for binary segmentation of temporal satellite imagery sequences. It predicts water/non-water pixels from multi-year satellite data using learnable spatio-temporal patch embeddings.

**Key Features:**
- Swin Transformer encoder-decoder with skip connections
- Learnable temporal position embeddings
- Multiple temporal aggregation strategies
- Two model variants (tiny and small)
- Efficient caching system for fast iteration
- Comprehensive visualization tools for predictions

**Achieved Performance (Tiny Variant):**
- **Best F1 Score:** 0.697 at epoch 42
- **Best CSI:** ~0.53
- **Accuracy:** ~93%
- **Training on RTX 4090:** ~1-2 min/epoch

## Quick Reference

**Train Tiny Variant (6.8M params):**
```bash
python -m swin-unet.train
```

**Train Small Variant (11M params):**
```bash
python -m swin-unet.train --variant small --batch-size 4
```

**Evaluate Best Checkpoint:**
```bash
# Evaluate tiny variant (default)
python -m swin-unet.eval_all_checkpoints

# Evaluate small variant
python -m swin-unet.eval_all_checkpoints --variant small
```

**Visualize Predictions:**
```bash
# Visualize tiny variant predictions
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny/stswin_tiny_epoch042.pt

# Visualize small variant predictions
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_small/stswin_small_epoch042.pt \
    --variant small
```

**Plot Metrics:**
```bash
# Plot tiny variant metrics (default)
python -m swin-unet.plot_score

# Plot small variant metrics
python -m swin-unet.plot_score --variant small
```

---

## Quick Start

### 1. Configure the Model

Edit `config.py` to set your preferences:

```python
# Model variant: "tiny" (faster) or "small" (better performance)
model_cfg.variant = "tiny"

# Temporal aggregation: how to combine temporal frames
model_cfg.temporal_aggregation = "concat_proj"  # or "learned_weighted_sum", "mean"

# Training parameters
train_cfg.num_epochs = 50
train_cfg.lr = 1e-4
data_cfg.batch_size = 8
```

### 2. Train the Model

**Train Tiny Variant (default):**
```bash
python -m swin-unet.train
```

**Train Small Variant:**
```bash
python -m swin-unet.train --variant small --batch-size 4
```

**Advanced training options:**
```bash
# Custom learning rate and epochs
python -m swin-unet.train --lr 5e-5 --epochs 100

# Use learning rate scheduler
python -m swin-unet.train --use-scheduler

# Different temporal aggregation
python -m swin-unet.train --temporal-aggregation learned_weighted_sum

# Combine multiple options
python -m swin-unet.train --variant small --batch-size 4 \
    --lr 1e-4 --epochs 50 --use-scheduler
```

This will:
- Load and cache the dataset (first run may take a while)
- Train the model for the specified number of epochs
- Save checkpoints to `swin-unet/checkpoints_{variant}/`
- Print training and validation metrics

### 3. Evaluate All Checkpoints

After training, evaluate all saved checkpoints on the test set:

```bash
# Evaluate tiny variant (default)
python -m swin-unet.eval_all_checkpoints

# Evaluate small variant
python -m swin-unet.eval_all_checkpoints --variant small
```

This generates a CSV file at `swin-unet/scores/test_metrics_all_epochs_stswin_{variant}.csv` with metrics for each epoch.

**Important:** The script automatically uses the correct checkpoint directory and output paths based on the `--variant` argument. Tiny and small model checkpoints are kept completely separate.

### 4. Visualize Metrics

Create plots and find the best-performing epoch:

```bash
# Plot tiny variant metrics (default)
python -m swin-unet.plot_score

# Plot small variant metrics
python -m swin-unet.plot_score --variant small
```

This will:
- Print best epochs for each metric (F1, CSI, loss, etc.)
- Save summary CSV to `swin-unet/plots/best_epoch_summary_stswin_{variant}.csv`
- Generate plots: loss vs epoch, F1/CSI vs epoch
- Save plots to `swin-unet/plots/` with variant-specific filenames

**Important:** The script automatically reads the correct CSV file based on the `--variant` argument. Plots for tiny and small models are saved with different filenames to prevent overwriting.

### 5. Visualize Predictions

See actual model predictions on test samples:

```bash
# Visualize tiny variant predictions
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny/stswin_tiny_epoch042.pt \
    --num-samples 5

# Visualize small variant predictions
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_small/stswin_small_epoch042.pt \
    --variant small \
    --num-samples 5
```

This will:
- Load the specified checkpoint
- Generate predictions on test samples
- Create comprehensive visualizations showing:
  - Input temporal sequence (first, middle, last frames + temporal mean)
  - Ground truth target
  - Model prediction (probabilities and binary)
  - Error map highlighting incorrect pixels
- Save high-resolution images to `swin-unet/plots/` with variant-specific filenames
- Print per-sample metrics (accuracy, precision, recall, F1, IoU)

**Important:** The `--variant` argument ensures that visualization filenames include the model variant (e.g., `prediction_tiny_test_sample_001.png` vs `prediction_small_test_sample_001.png`) to prevent overwriting when comparing models.

---

## Managing Tiny and Small Variants Separately

All scripts now properly handle tiny and small model variants as completely separate entities. This allows you to train, evaluate, and compare both variants simultaneously without any risk of overwriting data.

### Automatic Variant-Specific Paths

When you specify `--variant tiny` or `--variant small`, each script automatically uses the correct paths:

| Component | Tiny Variant | Small Variant |
|-----------|--------------|---------------|
| **Checkpoints** | `checkpoints_tiny/stswin_tiny_epoch*.pt` | `checkpoints_small/stswin_small_epoch*.pt` |
| **Evaluation CSV** | `scores/test_metrics_all_epochs_stswin_tiny.csv` | `scores/test_metrics_all_epochs_stswin_small.csv` |
| **Plot files** | `plots/test_loss_stswin_tiny.png` | `plots/test_loss_stswin_small.png` |
| **Visualizations** | `plots/prediction_tiny_test_sample_*.png` | `plots/prediction_small_test_sample_*.png` |

### Complete Workflow for Both Variants

```bash
# Train both variants
python -m swin-unet.train --variant tiny --epochs 50
python -m swin-unet.train --variant small --batch-size 4 --epochs 50

# Evaluate both variants
python -m swin-unet.eval_all_checkpoints --variant tiny
python -m swin-unet.eval_all_checkpoints --variant small

# Plot metrics for both
python -m swin-unet.plot_score --variant tiny
python -m swin-unet.plot_score --variant small

# Visualize predictions from both
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny/stswin_tiny_epoch042.pt \
    --variant tiny

python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_small/stswin_small_epoch042.pt \
    --variant small
```

All outputs will be stored in separate locations, making it easy to compare the performance of the two variants side-by-side.

---

## Model Architecture

### Overview

st-Swin-UNet is a U-Net style encoder-decoder architecture built on Swin Transformer blocks instead of convolutional layers. It processes temporal satellite imagery sequences to predict future water/land distribution.

```
Input: (B, T, H, W)     T temporal frames of satellite images
         │
         ▼
┌─────────────────────────────────────────┐
│   Spatio-Temporal Patch Embedding       │  ← Learnable temporal position embeddings
│   (B, T, H, W) → (B, H/4, W/4, 48)      │    Temporal aggregation (concat_proj)
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            ENCODER                       │
│  ┌─────────────────────────────────┐    │
│  │ Stage 1: BasicLayer (dim=48)    │────┼──► Skip 1
│  │ 2× SwinTransformerBlock         │    │
│  └─────────────────────────────────┘    │
│              │ PatchMerging (↓2×)        │
│              ▼                           │
│  ┌─────────────────────────────────┐    │
│  │ Stage 2: BasicLayer (dim=96)    │────┼──► Skip 2
│  │ 2× SwinTransformerBlock         │    │
│  └─────────────────────────────────┘    │
│              │ PatchMerging (↓2×)        │
│              ▼                           │
│  ┌─────────────────────────────────┐    │
│  │ Stage 3: BasicLayer (dim=192)   │────┼──► Skip 3
│  │ 2× SwinTransformerBlock         │    │
│  └─────────────────────────────────┘    │
│              │ PatchMerging (↓2×)        │
│              ▼                           │
│  ┌─────────────────────────────────┐    │
│  │ Stage 4: BasicLayer (dim=384)   │    │  ← Bottleneck
│  │ 2× SwinTransformerBlock         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            DECODER                       │
│  ┌─────────────────────────────────┐    │
│  │ PatchExpanding (↑2×)            │◄───┼── Skip 3
│  │ Skip Fusion + BasicLayer        │    │
│  └─────────────────────────────────┘    │
│              │                           │
│              ▼                           │
│  ┌─────────────────────────────────┐    │
│  │ PatchExpanding (↑2×)            │◄───┼── Skip 2
│  │ Skip Fusion + BasicLayer        │    │
│  └─────────────────────────────────┘    │
│              │                           │
│              ▼                           │
│  ┌─────────────────────────────────┐    │
│  │ PatchExpanding (↑2×)            │◄───┼── Skip 1
│  │ Skip Fusion + BasicLayer        │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Final Head                      │
│  ConvTranspose2d (↑2×) → ConvT (↑2×)    │
│  → Conv2d (1×1) → Sigmoid               │
└─────────────────────────────────────────┘
         │
         ▼
Output: (B, 1, H, W)    Binary segmentation mask
```

### Key Components

#### 1. Spatio-Temporal Patch Embedding (`SpatioTemporalPatchEmbed`)

Handles the temporal dimension of input satellite imagery:

```python
Input:  (B, T, H, W)           # T yearly frames
Output: (B, H/4, W/4, embed_dim)  # Spatially downsampled, temporally aggregated
```

**How it works:**
1. Each temporal frame is embedded separately using a shared Conv2d projection
2. Learnable temporal position embeddings are added to each frame
3. Frames are aggregated using one of three strategies:
   - `concat_proj`: Concatenate all frames, project back to embed_dim (most expressive)
   - `learned_weighted_sum`: Learn scalar weights per frame
   - `mean`: Simple average (baseline)

#### 2. Swin Transformer Blocks

Uses torchvision's `SwinTransformerBlock` with **window-based self-attention**:

- Attention computed in local windows (e.g., 8×8 patches)
- Alternating blocks use **shifted windows** for cross-window connections
- Much more memory efficient than global attention

```python
# Window attention: O(H*W * window_size²) instead of O((H*W)²)
# For 1000×500 image with 8×8 windows: 500K × 64 vs 500K × 500K operations
```

#### 3. PatchMerging (Encoder Downsampling)

Reduces spatial resolution by 2× while doubling channels:
```python
Input:  (B, H, W, C)
Output: (B, H/2, W/2, 2C)
```
Concatenates 2×2 neighboring patches and projects to 2C dimensions.

#### 4. PatchExpanding (Decoder Upsampling)

Inverse of PatchMerging - increases resolution by 2×, halves channels:
```python
Input:  (B, H, W, C)
Output: (B, 2H, 2W, C/2)
```

#### 5. Skip Connections

Encoder features are concatenated with decoder features at matching resolutions:
```python
x = torch.cat([upsampled, skip], dim=-1)  # Concatenate along channel dim
x = skip_fusion(x)  # Linear projection back to expected dim
```

### Memory Efficiency: Why Swin-UNet Uses Less VRAM

Despite having **14× more parameters** than TransformerUNet (6.8M vs 500K), st-Swin-UNet uses significantly less GPU memory:

| Model | Parameters | Batch Size | VRAM Usage |
|-------|------------|------------|------------|
| TransformerUNet | 500K | 1 | >12 GB |
| st-Swin-UNet (tiny) | 6.8M | 8 | ~12 GB |

**The reason: Activation memory, not parameter count**

**TransformerUNet's bottleneck:**
```python
# TemporalTransformerBlock processes EVERY PIXEL as a separate sequence
x = x.reshape(B * H * W, T, 1)  # 1 × 1000 × 500 = 500,000 sequences!
x = self.encoder(x)  # Each stores Q, K, V, attention scores for backprop
```

For a 1000×500 image, this creates **500,000 separate transformer sequences**, each storing intermediate activations for backpropagation.

**st-Swin-UNet's efficiency:**
```python
# Window-based attention: only 8×8 = 64 tokens per window
# Attention is local, not global
# Feature maps shrink through encoder (1000×500 → 250×125 → 125×62 → ...)
```

| Aspect | TransformerUNet | st-Swin-UNet |
|--------|-----------------|--------------|
| Attention scope | 500K separate pixel sequences | 64 tokens per 8×8 window |
| Spatial processing | Full resolution in transformer | Hierarchical (resolution shrinks) |
| Activation storage | Massive (all pixel sequences) | Small (window-based, shared) |

### Dimension Flow (Tiny Variant)

For input shape `(B, 4, 1000, 500)` with 4 temporal frames:

| Stage | Output Shape | Channels |
|-------|--------------|----------|
| Input | (B, 4, 1000, 500) | 4 |
| Patch Embed | (B, 250, 125, 48) | 48 |
| Encoder Stage 1 | (B, 250, 125, 48) | 48 |
| After PatchMerging | (B, 125, 62, 96) | 96 |
| Encoder Stage 2 | (B, 125, 62, 96) | 96 |
| After PatchMerging | (B, 62, 31, 192) | 192 |
| Encoder Stage 3 | (B, 62, 31, 192) | 192 |
| After PatchMerging | (B, 31, 15, 384) | 384 |
| Encoder Stage 4 (Bottleneck) | (B, 31, 15, 384) | 384 |
| Decoder Stage 1 | (B, 62, 31, 192) | 192 |
| Decoder Stage 2 | (B, 125, 62, 96) | 96 |
| Decoder Stage 3 | (B, 250, 125, 48) | 48 |
| Final Head | (B, 1, 1000, 500) | 1 |

## Model Variants

### Tiny Variant (Default)
- **Parameters:** 6.8M (6,823,261)
- **Embedding dimension:** 48
- **Depths:** [2, 2, 2, 2]
- **Attention heads:** [3, 6, 12, 24]
- **Window size:** [8, 8]
- **Use case:** Faster training, experimentation, limited GPU memory
- **Memory:** ~8-10GB GPU RAM with batch_size=8
- **Achieved F1:** 0.697 (epoch 42)

### Small Variant
- **Parameters:** ~11M
- **Embedding dimension:** 96
- **Depths:** [2, 2, 6, 2] (Swin-T configuration)
- **Use case:** Maximum performance, sufficient GPU memory
- **Memory:** ~16GB GPU RAM with batch_size=4-8

**Training the small variant:**
```bash
# Recommended settings for small variant
python -m swin-unet.train --variant small --batch-size 4

# With learning rate scheduler (often helps with larger models)
python -m swin-unet.train --variant small --batch-size 4 --use-scheduler
```

**Or edit `config.py` for persistent changes:**
```python
model_cfg.variant = "small"
data_cfg.batch_size = 4  # Reduce for memory constraints
```

## Temporal Aggregation Methods

The st-Swin-UNet uses learnable temporal position embeddings and supports three aggregation strategies:

### 1. concat_proj (Recommended, Default)
- Concatenates all temporal embeddings and projects back to embedding dimension
- Most expressive, allows complex temporal interactions
- Used in video transformers (ViViT)
- Best performance expected

### 2. learned_weighted_sum
- Learns scalar weights for each temporal frame
- Lightweight, adds minimal parameters
- Good for ablation studies

### 3. mean
- Simple averaging across temporal dimension
- Baseline, no additional parameters
- Fastest, least expressive

To change the aggregation method, edit `config.py`:
```python
model_cfg.temporal_aggregation = "concat_proj"  # or "learned_weighted_sum", "mean"
```

## Configuration Options

### Data Configuration (`DataConfig`)
```python
year_target = 5            # Temporal sequence: 4 input years + 1 target
dir_folders = "data/satellite/dataset_month3"
batch_size = 8             # Reduced for larger model
num_workers = 12           # Parallel data loading workers
use_cache = True           # Cache preprocessed tensors
```

### Model Configuration (`ModelConfig`)
```python
variant = "tiny"                          # "tiny" or "small"
temporal_aggregation = "concat_proj"      # Temporal fusion strategy
img_size = (1000, 500)                    # Input image dimensions
num_classes = 1                           # Binary segmentation
```

### Training Configuration (`TrainConfig`)
```python
num_epochs = 50
lr = 1e-4                  # Learning rate (conservative for transformers)
weight_decay = 0.05        # Standard for vision transformers
loss_f = "BCE"             # Loss function: "BCE", "BCE_Logits", or "Focal"
use_scheduler = False      # Optional: cosine annealing scheduler
warmup_epochs = 5          # Warmup epochs if scheduler enabled
```

## Performance & Memory Requirements

### Tiny Variant Performance (Validated on RTX 4090)
| Metric | Value |
|--------|-------|
| **Best F1 Score** | 0.697 (epoch 42) |
| **Best CSI/IoU** | ~0.53 |
| **Accuracy** | ~93% |
| **Precision** | ~67% |
| **Recall** | ~72% |

### Memory & Speed Requirements

| Variant | Batch Size | GPU RAM | Training Speed (RTX 4090) | Checkpoint Size |
|---------|-----------|---------|---------------------------|-----------------|
| Tiny    | 8         | ~8-10GB | ~1-2 min/epoch            | 27 MB           |
| Tiny    | 16        | ~14GB   | ~1 min/epoch              | 27 MB           |
| Small   | 4         | ~12GB   | ~3-4 min/epoch (est.)     | ~44 MB          |
| Small   | 8         | ~20GB   | ~2-3 min/epoch (est.)     | ~44 MB          |

*Note: Times measured on NVIDIA RTX 4090 with cached dataset. First run will be slower due to dataset preprocessing.*

## Advanced Usage

### Training Command-Line Options

The training script supports extensive command-line configuration:

**Available Arguments:**
```
--variant {tiny,small}           Model variant (default: tiny)
--temporal-aggregation {concat_proj,learned_weighted_sum,mean}
--epochs N                       Number of training epochs (default: 50)
--lr FLOAT                       Learning rate (default: 1e-4)
--weight-decay FLOAT            Weight decay (default: 0.05)
--batch-size N                  Batch size (default: 8)
--use-scheduler                 Enable cosine annealing LR scheduler
--warmup-epochs N               Warmup epochs before scheduler (default: 5)
--checkpoint-dir PATH           Custom checkpoint directory
--cpu                           Force CPU training
--num-workers N                 Data loading workers (default: 12)
```

**Common Training Scenarios:**

```bash
# Quick experiment with fewer epochs
python -m swin-unet.train --epochs 10

# Lower learning rate for fine-tuning
python -m swin-unet.train --lr 5e-5

# Train small variant with scheduler
python -m swin-unet.train --variant small --batch-size 4 --use-scheduler

# Ablation study: different temporal aggregation
python -m swin-unet.train --temporal-aggregation mean \
    --checkpoint-dir swin-unet/checkpoints_tiny_mean

# Memory-constrained training
python -m swin-unet.train --batch-size 4 --num-workers 4
```

### Evaluation Command-Line Options

You can override config values via command-line arguments:

**Evaluation:**
```bash
python -m swin-unet.eval_all_checkpoints \
    --checkpoint-dir swin-unet/checkpoints \
    --checkpoint-pattern "stswin_tiny_epoch*.pt" \
    --output-csv swin-unet/scores/custom_metrics.csv \
    --variant tiny \
    --temporal-aggregation concat_proj
```

**CPU-only evaluation:**
```bash
python -m swin-unet.eval_all_checkpoints --cpu
```

### Experiment Tracking

To compare different configurations:

1. Train with variant A:
   ```python
   # config.py
   model_cfg.variant = "tiny"
   model_cfg.temporal_aggregation = "concat_proj"
   train_cfg.ckpt_dir = Path("swin-unet/checkpoints_tiny_concat")
   ```

2. Train with variant B:
   ```python
   # config.py
   model_cfg.variant = "tiny"
   model_cfg.temporal_aggregation = "learned_weighted_sum"
   train_cfg.ckpt_dir = Path("swin-unet/checkpoints_tiny_weighted")
   ```

3. Compare F1/CSI scores from the evaluation CSVs

## Prediction Visualization

The `visualize_predictions.py` script creates comprehensive visualizations of model predictions, making it easy to understand model behavior and identify failure cases.

### What Gets Visualized

Each visualization shows **2 rows × 4 columns**:

**Row 1: Input Temporal Sequence**
- Column 1: First input frame (year 1 of 4)
- Column 2: Middle input frame (year 2 of 4)
- Column 3: Last input frame (year 4 of 4)
- Column 4: Temporal mean across all 4 frames

**Row 2: Predictions & Analysis**
- Column 1: Ground truth target (actual water distribution)
- Column 2: Model prediction probabilities (0-1 values)
- Column 3: Binary prediction (after threshold)
- Column 4: Error map (red pixels = incorrect predictions)

### Basic Usage

```bash
# Visualize 5 test samples with best checkpoint
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt
```

### Advanced Options

**Visualize more samples:**
```bash
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt \
    --num-samples 12  # All test samples
```

**Use validation set instead of test:**
```bash
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt \
    --split val
```

**Adjust classification threshold:**
```bash
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt \
    --threshold 0.6  # More conservative (higher precision)
```

**Compare different epochs:**
```bash
# Early training (epoch 10)
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch010.pt \
    --output-dir swin-unet/plots/epoch10

# Best model (epoch 42)
.venv/Scripts/python.exe -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt \
    --output-dir swin-unet/plots/epoch42
```

### Output

The script generates:
- High-resolution PNG images (~2 MB each) in `swin-unet/plots/`
- Per-sample metrics printed to console:
  - Accuracy, Precision, Recall, F1, IoU/CSI
- Average metrics across all visualized samples

### Example Results

From epoch 42 checkpoint on 5 test samples:
- **Average F1:** 0.6895 ± 0.0401
- **Average Accuracy:** 0.9333 ± 0.0088
- **Average IoU/CSI:** 0.5275 ± 0.0468

### Interpretation Tips

**Good Predictions:**
- Error map mostly dark (few errors)
- Smooth water boundaries
- High F1 score (>0.70)

**Challenging Cases:**
- More red in error map
- Often at temporal boundaries
- Rapid morphological changes
- Lower F1 scores (~0.63-0.65)

## File Structure

```
swin-unet/
├── st_swin_unet_model.py      # Core model architecture (17 KB)
├── config.py                   # Configuration dataclasses (3 KB)
├── train.py                    # Training script (7.6 KB)
├── eval_all_checkpoints.py     # Evaluation script (10.6 KB)
├── plot_score.py               # Metrics visualization (5.6 KB)
├── visualize_predictions.py    # Prediction visualization (12.5 KB)
├── README.md                   # This file (documentation)
├── cache/                      # Dataset cache (6.8 GB total)
│   ├── dataset_month3_training.pt     (6.3 GB)
│   ├── dataset_month3_validation.pt   (229 MB)
│   └── dataset_month3_testing.pt      (229 MB)
├── checkpoints_tiny/           # Tiny variant checkpoints (27 MB each)
│   ├── stswin_tiny_epoch001.pt
│   ├── stswin_tiny_epoch002.pt
│   └── ... (up to epoch 050)
├── checkpoints_small/          # Small variant checkpoints (44 MB each)
│   ├── stswin_small_epoch001.pt
│   ├── stswin_small_epoch002.pt
│   └── ... (up to epoch 050)
├── plots/                      # Visualization outputs
│   ├── test_loss_stswin_tiny.png
│   ├── test_loss_stswin_small.png
│   ├── test_f1_csi_stswin_tiny.png
│   ├── test_f1_csi_stswin_small.png
│   ├── best_epoch_summary_stswin_tiny.csv
│   ├── best_epoch_summary_stswin_small.csv
│   ├── prediction_tiny_test_sample_001.png
│   ├── prediction_small_test_sample_001.png
│   └── ... (prediction visualizations)
└── scores/                     # Evaluation CSVs
    ├── test_metrics_all_epochs_stswin_tiny.csv
    └── test_metrics_all_epochs_stswin_small.csv
```

**Important:** Checkpoints, evaluation results, and visualizations for tiny and small variants are now completely separated to prevent any accidental overwriting. Each variant has its own:
- Checkpoint directory: `checkpoints_{variant}/`
- Evaluation CSV: `scores/test_metrics_all_epochs_stswin_{variant}.csv`
- Plot files with variant in filename: `test_loss_stswin_{variant}.png`
- Prediction visualizations: `prediction_{variant}_{split}_sample_*.png`

## Code Reuse

This implementation maximizes code reuse from `transformer_cnn_model`:

**Shared utilities (no modifications):**
- `transformer_cnn_model/preprocessing/load_data.py` - Data loading
- `transformer_cnn_model/preprocessing/dataset_generation.py` - Dataset creation
- `transformer_cnn_model/train_eval_functions/train_eval.py` - Training/validation functions

This ensures:
- Consistent metrics across all models
- Fair comparison with baseline models
- Easier maintenance (bug fixes benefit all models)

## Comparison with TransformerUNet

| Aspect | TransformerUNet | st-Swin-UNet (Tiny) |
|--------|----------------|---------------------|
| Backbone | CNN (U-Net) | Vision Transformer (Swin) |
| Temporal handling | Per-pixel temporal transformer | Spatio-temporal patch embedding |
| Parameters | ~500K | 6.8M |
| Attention scope | Global (per pixel across time) | Local (8×8 windows) |
| Memory (batch=1) | >12 GB | ~2 GB |
| Memory (batch=8) | OOM | ~12 GB |
| Training Speed (RTX 4090) | ~30 sec/epoch | ~1-2 min/epoch |
| **F1 Score** | ~0.60-0.70 (est.) | **0.697** (verified) |
| **Accuracy** | ~85-90% (est.) | **93%** (verified) |
| Checkpoint Size | ~2 MB | 27 MB |

**Why st-Swin-UNet uses less memory despite 14× more parameters:**

TransformerUNet's `TemporalTransformerBlock` reshapes input to `(B×H×W, T, 1)`, creating 500,000 separate sequences for a 1000×500 image. Each sequence stores activations (Q, K, V, attention scores) for backpropagation, consuming massive GPU memory.

st-Swin-UNet uses window-based attention (64 tokens per window) and hierarchical feature maps that shrink through the encoder, dramatically reducing activation memory. See [Memory Efficiency](#memory-efficiency-why-swin-unet-uses-less-vram) for details.

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `data_cfg.batch_size` (try 4 or 2)
- Use "tiny" variant instead of "small"
- Reduce `data_cfg.num_workers` if RAM is limited

### Slow Training
- Enable caching: `data_cfg.use_cache = True`
- Increase `data_cfg.batch_size` if memory allows
- Increase `data_cfg.num_workers` for faster data loading

### Model Not Learning
- Check learning rate (try 5e-5 to 2e-4)
- Enable scheduler: `train_cfg.use_scheduler = True`
- Try different temporal aggregation methods
- Check data normalization and loss function

## Citation

If you use this code, please cite:

```
st-Swin-UNet: Spatio-Temporal Swin Transformer U-Net for
River Morphology Prediction from Satellite Imagery
```

## License

See repository root for license information.
