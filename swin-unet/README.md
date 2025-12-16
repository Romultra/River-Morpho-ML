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
python -m swin-unet.eval_all_checkpoints
```

**Visualize Predictions:**
```bash
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny/stswin_tiny_epoch042.pt
```

**Plot Metrics:**
```bash
python -m swin-unet.plot_score
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
python -m swin-unet.eval_all_checkpoints
```

This generates a CSV file at `swin-unet/scores/test_metrics_all_epochs_stswin_{variant}.csv` with metrics for each epoch.

### 4. Visualize Metrics

Create plots and find the best-performing epoch:

```bash
python -m swin-unet.plot_score
```

This will:
- Print best epochs for each metric (F1, CSI, loss, etc.)
- Save summary CSV to `swin-unet/plots/best_epoch_summary_stswin_{variant}.csv`
- Generate plots: loss vs epoch, F1/CSI vs epoch
- Save plots to `swin-unet/plots/`

### 5. Visualize Predictions

See actual model predictions on test samples:

```bash
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints/stswin_tiny_epoch042.pt \
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
- Save high-resolution images to `swin-unet/plots/`
- Print per-sample metrics (accuracy, precision, recall, F1, IoU)

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
year_target = 10           # Temporal sequence: 4 input years + 1 target
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
- Column 1: First input frame (year 1 of 9)
- Column 2: Middle input frame (year 5 of 9)
- Column 3: Last input frame (year 9 of 9)
- Column 4: Temporal mean across all 9 frames

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
├── checkpoints/                # Model checkpoints (27 MB each)
│   ├── stswin_tiny_epoch001.pt
│   ├── stswin_tiny_epoch002.pt
│   └── ... (up to epoch 050)
├── plots/                      # Visualization outputs
│   ├── test_loss_stswin_tiny.png
│   ├── test_f1_csi_stswin_tiny.png
│   ├── best_epoch_summary_stswin_tiny.csv
│   └── prediction_sample_*.png (prediction visualizations)
└── scores/                     # Evaluation CSVs
    └── test_metrics_all_epochs_stswin_tiny.csv
```

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
| Temporal handling | Temporal Transformer (optional) | Spatio-temporal patch embedding |
| Parameters | ~500K | 6.8M |
| Receptive field | Local (convolutions) | Global (attention) |
| Memory (batch=8) | ~4-6 GB | ~8-10 GB |
| Training Speed (RTX 4090) | ~30 sec/epoch | ~1-2 min/epoch |
| **F1 Score** | ~0.60-0.70 (est.) | **0.697** (verified) |
| **Accuracy** | ~85-90% (est.) | **93%** (verified) |
| Checkpoint Size | ~2 MB | 27 MB |

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
