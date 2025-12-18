# Improving River Morphology Prediction with Transformer-Based Architectures

**EPFL Machine Learning Course (CS-433) - Project 2: Machine Learning for Science**
*Fall 2025*

<table>
  <tr>
    <td>
      <img src=".\images\1994-01-25.png" width="1000" alt="Brahmaputra-Jamuna River">
    </td>
    <td>
      <p style="font-size: 16px;">
        This project builds upon the <a href="https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348">JamUNet MSc thesis</a> by Antonio Magherini to investigate whether transformer-based architectures can improve deep learning predictions of braided river morphology from temporal satellite imagery.
      </p>
      <p style="font-size: 16px;">
        We design and evaluate two novel spatio-temporal architectures—<strong>TransformerUNet</strong> and <strong>st-Swin-UNet</strong>—that explicitly model temporal dependencies to predict year-to-year morphological changes of the Brahmaputra-Jamuna River.
      </p>
    </td>
  </tr>
</table>

---

## Project Overview

### Problem Statement

Braided sand-bed rivers like the Brahmaputra undergo complex morphological changes driven by path-dependent hydrological processes. The original **JamUNet** model (Magherini, 2024) demonstrated that CNNs can predict future water distribution from temporal satellite imagery, but its reliance on 3D convolutions provides only shallow temporal receptive fields, limiting its ability to capture long-range temporal evolution such as progressive bar migration or cumulative erosion.

### Our Contribution

We investigate whether **transformer-based architectures** can overcome these limitations by explicitly modeling temporal dependencies. Our approach:

1. **TransformerUNet**: Adds a temporal transformer block before the U-Net encoder to learn long-range dependencies across yearly observations
2. **st-Swin-UNet**: Replaces the entire CNN backbone with a Swin Transformer-based U-Net that uses learnable spatio-temporal patch embeddings

**Key Findings:**
- **JamUNet baseline achieves best performance with F1 = 0.722** despite being the simplest architecture
- TransformerUNet performs worse than baseline (F1: 0.691 vs 0.722), likely due to limited training data
- st-Swin-UNet achieves F1 = 0.704 (small variant), falling between TransformerUNet and baseline but not surpassing the simple CNN
- Window-based attention (Swin) is more memory-efficient than per-pixel temporal transformers but doesn't improve accuracy
- All models face the challenge of limited temporal training data (~700 images over 30 years)

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - 5-minute setup and first model training
- **[swin-unet/README.md](swin-unet/README.md)** - Comprehensive st-Swin-UNet documentation
- **[Report/main.tex](Report/main.tex)** - Academic report with detailed methodology and results

---

## Repository Structure

```
River-Morpho-ML/
├── model/                      # Original UNet3D (JamUNet) baseline
│   ├── st_unet/st_unet.py     # UNet3D architecture
│   └── models_trained/         # Baseline checkpoints
│
├── transformer_cnn_model/      # TransformerUNet implementation
│   ├── model_architecture.py  # TransformerUNet + TemporalTransformerBlock
│   ├── config.py              # Hyperparameter configuration
│   ├── train.py               # Training script
│   ├── eval_all_checkpoints.py
│   ├── preprocessing/         # Data loading with caching
│   └── checkpoints_*/         # Model checkpoints
│
├── swin-unet/                  # st-Swin-UNet implementation
│   ├── st_swin_unet_model.py  # Swin Transformer U-Net architecture
│   ├── config.py              # Model/training configuration
│   ├── train.py               # Training script
│   ├── eval_all_checkpoints.py
│   ├── visualize_predictions.py
│   ├── plot_score.py
│   └── README.md              # Detailed st-Swin-UNet documentation
│
├── preprocessing/              # Original data preprocessing (GDAL)
├── postprocessing/            # Metrics and visualization tools
│   ├── metrics.py             # Binary classification metrics
│   └── plot_results.py
│
├── data/                      # Satellite imagery and auxiliary data
├── benchmarks/                # Baseline comparison models
├── Report/                    # LaTeX report and figures
│   ├── main.tex
│   └── ML Course Project 2 description and guidelines.pdf
│
├── braided.yml                # Conda environment (legacy models)
├── requirements.txt           # pip requirements (transformer models)
└── CLAUDE.md                  # Project context for Claude Code
```

---

## Quick Start

### Environment Setup

**Option 1: Conda**
```bash
conda env create -f braided.yml
conda activate braided
```

**Option 2: pip**

Works on Windows with **Python 3.9** and **CUDA 11.8**

```bash
# 1. Create and activate virtual environment with Python 3.9
python3.9 -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # macOS/Linux

# 2. Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install Custom GDAL Wheel (Windows only - for baseline JamUNet data loading)
pip install ./GDAL-3.9.2-cp39-cp39-win_amd64.whl

# 4. Install other dependencies
pip install -r requirements.txt
```

**Note:** For macOS/Linux, install GDAL using your system package manager before step 4.

### Training Models

**1. TransformerUNet**
```bash
# Configure in transformer_cnn_model/config.py
python -m transformer_cnn_model.train

# Evaluate all checkpoints
python -m transformer_cnn_model.eval_all_checkpoints

# Plot training metrics
python -m transformer_cnn_model.plot_score
```

**2. st-Swin-UNet**

**Quick Start - Test Pretrained Models:**
```bash
# Test pretrained tiny variant (no training needed!)
python -m swin-unet.test_pretrained --variant tiny

# Test pretrained small variant
python -m swin-unet.test_pretrained --variant small

# Test on CPU
python -m swin-unet.test_pretrained --variant tiny --cpu
```

Pretrained checkpoints are in `swin-unet/pretrained/` and include the best tiny and small variants selected from validation F1 scores.

For visualizations, use: `python -m swin-unet.visualize_predictions --checkpoint swin-unet/pretrained/stswin_tiny_4y_best.pt --variant tiny --temporal-frames 4`

**Training from Scratch:**
```bash
# Train tiny variant with 4-year input (6.8M params, batch size 8)
python -m swin-unet.train --variant tiny --temporal-frames 4

# Train small variant with 4-year input (41.3M params, batch size 4)
python -m swin-unet.train --variant small --batch-size 4 --temporal-frames 4

# Evaluate all checkpoints on VALIDATION set (for model selection)
python -m swin-unet.eval_all_checkpoints --variant tiny --temporal-frames 4 --split val

# Plot validation metrics
python -m swin-unet.plot_score --variant tiny --temporal-frames 4 --split val

# Final test evaluation (unbiased, single evaluation of best checkpoint)
python -m swin-unet.final_test_eval --variant tiny --temporal-frames 4

# Visualize predictions from best checkpoint
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny_4y/stswin_tiny_4y_epoch033.pt \
    --variant tiny --temporal-frames 4 --num-samples 5
```

See [swin-unet/README.md](swin-unet/README.md) for comprehensive st-Swin-UNet documentation.

---

## Model Architectures

### 1. JamUNet (Baseline) ⭐ **Best Performance**
**Original thesis model** by Magherini (2024)
- U-Net with 4 encoder/decoder stages
- 3D convolution at bottleneck for temporal fusion
- 2D+3D convolution blocks (DoubleConv)
- ~500K parameters

**Results (4 input years):**
- **F1 Score: 0.722** (epoch 8) - **BEST across all models**
- **Accuracy: 93.5%**
- **Precision: 75.3%**
- **Recall: 77.0%**

**Why it works:** Despite lacking explicit long-range temporal modeling, the shallow 3D convolutions combined with strong spatial features are sufficient for this dataset. Outperforms both transformer approaches, suggesting that model complexity doesn't always improve performance with limited training data.

### 2. TransformerUNet
**Hybrid CNN-Transformer architecture**
- Temporal transformer block processes each pixel's time series independently (B×H×W sequences of length T)
- Multi-head self-attention with 2 encoder layers (d_model=8, nhead=4, dim_feedforward=64)
- Standard U-Net backbone (4 encoder/decoder stages with 2D convolutions)
- ~500K parameters total
- Chunked processing (max 60,000 tokens per batch to avoid transformer batch limits)

**Architecture Flow:**
```
Input (B×T×H×W) → TemporalTransformer → U-Net → Output (B×H×W)
```

**Results (4 input years):**
- F1 Score: 0.691 (epoch 5)
- Accuracy: 93.4%
- Precision: ~90%
- Recall: ~70%

**Results (9 input years):**
- F1 Score: 0.708 (epoch 5) - slight improvement
- Accuracy: 93.6%

**Key Finding:** TransformerUNet performs **slightly worse** than JamUNet baseline (F1: 0.691 vs 0.722), likely due to:
- Insufficient training data (~700 images) for transformer to learn temporal patterns effectively
- CNN component dominates training, transformer provides minimal benefit
- Linear projection back to scalar per timestep may limit temporal information flow

### 3. st-Swin-UNet
**Swin Transformer-based U-Net**
- Learnable spatio-temporal patch embeddings with temporal position encoding
- Window-based self-attention (8×8 windows for tiny, 7×7 for small) instead of global attention
- Hierarchical feature maps with PatchMerging/PatchExpanding
- Factory functions: `create_swin_unet_tiny()` (6.8M params), `create_swin_unet_small()` (41.3M params)

**Architecture Flow:**
```
Input (B×T×H×W)
  → SpatioTemporalPatchEmbed (temporal aggregation)
  → Swin Encoder (4 stages, window attention)
  → Swin Decoder (skip connections, patch expanding)
  → Final Head
  → Output (B×1×H×W)
```

**Results (Small Variant, 4 input years):**
- **F1 Score:** 0.7044 (epoch 14, selected by validation F1=0.6682)
- **Accuracy:** 92.9%
- **CSI/IoU:** 0.5438
- **Precision:** 67.1% (lower than baseline's 75.3%)
- **Recall:** 74.2% (lower than baseline's 77.0%)
- **Memory:** ~13GB GPU RAM (batch size 4)
- **Training Speed:** 2-4 min/epoch on RTX 4090

**Results (Tiny Variant, 4 input years):**
- **F1 Score:** 0.7038 (epoch 33, selected by validation F1=0.6652)
- **Accuracy:** 92.8%
- **CSI/IoU:** 0.5431
- **Precision:** 66.4%
- **Recall:** 74.9%
- **Memory:** 8-10GB GPU RAM (batch size 8)

**Architecture Characteristics:**
- 14-83× more parameters than TransformerUNet (6.8M-41.3M vs 500K) but uses less memory due to window-based attention
- Hierarchical spatial processing (1000×500 → 250×125 → 125×62 → ...)
- More memory-efficient than per-pixel transformers, but does not outperform simple CNN baseline

**Evaluation Methodology:** All st-Swin-UNet results use **proper validation-based checkpoint selection** followed by single test set evaluation to avoid data leakage.

---

## Performance Comparison

All results below are for **4 input years** (base configuration):

| Model | Parameters | F1 Score | Accuracy | Precision | Recall | Best Epoch (Val) |
|-------|------------|----------|----------|-----------|--------|------------------|
| **JamUNet** (baseline) | ~500K | **0.722** | 93.5% | 75.3% | 77.0% | 8 |
| **TransformerUNet** | ~500K | 0.691 | 93.4% | 90.0% | 70.0% | 5 |
| **st-Swin-UNet (Tiny)** | 6.8M | 0.7038 | 92.8% | 66.4% | 74.9% | 33 |
| **st-Swin-UNet (Small)** | 41.3M | **0.7044** | **92.9%** | 67.1% | 74.2% | 14 |

**With 9 input years:**
- **JamUNet**: F1 = 0.715, Accuracy = 93.7%
- **TransformerUNet**: F1 = 0.708, Accuracy = 93.6%
- **st-Swin-UNet (Tiny)**: F1 = 0.7019, Accuracy = 93.0% (epoch 20)
- **st-Swin-UNet (Small)**: F1 = 0.6976, Accuracy = 92.9% (epoch 30)

**Key Insights:**
- **JamUNet baseline is the best model** (F1: 0.722) despite being the simplest architecture
- st-Swin-UNet (F1: 0.7044) outperforms TransformerUNet (F1: 0.691) but falls 2.5% short of baseline
- Small variant (41.3M params) provides minimal improvement over tiny variant (6.8M params): F1 0.7044 vs 0.7038
- Both transformer approaches fail to surpass the simple CNN, suggesting transformers need more training data
- **Adding more input years (4→9) hurts performance** for st-Swin-UNet: tiny drops from 0.7038 to 0.7019, small drops from 0.7044 to 0.6976
- All models plateau around 93-94% accuracy, suggesting fundamental data quality/quantity limitations
- **Proper validation-based model selection** prevents test set leakage and provides unbiased performance estimates

---

## Data

**Source:** JRC Global Surface Water collection (Landsat 5 satellite imagery)
**Study Area:** Brahmaputra-Jamuna River (India-Bangladesh border)
**Temporal Coverage:** 30 years (~700 images), January-April window
**Spatial Resolution:** 1000×500 pixels per image
**Task:** Binary water/non-water pixel classification

**Preprocessing:**
- Pixel scaling: `0 → -1` (no-data), `1 → 0` (non-water), `2 → 1` (water)
- Temporal sequences: 4-9 consecutive yearly frames → 1 target year
- Train/validation/test split based on spatial reaches

---

## Key Configuration

All models use similar training configurations for fair comparison:

```python
# Data
year_target = 10           # 4 input years + 1 target (or 9 + 1)
batch_size = 8             # Varies by model
img_size = (1000, 500)

# Training
num_epochs = 50
lr = 1e-4
loss_f = "BCE"             # Binary cross-entropy
weight_decay = 0.05        # For transformers

# Metrics
threshold = 0.5            # Water classification threshold
```

Model-specific configurations in respective `config.py` files:
- `transformer_cnn_model/config.py` - TransformerUNet
- `swin-unet/config.py` - st-Swin-UNet

---

## Evaluation & Visualization

### Metrics
All models evaluated on:
- **F1 Score** (primary metric for imbalanced data)
- **Critical Success Index (CSI)**
- **Accuracy, Precision, Recall**
- **ROC AUC**

### Visualization Tools

**TransformerUNet:**
```bash
python -m transformer_cnn_model.plot_score
python -m transformer_cnn_model.plot_misclassification
```

**st-Swin-UNet:**
```bash
# Validation metrics over epochs (for model selection)
python -m swin-unet.plot_score --variant tiny --temporal-frames 4 --split val

# Final test evaluation (unbiased single evaluation)
python -m swin-unet.final_test_eval --variant tiny --temporal-frames 4

# Prediction visualizations with error maps
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/checkpoints_tiny_4y/stswin_tiny_4y_epoch033.pt \
    --variant tiny --temporal-frames 4 --num-samples 5
```

---

## Findings & Discussion

### Surprising Result: Simple CNN Wins
Despite our hypothesis that transformers would improve temporal modeling, **JamUNet baseline (F1=0.722) outperforms both transformer architectures**:
- TransformerUNet (F1=0.691) underperforms by 4.3%
- st-Swin-UNet tiny (F1=0.7038) underperforms by 2.5%
- st-Swin-UNet small (F1=0.7044) underperforms by 2.4%
- Neither transformer approach justifies the added architectural complexity
- Adding more input years (4→9) **hurts st-Swin-UNet performance**: tiny drops 0.27% (0.7038→0.7019), small drops 0.96% (0.7044→0.6976)

### Why Transformers Didn't Help
1. **Insufficient training data** - ~700 images over 30 years is too small for transformers to learn complex temporal patterns
2. **CNN dominates training** - In TransformerUNet, the U-Net backbone dominates, transformer provides minimal benefit
3. **Linear projection bottleneck** - TransformerUNet's final projection back to scalar per timestep may limit temporal information flow
4. **Data quality limitations** - Only January-April imagery available (cloud cover), missing critical flood season dynamics
5. **Overfitting with larger models** - st-Swin-UNet's 41.3M parameters (small variant) show minimal improvement over 6.8M (tiny variant): F1 0.7044 vs 0.7038, only +0.0006
6. **More temporal context hurts** - Extended 9-year input performs worse than 4-year, suggesting noise/training difficulty increases with longer sequences

### What We Learned
1. **Simpler is sometimes better** - JamUNet's shallow 3D convolutions outperform complex attention mechanisms with limited data
2. **Window-based attention** (Swin) is more memory-efficient than per-pixel temporal transformers, enabling larger models
3. **Transformers need more data** - ~700 training images is insufficient for transformers to learn complex temporal patterns
4. **Architecture complexity doesn't guarantee improvement** - More parameters (6.8M-41.3M vs 500K) and sophisticated mechanisms don't overcome data limitations
5. **st-Swin-UNet outperforms TransformerUNet** - Window-based attention (F1=0.7044) performs better than per-pixel temporal attention (F1=0.691), though both fall short of the CNN baseline
6. **Diminishing returns with model size** - 6× more parameters (41.3M vs 6.8M) yields only +0.0006 F1 improvement
7. **4-year temporal window is optimal** - Extended 9-year history degrades performance, suggesting limited benefit from distant temporal context
8. **Proper evaluation methodology matters** - Validation-based checkpoint selection prevents test set leakage and provides unbiased estimates

### Future Work
1. **Data augmentation** - Incorporate more river systems for transfer learning
2. **Physics-informed constraints** - Benchmark against hydrodynamic simulations
3. **Temporal data expansion** - Include flood season imagery despite cloud challenges
4. **Cross-platform testing** - Improve Linux/macOS compatibility for HPC deployment

---

## Ethical Considerations

**Stakeholders:** Communities living along Brahmaputra River banks, civil engineers, flood management authorities

**Primary Risk:** Misinterpretation of probabilistic predictions as deterministic, leading to:
- Over-preparation (resource waste)
- Under-preparation (endangering lives)

**Mitigation Strategies:**
1. Focus on **high recall and F1 scores** rather than accuracy alone
2. Emphasize that predictions are probabilistic, not certain
3. **Do not deploy models directly** - benchmark against physics-based fluid simulations first
4. Acknowledge that ML models lack physical constraints of computational fluid dynamics

See `Report/main.tex` Section 1 for detailed ethical risk assessment.

---

## Citations

### Original JamUNet Thesis
```bibtex
@mastersthesis{magherini2024,
  author = {Magherini, A.},
  title = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
  school = {{Delft University of Technology}},
  year = {2024},
  month = {10},
  howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
```

### This Project
If you use our transformer-based extensions, please cite this repository and acknowledge the EPFL ML course:

```
Improving River Morphology Prediction with Transformer-Based Architectures
EPFL Machine Learning Course (CS-433), Fall 2025
https://github.com/Romultra/River-Morpho-ML
```

---

## Authors

This project was developed by:
- Ziyang He (`ziyang.he@epfl.ch`)
- Romeo Estezet (`romeo.estezet@epfl.ch`)
- Capucine Denis (`capucine.denis@epfl.ch`)

---

## Acknowledgments

- **Antonio Magherini** - Original JamUNet thesis and codebase foundation
- **EPFL Machine Learning Course (CS-433)** - Project framework and guidance
- **TU Delft & Deltares** - Original research collaboration
- **Google Earth Engine** - Satellite imagery data (Landsat 5)

---

## License

See [LICENSE](LICENSE) file for details.


