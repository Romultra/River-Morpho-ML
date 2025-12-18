# Pretrained st-Swin-UNet Models

This directory contains the best-performing st-Swin-UNet checkpoints, selected based on validation F1 scores following proper machine learning methodology.

## Available Models

### Tiny Variant (4-year input)
- **File:** `stswin_tiny_4y_best.pt`
- **Epoch:** 33
- **Parameters:** ~6.8M
- **Validation F1:** 0.665
- **Test Performance:**
  - F1 Score: 0.7038
  - CSI/IoU: 0.5431
  - Accuracy: 92.8%
  - Precision: 66.4%
  - Recall: 74.9%

### Small Variant (4-year input)
- **File:** `stswin_small_4y_best.pt`
- **Epoch:** 14
- **Parameters:** ~41.3M
- **Validation F1:** 0.668
- **Test Performance:**
  - F1 Score: 0.7044
  - CSI/IoU: 0.5438
  - Accuracy: 92.9%
  - Precision: 67.1%
  - Recall: 74.2%

## Usage

### Quick Test

Test a pretrained model on the test set:

```bash
# Test tiny variant
python -m swin-unet.test_pretrained --variant tiny

# Test small variant
python -m swin-unet.test_pretrained --variant small
```

### Test on CPU

Force CPU usage (useful if CUDA is not available):

```bash
python -m swin-unet.test_pretrained --variant tiny --cpu
```

### Custom Checkpoint

Load a specific checkpoint file:

```bash
python -m swin-unet.test_pretrained --checkpoint path/to/your/checkpoint.pt --variant tiny
```

### Generate Visualizations

To visualize predictions, use the dedicated visualization script:

```bash
python -m swin-unet.visualize_predictions \
    --checkpoint swin-unet/pretrained/stswin_tiny_4y_best.pt \
    --variant tiny --temporal-frames 4 --num-samples 5
```

## Model Architecture

Both models use:
- **Patch size:** 4×4
- **Window-based self-attention:** 8×8 (tiny), 7×7 (small)
- **Temporal aggregation:** concat_proj
- **Encoder/decoder stages:** 4 hierarchical stages
- **Learnable temporal position embeddings**
- **Training configuration:**
  - Learning rate: 1e-4
  - Weight decay: 0.05
  - Loss: Binary Cross-Entropy
  - Batch size: 8 (tiny), 4 (small)

## Checkpoint Selection Methodology

These checkpoints were selected using proper ML evaluation methodology:

1. All 50 epoch checkpoints were evaluated on the **validation set**
2. Best checkpoint selected based on **validation F1 score**
3. Selected checkpoint evaluated **once** on the **test set**

This prevents test set leakage and provides an unbiased performance estimate.

## File Size

- `stswin_tiny_4y_best.pt`: ~26 MB
- `stswin_small_4y_best.pt`: ~159 MB

## Performance Comparison

Compared to baseline models:
- **JamUNet (CNN baseline):** F1 = 0.722 ⭐ **Best**
- **st-Swin-UNet Small:** F1 = 0.7044 (2.4% behind baseline)
- **st-Swin-UNet Tiny:** F1 = 0.7038 (2.5% behind baseline)
- **TransformerUNet:** F1 = 0.691 (4.3% behind baseline)

While st-Swin-UNet does not surpass the simple CNN baseline, it demonstrates:
- More memory-efficient attention mechanism than per-pixel transformers
- Hierarchical feature extraction with window-based attention
- Better performance than TransformerUNet (+1.9% F1)

## Citation

If you use these pretrained models, please cite:

```
Improving River Morphology Prediction with Transformer-Based Architectures
EPFL Machine Learning Course (CS-433), Fall 2025
https://github.com/Romultra/River-Morpho-ML
```

And acknowledge the original JamUNet work:
```bibtex
@mastersthesis{magherini2024,
  author = {Magherini, A.},
  title = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
  school = {{Delft University of Technology}},
  year = {2024}
}
```
