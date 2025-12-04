# Quick Start Guide - River-Morpho-ML

**For the comprehensive summary, see [REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md)**

---

## 🚀 5-Minute Quick Start

### What is this project?
Deep learning model (JamUNet/UNet3D) to predict river morphology changes using satellite imagery of the Brahmaputra-Jamuna River.

### Conda Setup
```bash
# 1. Install environment
conda env create -f braided.yml
conda activate braided

# 2. Launch Jupyter
jupyter lab
```

### Pip Install (Alternative)
Works on Windows with **Python 3.9** and **CUDA 11.8**
```bash
# 1. Create and activate virtual environment with Python 3.9
python3.9 -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # macOS/Linux

# 2. Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install Custom GDAL Wheel
pip install ./GDAL-3.9.2-cp39-cp39-win_amd64.whl

# 4. Install other dependencies
pip install -r requirements.txt
```


### Key Directories
| Directory | Purpose | Size |
|-----------|---------|------|
| `data/satellite/` | Satellite imagery datasets | 3.8 GB |
| `model/` | JamUNet deep learning model | - |
| `preprocessing/` | Data preparation modules | - |
| `postprocessing/` | Results analysis & metrics | - |
| `benchmarks/` | Comparison models | - |
| `preliminary/` | Exploratory notebooks | - |

---

## 📖 Where to Start?

### For Understanding the Data:
- **Notebook:** `preliminary/satellite_img_visualization.ipynb`
- **Purpose:** Visualize satellite images and understand data structure

### For Training a Model:
- **Notebook:** `model/Unet3D_train_val_spatial_dataset.ipynb` or `model/Unet3D_train_val_temporal_dataset.ipynb`
- **Module:** `model/train_eval.py`

### For Testing a Trained Model:
- **Notebook:** `model/trained_Unet3D_spatial_dataset.ipynb` or `model/trained_Unet3D_temporal_dataset.ipynb`
- **Weights:** `model/models_trained/*.pth`

### For Understanding the Code:
- **Model Architecture:** `model/st_unet/st_unet.py` (142 lines)
- **Dataset Creation:** `preprocessing/dataset_generation.py`
- **Metrics:** `postprocessing/metrics.py`

---

## 🎯 Common Tasks

### Load a Satellite Image
```python
from preprocessing.dataset_generation import load_image_array

img = load_image_array('path/to/image.tif', scaled_classes=True)
# Returns: numpy array with values -1 (no-data), 0 (non-water), 1 (water)
```

### Create the Model
```python
from model.st_unet.st_unet import UNet3D

model = UNet3D(
    n_channels=4,      # 4 input years
    n_classes=1,       # Binary output
    init_hid_dim=8,    # Initial hidden dimensions
    kernel_size=3,     # Convolution kernel
    pooling='max'      # Max pooling
)
```

### Compute Metrics
```python
from postprocessing.metrics import compute_metrics

accuracy, precision, recall, f1, csi = compute_metrics(
    pred=predictions,
    target=targets,
    water_threshold=0.5
)
```

---

## 📊 Model Input/Output

- **Input:** 4 consecutive yearly satellite images (shape: batch × 4 × 1000 × 500)
- **Output:** Binary prediction for year 5 (shape: batch × 1 × 1000 × 500)
- **Classes:** 0 = non-water, 1 = water
- **Threshold:** 0.5 (adjustable)

---

## 🔧 Key Python Dependencies

- PyTorch 2.0.1
- NumPy 1.24.2
- GDAL (for satellite images)
- Matplotlib 3.5.2
- Scikit-learn 1.2.2
- Jupyter Lab 3.4.4

---

## 📝 File Naming Convention

### Trained Models:
```
UNet3D_{spatial/temporal}_{bloss/brecall}_month3_4dwns_8ihiddim_3ker_maxpool_0.05ilr_15step_0.75gamma_16batch_100epochs_0.5wthr.pth
```
- `spatial/temporal`: Dataset type
- `bloss/brecall`: Best loss or best recall
- `4dwns`: 4 downsampling layers
- `8ihiddim`: Initial hidden dimension = 8
- `3ker`: Kernel size = 3
- `0.05ilr`: Initial learning rate = 0.05
- `0.5wthr`: Water threshold = 0.5

---

## 💡 Tips

1. **Memory:** The datasets are large; ensure 16+ GB RAM
2. **GPU:** Code assumes `cuda:0`; modify to `cuda` if needed
3. **GDAL:** Must be properly configured for image loading
4. **Temporal sequences:** Always 4 consecutive years as input
5. **Pre-trained models:** Available in `model/models_trained/`

---

## 📚 Additional Resources

- **Full Summary:** [REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md)
- **Thesis PDF:** `other/MSc_AntonioMagherini.pdf`
- **Thesis Online:** https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348
- **Contact:** antonio.magherini@gmail.com

---

## 🔍 Project Stats

- **Python files:** 32
- **Jupyter notebooks:** 14
- **Trained models:** 4 configurations
- **Total repository size:** 4.7 GB
- **Lines of code:** ~4,869 (Python modules)

---

**Need more details?** See the comprehensive [REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md) document!
