# River-Morpho-ML Repository Summary

## 📋 Overview

**Project Title:** JamUNet - Predicting Morphological Changes of Braided Sand-Bed Rivers with Deep Learning

**Author:** Antonio Magherini (MSc Civil Engineering - Hydraulic Engineering, TU Delft)

**Affiliation:** Delft University of Technology & Deltares

**Purpose:** This repository contains the complete codebase, data, and documentation for a Master's thesis focused on predicting river morphological changes using deep learning. The project specifically targets the Brahmaputra-Jamuna River system using satellite imagery and hydraulic data.

**License:** MIT License (Copyright 2024)

**Repository Size:** ~4.7 GB (primarily satellite imagery data)

**Thesis:** Available at [TU Delft Repository](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348)

---

## 🏗️ Repository Structure

```
River-Morpho-ML/
├── benchmarks/          # Benchmark models for comparison
├── data/                # Raw satellite images and river variables (~3.8 GB)
├── images/              # Thesis report images and visualizations
├── model/               # Deep learning model (JamUNet/UNet3D)
├── other/               # Additional documents (MSc thesis PDF)
├── postprocessing/      # Results analysis and visualization
├── preliminary/         # Exploratory analysis notebooks
├── preprocessing/       # Data preprocessing modules
├── braided.yml          # Conda environment specification
├── LICENSE              # MIT License
├── README.md            # Main documentation
└── markdownlint.json    # Markdown linting configuration
```

---

## 📂 Directory Details

### 1. **`data/` Directory** (~3.8 GB)

Contains all raw data used in the project:

#### Subdirectories:
- **`satellite/`** (~3.8 GB) - Satellite imagery from Google Earth Engine
  - `original/` (333 MB) - Original satellite images
  - `preprocessed/` (1.3 GB) - Preprocessed images
  - `averages/` (2.0 GB) - Averaged images
  - `dataset/` (34 MB) - Training/validation/testing datasets
  - `dataset_month1/` to `dataset_month4/` - Monthly dataset splits
  
- **`flow/`** - River hydraulic data (Excel files)
  - Daily water levels at Bahadurabad (1964-1994)
  - Daily water levels at Sirajganj (2006-2022)
  - Biweekly discharge and maximum velocity data (1990-2016)
  - Flood water level data (2003-2022)

- **`qgis/`** - Geographic information system data
  - Shapefiles of river reaches
  - Past river interventions in lower Jamuna River
  - QGIS project file (`jamuna_areas.qgz`)

- **`bathymetry_1992.csv`** - Surveyed bathymetry from 1992 River Survey Project

**Key Data Sources:**
- Satellite images: Google Earth Engine (USGS Landsat 5, JRC Global Surface Water)
- Collection: JRC_GSW1_4_MonthlyHistory
- Study area: Brahmaputra-Jamuna River (India-Bangladesh border)

---

### 2. **`model/` Directory** (Core Deep Learning Implementation)

Contains the JamUNet (UNet3D) deep learning model implementation.

#### Key Files:

**Python Modules (32 total in repository):**
- **`st_unet/st_unet.py`** (142 lines) - Main model architecture
  - `DoubleConv`: 2D + 3D convolution blocks
  - `Down`: Downsampling with pooling
  - `Up`: Upsampling with skip connections
  - `UNet3D`: Complete U-Net architecture with temporal 3D convolution
  
- **`train_eval.py`** - Training and validation functions
  - `get_predictions()`: Model inference
  - `training_unet()`: Training loop with customizable loss functions
  - Supports physics-induced loss terms
  - Binary classification for water/non-water pixels
  
- **`additional_losses.py`** - Additional loss functions (experimental)

**Notebooks (14 total in repository):**
- `Unet3D_train_val_spatial_dataset.ipynb` - Spatial dataset training
- `Unet3D_train_val_temporal_dataset.ipynb` - Temporal dataset training
- `trained_Unet3D_spatial_dataset.ipynb` - Testing spatial model
- `trained_Unet3D_temporal_dataset.ipynb` - Testing temporal model

**Subdirectories:**
- **`models_trained/`** - Saved model weights (.pth files, ~2.1 MB each)
  - 4 trained models with different configurations
  - Naming convention: `UNet3D_{spatial/temporal}_{bloss/brecall}_month3_...`
  
- **`scalers/`** - Normalization scalers for training datasets
  
- **`losses_metrics/`** - CSV files tracking training progress

**Model Architecture:**
- Input: 4 consecutive yearly images (shape: n_samples × 4 × 1000 × 500)
- Output: Binary prediction (water/non-water classification)
- Architecture: U-Net with 4 downsampling layers, 4 upsampling layers
- Special feature: Temporal 3D convolution layer
- Configurable: kernel size, pooling type, initial hidden dimensions, dropout

---

### 3. **`preprocessing/` Directory**

Data preparation and dataset generation modules.

#### Files (~1,200 lines total):
- **`dataset_generation.py`** - Core dataset creation functions
  - `load_image_array()`: Load satellite images using GDAL
  - `create_dir_list()`: Organize training/validation/testing data
  - `create_list_images()`: Generate image file lists
  - Pixel class scaling: no-data (-1), non-water (0), water (1)
  
- **`images_analysis.py`** - Image loading and analysis functions
  
- **`river_analysis_pre.py`** - Hydraulic data preprocessing
  
- **`satellite_analysis_pre.py`** - Satellite image preprocessing
  - `count_pixels()`: Pixel statistics
  - `load_avg()`: Load averaged images

**Data Processing Pipeline:**
1. Load raw satellite images (JRC collection, grayscale)
2. Scale pixel values: 0→-1 (no-data), 1→0 (non-water), 2→1 (water)
3. Organize into temporal sequences (4 consecutive years)
4. Split into training/validation/testing datasets
5. Apply normalization scalers

---

### 4. **`postprocessing/` Directory**

Results analysis, visualization, and metrics computation (~3,000 lines total).

#### Files:
- **`metrics.py`** - Performance metrics computation
  - `compute_metrics()`: Accuracy, precision, recall, F1, CSI
  - `single_roc_curve()`: ROC curve for individual samples
  - `get_total_roc_curve()`: Average ROC curve across dataset
  - Binary classification with configurable water threshold (default: 0.5)
  - Positive class: water (1), Negative class: non-water (0)
  
- **`plot_results.py`** - Visualization functions for model predictions
  
- **`save_figures.py`** - Figure export utilities
  
- **`save_results.py`** - Results export (losses, metrics, etc.)
  
- **`satellite_analysis_post.py`** - Post-analysis of satellite images
  
- **`river_analysis_post.py`** - Hydraulic data post-analysis (to be added)

**Key Metrics:**
- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall
- CSI (Critical Success Index): Threat score
- ROC AUC: Area under receiver operating characteristic curve
- Average Precision Score: Precision-recall curve metric

---

### 5. **`benchmarks/` Directory**

Comparison models for validation.

#### Subdirectories:
- **`jagers_nn/`** - Neural network model by Jagers (2003)
  - `data_metrics_plots_nn.py`: Data loading and visualization
  - `data_neural_network.ipynb`: Benchmark notebook
  - Uses NetCDF format (.nc files)
  
- **`no_change/`** - "No-change" baseline benchmark
  - `nochange.py`: Baseline prediction (assumes no change)
  - `nochange_scenario.ipynb`: Baseline testing

**Purpose:** Provide baseline comparisons to evaluate JamUNet performance improvements.

---

### 6. **`preliminary/` Directory**

Exploratory data analysis and preprocessing experiments.

#### Notebooks:
1. **`satellite_img_visualization.ipynb`** - Visualize satellite images across years/locations
2. **`satellite_collections.ipynb`** - Analyze available image collections (JRC, Landsat, Sentinel)
3. **`river_data_analysis.ipynb`** - Analyze discharge, water level, flow velocity
4. **`edit_satellite_img.ipynb`** - Image preprocessing experiments
5. **`create_dataset_temporal.ipynb`** - Temporal dataset creation
6. **`create_dataset_monthly.ipynb`** - Monthly dataset creation
7. **`create_dataset_seasons.ipynb`** - Seasonal dataset creation
8. **`copy_images_for_dataset.ipynb`** - Dataset organization

**Purpose:** Initial exploration before finalizing the preprocessing pipeline and model training approach.

---

### 7. **`images/` Directory**

Contains images used in the thesis report and documentation.

- Example: `1994-01-25.png` - Brahmaputra-Jamuna River satellite image from Landsat 5

---

### 8. **`other/` Directory**

Additional project documentation.

- **`MSc_AntonioMagherini.pdf`** (40 MB) - Complete Master's thesis document

---

## 🔧 Technical Stack

### Environment: Conda (`braided.yml`)

**Python Version:** 3.9.17

**Key Dependencies:**
- **Deep Learning:**
  - PyTorch 2.0.1 (CPU-only version)
  - TorchVision
  - Torchinfo 1.8.0
  
- **Scientific Computing:**
  - NumPy 1.24.2
  - SciPy 1.9.3
  - Pandas 1.5.3
  - Scikit-learn 1.2.2
  
- **Visualization:**
  - Matplotlib 3.5.2
  - Seaborn 0.12.2
  
- **Geospatial:**
  - GDAL (for satellite image processing)
  
- **Jupyter:**
  - Jupyter 1.0.0
  - JupyterLab 3.4.4
  - ipympl 0.9.3 (interactive matplotlib)
  
- **Other:**
  - Dash 2.13.0 (dashboards)
  - Dask (parallel computing)
  - tqdm 4.66.1 (progress bars)
  - openpyxl (Excel file handling)
  - tabulate (table formatting)

**Installation:**
```bash
conda env create -f braided.yml
conda activate braided
```

---

## 🔄 Typical Workflow

### 1. **Data Preparation**
- Download satellite images from Google Earth Engine
- Preprocess images (scaling, cropping, formatting)
- Organize into temporal sequences
- Split into train/validation/test sets

**Tools:** `preprocessing/` modules, `preliminary/` notebooks

### 2. **Model Training**
- Configure UNet3D architecture
- Set hyperparameters (learning rate, batch size, epochs)
- Train on spatial or temporal datasets
- Monitor losses and metrics

**Tools:** `model/Unet3D_train_val_*.ipynb`, `model/train_eval.py`

### 3. **Model Evaluation**
- Load trained model weights
- Run predictions on test set
- Compute metrics (accuracy, F1, CSI, ROC AUC)
- Generate visualizations

**Tools:** `model/trained_Unet3D_*.ipynb`, `postprocessing/` modules

### 4. **Benchmark Comparison**
- Run baseline models (no-change, Jagers NN)
- Compare metrics with JamUNet
- Generate comparison plots

**Tools:** `benchmarks/` modules and notebooks

### 5. **Results Analysis**
- Analyze prediction quality
- Study failure cases
- Generate publication-ready figures
- Document findings

**Tools:** `postprocessing/plot_results.py`, `postprocessing/save_figures.py`

---

## 📊 Key Project Features

### 1. **Spatiotemporal Modeling**
- Uses 4 consecutive yearly images as input
- Predicts morphological change for year 5
- Incorporates temporal dynamics through 3D convolution

### 2. **Binary Classification**
- Water vs. non-water pixel classification
- Threshold-based predictions (default: 0.5)
- Handles no-data regions (value: -1)

### 3. **Multiple Dataset Configurations**
- Spatial datasets: Different geographic reaches
- Temporal datasets: Different time periods
- Monthly datasets: Seasonal variations
- Organized by reach ID and time period

### 4. **Flexible Model Architecture**
- Configurable depth (number of downsampling layers)
- Adjustable kernel sizes
- Optional dropout for regularization
- Choice of pooling types (max/average)
- Bilinear or transposed convolution upsampling

### 5. **Comprehensive Evaluation**
- Multiple metrics (accuracy, precision, recall, F1, CSI)
- ROC and precision-recall curves
- Per-sample and aggregate statistics
- Comparison with benchmark models

---

## 🎯 Important Code Components

### Model Architecture (UNet3D)
```python
UNet3D(
    n_channels=4,          # 4 input years
    n_classes=1,           # Binary output
    init_hid_dim=8,        # Initial hidden dimensions
    kernel_size=3,         # Convolution kernel
    pooling='max',         # Pooling type
    bilinear=False,        # Upsampling method
    drop_channels=False,   # Dropout toggle
    p_drop=None           # Dropout probability
)
```

### Key Functions:
- **Training:** `training_unet(model, loader, optimizer, ...)`
- **Prediction:** `get_predictions(model, input_dataset, device)`
- **Metrics:** `compute_metrics(pred, target, water_threshold)`
- **Dataset:** `load_image_array(path, scaled_classes=True)`

---

## 📝 File Naming Conventions

### Satellite Images:
- Format: `{collection}_{location}_{date}.tif`
- Example: `JRC_GSW1_4_MonthlyHistory_r1_training_1990.tif`

### Trained Models:
- Format: `UNet3D_{type}_{criterion}_{config}.pth`
- Example: `UNet3D_spatial_bloss_month3_4dwns_8ihiddim_3ker_maxpool_0.05ilr_15step_0.75gamma_16batch_100epochs_0.5wthr.pth`
- Components:
  - type: spatial/temporal
  - criterion: bloss (best loss) / brecall (best recall)
  - dwns: number of downsampling layers
  - ihiddim: initial hidden dimension
  - ker: kernel size
  - ilr: initial learning rate
  - wthr: water threshold

---

## 🚀 Getting Started

### Prerequisites:
1. Anaconda or Miniconda installed
2. ~5 GB free disk space
3. CUDA-capable GPU recommended (but CPU-only version available)

### Setup Steps:
```bash
# 1. Clone repository
git clone https://github.com/Romultra/River-Morpho-ML.git
cd River-Morpho-ML

# 2. Create environment
conda env create -f braided.yml

# 3. Activate environment
conda activate braided

# 4. Launch Jupyter
jupyter lab

# 5. Open a notebook (e.g., model/trained_Unet3D_spatial_dataset.ipynb)
```

### Quick Test:
Open `preliminary/satellite_img_visualization.ipynb` to visualize sample satellite images and verify the environment is working correctly.

---

## 📈 Project Statistics

- **Total Python files:** 32
- **Total Jupyter notebooks:** 14
- **Total lines of code:** ~4,869 (Python modules only)
- **Trained models:** 4 configurations
- **Data size:** 4.7 GB total
  - Satellite data: ~3.8 GB
  - Thesis PDF: 40 MB
  - Other: ~800 MB
- **Image dimensions:** 1000 × 500 pixels (typical)
- **Temporal sequence length:** 4 years input, 1 year prediction

---

## 🔍 Research Context

### Study Area:
- **River:** Brahmaputra-Jamuna River
- **Location:** India-Bangladesh border
- **Characteristics:** Braided sand-bed river with high morphological activity
- **Period:** 1990s - 2020s

### Scientific Contribution:
- Novel application of deep learning to river morphology prediction
- Spatiotemporal modeling approach for braided rivers
- Comparison with traditional methods (Jagers NN, no-change baseline)
- Integration of satellite imagery and hydraulic data

### Thesis Information:
- **Title:** JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning
- **Author:** Antonio Magherini
- **Institution:** TU Delft (Faculty of Civil Engineering and Geosciences)
- **Collaboration:** Deltares
- **Year:** 2024
- **Track:** MSc Civil Engineering - Hydraulic Engineering (River Engineering specialization)

---

## 📧 Contact

**Author:** Antonio Magherini  
**Email:** antonio.magherini@gmail.com  
**LinkedIn:** [antonio-magherini-4349b2229](https://nl.linkedin.com/in/antonio-magherini-4349b2229)

---

## 📚 Citation

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

---

## 🔐 Git Configuration

- **Ignored files:** Python cache files (`*.pyc`, `__pycache__/`), Jupyter checkpoints (`.ipynb_checkpoints/`)
- **No GitHub Actions:** No CI/CD workflows configured
- **No test suite:** Project focused on research/analysis rather than software testing

---

## 💡 Tips for Working with This Repository

1. **Start with visualization:** Use `preliminary/satellite_img_visualization.ipynb` to understand the data
2. **Understand the pipeline:** Review `preprocessing/dataset_generation.py` before training
3. **Check trained models:** Examine `model/models_trained/` for available pre-trained weights
4. **Monitor training:** Watch `model/losses_metrics/` for convergence during training
5. **GPU setup:** If using GPU, modify device strings from 'cuda:0' to 'cuda' if needed
6. **Memory management:** The datasets are large; ensure adequate RAM (16+ GB recommended)
7. **GDAL dependency:** Ensure GDAL is properly configured for image loading
8. **Temporal sequences:** Remember that inputs are 4 consecutive years (not arbitrary time spans)

---

## ⚠️ Known Limitations & TODOs

From README files across the repository:
- Some preliminary notebooks marked "to be added soon"
- `postprocessing/river_analysis_post.py` - to be added soon
- `postprocessing/satellite_analysis_post.py` - to be added soon
- QGIS data marked "to be updated soon"
- Images folder "to be added soon"

---

## 🎓 Academic Context

This is a **research repository** for an academic thesis, not production software. Therefore:
- No formal test suite
- No CI/CD pipelines
- Documentation focused on research methodology
- Code structured for reproducibility rather than deployment
- Emphasis on exploratory analysis and experimentation

The codebase is designed to support the scientific findings in the thesis, enable reproducibility, and facilitate future research on river morphology prediction using deep learning.

---

**Last Updated:** November 2024  
**Repository Status:** Complete (thesis submitted)  
**Maintenance:** Active for research collaboration

---

*This summary provides a comprehensive overview to enable efficient work on the River-Morpho-ML project. For specific implementation details, refer to individual module documentation and Jupyter notebooks.*
