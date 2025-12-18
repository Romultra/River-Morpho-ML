# River Morphological Changes Prediction (EPFL Fall 2025)

---

<table>
  <tr>
    <td>
      <img src="./images/1994-01-25.png" width="000" alt="Brahmaputra-Jamuna River">
    </td>
    <td valign="bottom" style="text-align:left;">
      <em>
        The image represents the Brahmaputra-Jamuna River at the border between India and Bangladesh. 
        The image was taken on January 25, 1994. It was retrieved from 
        <a href="https://earthengine.google.com/">Google Earth Engine</a> 
        <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2">USGS Landsat 5 collection</a>.
      </em>
    </td>
  </tr>
</table>

---

## Project overview
This project implements a **machine learning pipeline** to predict the **morphological changes in the Brahmaputra-Jamuna River** using **satellite image analysis**. The images are extracted from the **[Global Surface Water Dataset (GSWD)](https://global-surface-water.appspot.com/download)** introduced by **[Pekel et al. (2016)](https://doi.org/10.1038/nature20584)**. 

The project aims to improve the **[JamUNet model](https://github.com/antoniomagherini/jamunet-morpho-braided)** developed by **[A. Magherini](https://people.epfl.ch/antonio.magherini)** by integrating a **transformer-based architecture** to better capture temporal dependencies in braided river morphology.

For details on the original JamUNet model, please refer to [A. Magherini (2024)](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348) : *JamUNet : predicting the morphological changes of braided sand-bed rivers with deep learning*.


## Authors
This project was developed by :
- Ziyang He (`ziyang.he@epfl.ch`)
- Romeo Estezet (`romeo.estezet@epfl.ch`)
- Capucine Denis (`capucine.denis@epfl.ch`)

---

### Acknowledgments
This project is built on the foundational work of **A. Magherini**, whose **JamUNet model** provided the basis for this project. We are grateful for his contributions and guidance.

---
## Repository structure

The structure of this repository is the following:
- <code>benchmarks</code>, contains modules and notebooks of the benchmark models used for comparison;
- <code>data</code>, contains raw data (satellite images, river variables);
- <code>images</code>, contains the images shown in the thesis report and other documents; (to be added soon)
- <code>model</code>, contains the modules and noteboooks with the JamUNet deep-learning model;
- <code>other</code>, contains documents, images, and other files used during the project;
- <code>postprocessing</code>, contains the modules used for the data postprocessing;
- <code>preliminary</code>, contains the notebooks with the preliminary data analysis, satellite image visualization, preprocessing steps, and other examples;
- <code>preprocessing</code>, contains the modules used for the data preprocessing.
- <code>swin-unet</code>, contains Swin-Unet deep learning model with accompanying modules and notebooks;
- <code>transformer_cnn_model</code>, contains Transformer-CNN deep learning model with accompanying modules and notebooks.

---

## Documentation

For detailed information about this repository:

- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick start guide
- **[REPOSITORY_SUMMARY.md](REPOSITORY_SUMMARY.md)** - Comprehensive project documentation

---

## Requirements
External libraries are required to run code. To install them, run in command line:

```bash
python -m venv .venv
source .venv/bin/activate     
pip install -r requirements.txt
```
---

## Install dependencies

<code>braided.yml</code> is the environment file with all dependencies, needed to run the notebooks.

To activate the environment follow these steps:

- make sure to have the file <code>braided.yml</code> in your system (for Windows users, store it in <code>C:\Windows\System32</code>);
- open the anaconda prompt;
- run <code>conda env create -f braided.yml</code>;
- verify that the environment is correctly installed by running <code>conda env list</code> and checking the environment exists;
- activate the environment by running <code>conda activate braided</code>;
- deactivate the environment by running <code>conda deactivate</code>;

---

## Transformer_CNN model
To run the Transformer_CNN model, inside the <code>transformer_cnn_model</code> folder:
- In <code>config</code>, select the model's architecture and parameters;
- In <code>train_eval</code>, choose the loss function and physical parameters for binary classification;
- Run <code>train</code>. 

---

## Swin-UNet model
To run the Swin-UNet model, please follow step by step the README in the <code>swin-unet</code> folder.

---

## Cite

Please cite [A. Magherini's Master thesis](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348) as:

```
@mastersthesis{magherini2024,
author = {Magherini, A.},
title = {{JamUNet: predicting the morphological changes of braided sand-bed rivers with deep learning}},
school = {{Delft University of Technology}},
year = {2024},
month = {10},
howpublished = {\url{https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348}}
}
