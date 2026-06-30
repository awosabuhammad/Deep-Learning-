# Automated Multimodal Crop Disease Diagnosis

## Overview

This is a Deep Learning course project for crop disease diagnosis using multimodal remote sensing images.

The project is based on the Kaggle / ICPR 2026 competition:

**Beyond Visible Spectrum: AI for Agriculture 2026**

The goal is to classify crop health conditions using three image modalities:

- **RGB**: normal color images
- **MS**: multispectral images
- **HS**: hyperspectral images

The target classes are:

- Health
- Other
- Rust

## Project Objective

The main objective is to build and compare deep learning models that classify crop health conditions using different remote sensing modalities.

The project includes:

1. Dataset structure inspection
2. Matching RGB, MS, and HS samples
3. Label extraction from filenames
4. Stratified train-validation split
5. RGB preprocessing
6. MS preprocessing
7. HS preprocessing
8. Baseline CNN model training
9. Fusion model experiments
10. Model evaluation using accuracy, precision, recall, F1-score, and confusion matrices
11. Kaggle submission generation

## Dataset Structure

The notebook expects the dataset folder to be organized like this:

```text
Kaggle_Prepared/
├── train/
│   ├── HS/
│   ├── MS/
│   └── RGB/
└── val/
    ├── HS/
    ├── MS/
    └── RGB/
```

The image folders are not included in this repository to keep the project lightweight.

## Preprocessing Summary

### RGB Images

- Loaded using PIL
- Converted to RGB
- Resized to `224 x 224 x 3`
- Normalized to the range `0 - 1`

### MS Images

- Loaded from `.tif` files
- Shape: `64 x 64 x 5`
- Min-max normalization applied

### HS Images

- Loaded from `.tif` files
- Standardized to `32 x 32 x 125`
- Some images had 126 bands, so the first 125 bands were kept

## Models Tested

| Model | Validation Accuracy |
|---|---:|
| RGB Baseline | 51.67% |
| MS Baseline | 60.83% |
| MS + HS Early Fusion | 30.83% |
| RGB + MS Late Fusion | 60.83% |
| Improved MS Model | 27.50% |

## Best Model

The best-performing model was the **MS Baseline CNN**, with a validation accuracy of:

```text
60.83%
```

This result shows that multispectral data was more useful than RGB data in this experiment because it contains spectral information that is not visible in normal images.

## Results Preview

### Confusion Matrices

![Confusion Matrices](assets/confusion_matrices.png)

### Kaggle Submission Result

![Kaggle Submission Result](assets/kaggle_submission_result.png)

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- tifffile
- Jupyter Notebook

## Project Files

| File | Description |
|---|---|
| `project.ipynb` | Main notebook containing the full implementation |
| `Automated_Multimodal_Crop_Disease_Diagnosis_Report.pdf` | Final project report |
| `requirements.txt` | Required Python libraries |
| `train_split.csv` | Training split metadata |
| `val_split.csv` | Validation split metadata |
| `submission.csv` | Kaggle submission file |
| `assets/` | Report preview images |
| `data/README.md` | Dataset instructions |

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Place the dataset folder beside the notebook:

```text
Kaggle_Prepared/
```

Then open and run:

```text
project.ipynb
```

## Notes

- The full image dataset is not included because it may be too large for a lightweight GitHub repository.
- Empty dataset folders are included only to show the required structure.
- The fusion models did not improve the result in this experiment.
- Future improvements could include attention-based fusion, dimensionality reduction for HS bands, and cross-validation.

## Author

**Awos Abuhammad**
