# Dataset Card: Wisconsin Breast Cancer (Diagnostic)

## 1. Dataset Overview
- **Dataset Name:** Wisconsin Breast Cancer Diagnostic (WDBC)
- **Source:** UCI ML Repository via sklearn.datasets.load_breast_cancer
- **Task:** Binary Disease Classification
- **Sample Count:** 569 instances
- **Feature Count:** 30 continuous medical features
- **Target Variable:** target (0=malignant, 1=benign)

## 2. Class Distribution
- **Malignant (Class 0):** 212 (37.3%)
- **Benign (Class 1):** 357 (62.7%)
- **Imbalance Ratio:** 1.684 (Benign : Malignant)

## 3. Data Integrity & Types
- **Missing Values:** 0
- **Duplicated Rows:** 0
- **Feature Data Types:** all continuous (float64)
- **Value Ranges:** Min 0.0000 to Max 4254.0000

## 4. Clinical Features
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass:
- radius, texture, perimeter, area, smoothness
- compactness, concavity, concave points, symmetry, fractal dimension
(Each measured across mean, standard error, and worst/largest values).

## 5. Usage & Research Safety
This dataset is used solely for developing and benchmarking an experimental hybrid classical-quantum machine learning prototype. Not approved for direct clinical diagnostics.
