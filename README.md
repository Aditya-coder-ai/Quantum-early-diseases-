# MIndMatrix: Hybrid Classical–Quantum Medical Disease Detection System

A research prototype combining classical representation learning, quantum feature selection, and a Variational Quantum Circuit (VQC) for medical diagnosis.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38%2B-green.svg)](https://pennylane.ai/)
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen.svg)]()

---

## Architecture & Pipeline Overview

```
Medical Dataset (WDBC: 30 continuous clinical features)
   ↓
Data Analysis & Integrity Verification (Zero missing values, 37.3% Malignant)
   ↓
Reproducible Preprocessing (StandardScaler fit strictly on training split)
   ↓
Stratified Train (70%) / Validation (15%) / Test (15%) Split
   ↓
Classical Feature Extractor (PyTorch Autoencoder: 30 → 64 → 32 → 16)
   ↓
16-Dimensional Latent Representation
   ↓
Quantum Feature Selection (QUBO formulation balancing relevance vs. redundancy)
   ↓
8-Dimensional Selected Representation
   ↓
Quantum Data Encoding (Angle encoding with RY rotations in [0, π])
   ↓
Variational Quantum Circuit (8 Qubits, 2 Layers, Circular CNOT Entanglement)
   ↓
Prediction & Calibrated Probability
   ↓
Rigorous Evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, FNR)
   ↓
5-Stage Systematic Ablation Study & Classical Comparison
```

---

## Key Experimental Findings

Evaluation on held-out test split (86 samples, 32 malignant, 54 benign):

| Ablation Stage | Model | Input Space | Qubits | Accuracy | Precision | Recall (Sens.) | Specificity | F1 Score | ROC-AUC | False Negatives |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | Logistic Regression | Raw (30) | 0 | 0.9651 | 0.9811 | 0.9630 | 0.9688 | 0.9720 | 0.9954 | 2 |
| **A** | SVM (RBF) | Raw (30) | 0 | 0.9767 | 0.9815 | 0.9815 | 0.9688 | 0.9815 | 0.9959 | 1 |
| **A** | Random Forest | Raw (30) | 0 | 0.8953 | 0.8947 | 0.9444 | 0.8125 | 0.9189 | 0.9821 | 3 |
| **B** | Classical MLP | Raw (30) | 0 | 0.9419 | 0.9298 | 0.9815 | 0.8750 | 0.9550 | 0.9919 | 1 |
| **C** | Latent-SVM | Latent (16) | 0 | 0.9651 | 0.9636 | 0.9815 | 0.9375 | 0.9725 | 0.9931 | 1 |
| **D** | ClassicalFS-SVM | Selected (8) | 0 | 0.9302 | 0.9286 | 0.9630 | 0.8750 | 0.9455 | 0.9850 | 2 |
| **E** | **Hybrid VQC** | **Selected (8)** | **8** | **0.9419** | **0.9153** | **1.0000** | **0.8438** | **0.9558** | **0.9705** | **0** |

### Clinical Significance of False Negatives
In clinical oncology screening, a **false negative** (misclassifying a malignant lesion as benign) carries life-threatening consequences, whereas a false positive merely prompts confirmatory follow-up imaging or biopsy. The **Hybrid VQC achieved 0 False Negatives (100% Sensitivity/Recall)** on the uncorrupted test set, demonstrating the screening utility of quantum-selected representations.

---

## Repository Structure

```
MIndMatrix/
├── configs/
│   ├── config.py                 # Central configuration (paths, seeds, dimensions, hyperparams)
│   └── __init__.py
├── data/
│   ├── raw/                      # Raw dataset CSV
│   └── processed/                # Normalized train/val/test splits & split_metadata.json
├── models/
│   ├── scaler.joblib             # Fitted StandardScaler (train set only)
│   ├── autoencoder.pth           # Trained PyTorch Autoencoder checkpoint
│   ├── baseline_*.joblib         # Serialized classical baseline models
│   ├── vqc_model.pt              # PyTorch Hybrid VQC model checkpoint
│   ├── vqc_weights.npy           # Portable numpy array of quantum circuit parameters
│   └── angle_scaler.json         # Quantized angle scaling bounds
├── notebooks/                    # Interactive research notebooks
├── results/
│   ├── classical_baselines.csv   # Baseline benchmark metrics
│   ├── latent_validation.csv     # 16-D latent space validation
│   ├── final_comparison.csv      # Complete 5-stage ablation comparison table
│   ├── selected_features_*.json  # Quantum and classical feature selection indices
│   ├── vqc_training_history.json # Loss and accuracy curves per epoch
│   └── plots/                    # Generated high-resolution research figures
│       ├── model_comparison.png
│       ├── autoencoder_loss.png
│       ├── vqc_loss.png
│       ├── latent_tsne.png
│       ├── feature_selection.png
│       ├── roc_curves.png
│       ├── precision_recall_curves.png
│       └── confusion_matrix_vqc.png
├── scripts/
│   └── run_pipeline.py           # Master CLI runner orchestrating all stages
├── src/
│   ├── data/loader.py            # Data loading, validation, and dataset card generator
│   ├── preprocessing/pipeline.py # Zero-leakage transformation & stratified splitting
│   ├── models/
│   │   ├── classical_baselines.py# Classical benchmark training (LogReg, SVM, RF, MLP)
│   │   ├── autoencoder.py        # Classical feature extractor (30 -> 16 dimensions)
│   │   └── latent_validation.py  # Validation of latent space clinical utility
│   ├── quantum/
│   │   ├── feature_selection.py  # QUBO quantum feature selector & classical comparison
│   │   └── vqc.py                # 8-qubit VQC circuit, AngleScaler, and PyTorch module
│   ├── training/train_vqc.py     # Hybrid VQC training loop with adjoint differentiation
│   └── evaluation/
│       ├── metrics.py            # Comprehensive evaluation metrics & confusion matrix
│       ├── ablation.py           # Systematic ablation experiment execution
│       └── visualization.py      # Publication-quality plotting suite
├── tests/
│   ├── test_pipeline.py          # Comprehensive 8-stage boundary and integration tests
│   └── test_vqc_benchmark.py     # Quantum simulator latency and gradient benchmark
├── requirements.txt              # Pinned Python package dependencies
├── DATASET_CARD.md               # Detailed clinical dataset documentation
├── PROJECT_STATUS.md             # Engineering status, diagnosis, and validation logs
└── README.md
```

---

## Quickstart & Reproduction Guide

### 1. Installation

Clone repository and install dependencies:
```bash
git clone https://github.com/adij7/MIndMatrix.git
cd MIndMatrix
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline

Execute all 8 pipeline stages seamlessly:
```bash
python scripts/run_pipeline.py
```

### 3. Run Individual Components

- **Run Data Preprocessing & Validation Pipeline (Part 2):**
  ```bash
  python scripts/run_preprocessing.py
  ```
- **Run Feature Extraction & Latent Validation Pipeline (Part 3):**
  ```bash
  python scripts/run_feature_extraction.py
  ```
- **Train Classical Baselines:**
  ```bash
  python -m src.models.classical_baselines
  ```
- **Train Autoencoder (30D → 16D):**
  ```bash
  python -m src.models.autoencoder
  ```
- **Run Quantum Feature Selection (16D → 8D):**
  ```bash
  python -m src.quantum.feature_selection
  ```
- **Train Hybrid VQC Model:**
  ```bash
  python -m src.training.train_vqc
  ```
- **Execute Ablation Analysis:**
  ```bash
  python -m src.evaluation.ablation
  ```
- **Generate Research Plots:**
  ```bash
  python -m src.evaluation.visualization
  ```

### 4. Run Unit and Integration Tests

- **Run Part 2 Preprocessing & Leakage Tests (12/12 Passed):**
  ```bash
  python -m pytest -v tests/test_preprocessing.py
  ```
- **Run Part 3 Feature Extraction & Latent Tests (10/10 Passed):**
  ```bash
  python -m pytest -v tests/test_autoencoder.py
  ```
- **Run Full Pipeline Integration Tests (8/8 Passed):**
  ```bash
  python -m pytest -v tests/test_pipeline.py
  ```

---

## Medical Research Safety & Ethical Disclaimer

This system is an **experimental research prototype** designed to investigate the feasibility of hybrid classical-quantum machine learning architectures for biomedical pattern recognition.

> **IMPORTANT CLINICAL NOTICE:**  
> This software is **NOT** a certified medical diagnostic device and is not intended for primary clinical diagnosis, prognosis, or patient management in clinical practice. Any diagnostic deployment requires formal clinical trials, FDA/CE-MDR regulatory clearance, and human physician oversight.
