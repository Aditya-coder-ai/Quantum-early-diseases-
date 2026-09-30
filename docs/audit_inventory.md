# MIndMatrix: Scientific Audit & Numbers Inventory Report

**Date:** September 2026  
**Auditor:** Scientific Integrity & Audit Subsystem  
**Source of Truth Policy:** Code-generated metrics strictly from `results/canonical/`.

---

## 1. Contradictory Metric Inventory

### A. VQC Accuracy Variations
- **Distinct Reported Numbers:** [75.0, 78.12, 79.38, 80.62, 81.25, 84.38, 84.52, 84.71, 85.75, 86.05, 87.0, 87.21, 87.5, 88.84, 89.09, 89.3, 89.47, 89.53, 89.66, 89.91, 90.0, 90.62, 90.7, 90.74, 91.53, 91.86, 91.89, 92.86, 93.02, 93.33, 93.75, 94.19, 94.44, 94.92, 95.13, 95.19, 95.54, 95.56, 95.58, 96.18, 96.23, 96.3, 96.47, 96.49, 96.67, 96.82, 96.88, 97.05, 97.2, 97.25, 97.7, 97.74, 97.78, 98.08, 99.41, 99.53, 99.63, 99.71, 99.88, 99.93]
- **Root Cause:**
  - `94.19%`: Single-seed test accuracy on held-out test split (Seed 42, 0 FN).
  - `90.70%`: 5-seed cross-evaluation mean before threshold calibration.
  - `93.02%`: Validation-tuned threshold score on specific validation partition.
  - `90.00%`: Truncated summary figure in early drafts.
- **Remediation:** Enforce Canonical Evaluation Protocol (Phase 1) with 25 outer folds (5x5 repeated stratified CV) reporting Mean ± 95% CI. No single-seed headline claims.

### B. Classical SVM Baseline Variations
- **Distinct Reported Numbers:** [81.25, 82.5, 87.5, 90.41, 90.62, 90.7, 91.86, 92.31, 92.73, 92.86, 93.02, 93.33, 93.49, 93.58, 93.75, 94.19, 94.44, 94.55, 95.35, 95.41, 96.3, 96.36, 96.51, 96.88, 97.25, 97.65, 97.67, 97.74, 98.11, 98.15, 98.26, 98.44, 98.5, 99.02, 99.07, 99.31, 99.44, 99.5, 99.59, 99.88, 99.93]
- **Root Cause:**
  - `97.67%`: Raw 30D features (Stage A).
  - `96.51%`: 16D Autoencoder latent space (Stage C).
  - `93.02%` / `91.86%`: 8-feature subset (Stage D / QAOA vs MI).
  - `94.19%`: 5-seed aggregated benchmark.
- **Remediation:** Label every baseline with exact feature space and protocol specification (e.g. `Raw-30 SVM (25-fold CV)` vs `8D-QAOA SVM (25-fold CV)`).

---

## 2. Model Architecture & Parameter Count Discrepancies

- **Reported VQC Parameters:** [16, 32]
- **Root Cause:**
  - **Ansatz A (Single-parameter RY):** $N \times L = 8 \times 2 = 16$ trainable parameters.
  - **Ansatz B (Dual-parameter RY + RZ):** $2 \times N \times L = 2 \times 8 \times 2 = 32$ trainable parameters.
- **Remediation:** Derive parameter counts strictly via `vqc.count_params()` programmatically. Never hardcode parameter totals in documentation.

---

## 3. Test Suite Count Mismatches

- **Reported Counts in Badges/Text:** Badge claims `188/188 Passed`; Part 8 claims `17/17 Passed`; Part 7 reports `18 Passed`.
- **Actual Pytest Inventory:** `207` collected tests across 11 test modules.
- **Root Cause:** Badges and text were manually typed during different sprint phases.
- **Remediation:** Programmatically query `pytest --collect-only -q` to generate test badges and text counts in `scripts/build_report.py`.

---

## 4. Duplicate Part Designation

- **Identified Conflict:**
  1. `Part 10: Privacy, Security & Safe Medical Data Handling` (`src/security/`)
  2. `Part 10: FastAPI Inference Service & Production REST API` (`app/`)
- **Remediation:** Renumber Privacy & Security foundation as Part 10A / Part 10 Foundation, and FastAPI Inference Service as Part 10B / Part 10 Production API, or unify under a single deployment structure.

---

## 5. Engineering Limit Conflicts

- **`src/security/validation.py`:** `MAX_BATCH_SIZE = 1000`
- **`app/config.py`:** `max_batch_size = 100`
- **Root Cause:** Independent configuration constants between security utility library and API runtime.
- **Remediation:** Unify into a single centralized configuration (`MAX_BATCH_SIZE = 100`) imported by both modules.

---

## 6. Audit Verdict

All contradictory figures are systematically cataloged. The repository is cleared to proceed with **Phase 1: Canonical Evaluation Protocol** to replace all hand-typed or single-seed numbers with reproducible 25-fold bootstrap estimates.
