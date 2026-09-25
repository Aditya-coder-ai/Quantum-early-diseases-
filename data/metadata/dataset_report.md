# Preprocessing Audit Report & Data Pipeline Summary

## 1. Dataset Overview
- **Raw Samples:** 569
- **Feature Count:** 30
- **Target Column:** target (0=Malignant, 1=Benign)

## 2. Leakage-Safe Splits
- **Train:** 398 samples (69.9%) | Classes: {1: np.int64(250), 0: np.int64(148)}
- **Validation:** 85 samples (14.9%) | Classes: {1: np.int64(53), 0: np.int64(32)}
- **Test:** 86 samples (15.1%) | Classes: {1: np.int64(54), 0: np.int64(32)}

## 3. Transformation & Scaling
- **Scaler Method:** standard
- **Fitted Strictly On:** Training Set ($N=398$)
- **Validation/Test Sets:** Transformed via Training-learned parameters (Zero Leakage)
- **Quantum Readiness:** Checked (No NaNs, No Infs, Float64 normalized values)
