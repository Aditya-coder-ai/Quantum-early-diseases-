"""
Master Orchestration Script for Part 5 — Class Imbalance Handling + Quantum GAN Experiment.

Executes end-to-end:
    1. Load Part 4 selected features (QAOA 8D primary, Classical MI 8D ablation)
    2. Measure original class imbalance on training data
    3. Evaluate classical imbalance strategies (Original, Class Weights, SMOTE at multiple ratios)
    4. Audit SMOTE synthetic data quality and mode collapse
    5. Run tiny synthetic QGAN verification test
    6. Train QGAN strictly on training minority samples
    7. Audit QGAN synthetic data quality and mode collapse
    8. Evaluate QGAN augmentation across multiple ratios
    9. Run QGAN multi-seed stability experiment
    10. Feature space ablation (QAOA features vs Classical MI features)
    11. Final evaluation on untouched test set
    12. Save all augmented datasets, models, metrics, and summary reports
"""
from __future__ import annotations

import os
import sys
import json
import time
import functools
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.imbalance.config import (
    PART5_RESULTS_DIR, PART5_MODELS_DIR, PART5_DATA_DIR, PART5_ARTIFACTS_DIR,
    QAOA_FEATURES_DIR, CLASSICAL_MI_DIR, FEATURE_DIM, MINORITY_CLASS,
    MAJORITY_CLASS, SEED, AUGMENTATION_RATIOS, QGAN_N_QUBITS,
    QGAN_CIRCUIT_DEPTH, QGAN_EPOCHS, QGAN_BATCH_SIZE, QGAN_STABILITY_SEEDS,
)
from src.imbalance.classical import (
    measure_imbalance, compute_class_weights, smote_oversample
)
from src.imbalance.quality import (
    evaluate_synthetic_quality, detect_mode_collapse
)
from src.imbalance.qgan import (
    QuantumGAN, run_tiny_synthetic_test
)
from src.imbalance.evaluation import (
    evaluate_imbalance_strategy, print_evaluation_summary, get_downstream_classifier
)


def load_datasets():
    """Load QAOA features, Classical MI features, and ground truth labels."""
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).squeeze().values
    y_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).squeeze().values
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).squeeze().values

    # QAOA 8-D features (primary)
    X_train_qaoa = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_train_selected.csv")).values
    X_val_qaoa = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_val_selected.csv")).values
    X_test_qaoa = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_test_selected.csv")).values

    # Classical MI 8-D features (ablation)
    X_train_mi = pd.read_csv(os.path.join(CLASSICAL_MI_DIR, "X_train_selected.csv")).values
    X_val_mi = pd.read_csv(os.path.join(CLASSICAL_MI_DIR, "X_val_selected.csv")).values
    X_test_mi = pd.read_csv(os.path.join(CLASSICAL_MI_DIR, "X_test_selected.csv")).values

    return {
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "qaoa": {"train": X_train_qaoa, "val": X_val_qaoa, "test": X_test_qaoa},
        "mi": {"train": X_train_mi, "val": X_val_mi, "test": X_test_mi},
    }


def main():
    t_global_start = time.time()

    print("\n" + "=" * 70)
    print("   PART 5 -- CLASS IMBALANCE HANDLING + QUANTUM GAN EXPERIMENT")
    print("=" * 70)

    # ── STEP 0: Load and verify data ──
    print("\n[STEP 0] Loading Part 4 selected features and labels...")
    data = load_datasets()
    X_tr = data["qaoa"]["train"]
    X_va = data["qaoa"]["val"]
    X_te = data["qaoa"]["test"]
    y_tr = data["y_train"]
    y_va = data["y_val"]
    y_te = data["y_test"]

    print(f"  Primary QAOA Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    assert X_tr.shape[1] == FEATURE_DIM, f"Expected {FEATURE_DIM}D features, got {X_tr.shape[1]}"

    # ── STEP 1: Measure original class imbalance on training data ──
    print("\n[STEP 1] Measuring class imbalance on training data...")
    imb_info = measure_imbalance(y_tr, minority_class=MINORITY_CLASS, majority_class=MAJORITY_CLASS)
    print(f"  Total train samples:     {imb_info['total_samples']}")
    print(f"  Majority (Class 1):      {imb_info['majority_samples']} ({imb_info['majority_percentage']}%)")
    print(f"  Minority (Class 0):      {imb_info['minority_samples']} ({imb_info['minority_percentage']}%)")
    print(f"  Imbalance ratio:         {imb_info['imbalance_ratio']}:1 (deficit = {imb_info['deficit']})")

    all_val_metrics = []

    # ── STEP 2: Imbalanced Baseline (class_weight=None) ──
    print("\n[STEP 2] Establishing original imbalanced baseline...")
    base_res = evaluate_imbalance_strategy(
        X_tr, y_tr, X_va, y_va,
        method_name="Original_Imbalanced",
        augmentation_ratio=0.0,
        class_weight=None,
        seed=SEED,
    )
    print_evaluation_summary(base_res)
    all_val_metrics.append(base_res)

    # ── STEP 3: Class Weight Baseline ──
    print("\n[STEP 3] Evaluating class-weighted loss baseline...")
    class_weights = compute_class_weights(y_tr, MINORITY_CLASS, MAJORITY_CLASS)
    print(f"  Computed weights: {class_weights}")
    cw_res = evaluate_imbalance_strategy(
        X_tr, y_tr, X_va, y_va,
        method_name="Class_Weighted",
        augmentation_ratio=0.0,
        class_weight="balanced",
        seed=SEED,
    )
    print_evaluation_summary(cw_res)
    all_val_metrics.append(cw_res)

    # ── STEP 4: SMOTE Experiments across ratios ──
    print("\n[STEP 4] Running SMOTE oversampling across multiple ratios...")
    smote_reports = {}
    smote_1_0_aug = None

    for r in AUGMENTATION_RATIOS:
        X_smote_tr, y_smote_tr, X_smote_synth = smote_oversample(
            X_tr, y_tr, ratio=r, seed=SEED
        )
        smote_eval = evaluate_imbalance_strategy(
            X_smote_tr, y_smote_tr, X_va, y_va,
            method_name=f"SMOTE_r{r:.2f}",
            augmentation_ratio=r,
            seed=SEED,
        )
        print_evaluation_summary(smote_eval)
        all_val_metrics.append(smote_eval)

        # Audit quality of synthetic samples
        q_audit = evaluate_synthetic_quality(
            X_tr[y_tr == MINORITY_CLASS], X_smote_synth, X_tr[y_tr == MAJORITY_CLASS]
        )
        smote_reports[str(r)] = q_audit

        if r == 1.0:
            smote_1_0_aug = (X_smote_tr, y_smote_tr, X_smote_synth)

    # Save SMOTE synthetic dataset at 1.0 balance
    smote_out_dir = os.path.join(PROJECT_ROOT, "data", "augmented", "smote")
    os.makedirs(smote_out_dir, exist_ok=True)
    pd.DataFrame(smote_1_0_aug[0]).to_csv(
        os.path.join(smote_out_dir, "X_train_augmented_smote_r1.0.csv"), index=False
    )
    pd.DataFrame(smote_1_0_aug[1]).to_csv(
        os.path.join(smote_out_dir, "y_train_augmented_smote_r1.0.csv"), index=False
    )
    pd.DataFrame(smote_1_0_aug[2]).to_csv(
        os.path.join(smote_out_dir, "X_synthetic_smote_r1.0.csv"), index=False
    )
    print(f"  SMOTE quality audit (r=1.0): Mode Collapse Status = {smote_reports['1.0']['mode_collapse']['status']}, "
          f"KS Match = {smote_reports['1.0']['ks_matching_features_pct']}%")

    # ── STEP 5: Tiny Synthetic QGAN Verification (Step 8) ──
    print("\n[STEP 5] Running tiny synthetic QGAN verification test (2D / 2 qubits)...")
    tiny_res = run_tiny_synthetic_test(seed=SEED)
    print(f"  Tiny QGAN test: {tiny_res['status']} (weights_norm={tiny_res['weights_norm']:.4f})")

    # ── STEP 6: Train Real QGAN on Training Minority Data ──
    print("\n[STEP 6] Training Quantum GAN strictly on training minority samples...")
    X_min = X_tr[y_tr == MINORITY_CLASS]
    print(f"  Minority training samples: {X_min.shape} (Zero validation/test samples used)")

    qgan = QuantumGAN(
        n_qubits=QGAN_N_QUBITS,
        n_layers=QGAN_CIRCUIT_DEPTH,
        empirical_mean=np.mean(X_min, axis=0),
        empirical_std=np.std(X_min, axis=0),
        seed=SEED,
    )

    t0_qgan = time.time()
    train_qgan_res = qgan.train(
        X_min,
        epochs=QGAN_EPOCHS,
        batch_size=QGAN_BATCH_SIZE,
        verbose=True,
    )
    qgan_train_time = round(time.time() - t0_qgan, 2)
    saved_models = qgan.save(PART5_MODELS_DIR, tag="primary")
    print(f"  QGAN models saved to: {saved_models}")

    # ── STEP 7: QGAN Augmentation across multiple ratios ──
    print("\n[STEP 7] Evaluating QGAN augmentation across multiple ratios...")
    qgan_reports = {}
    deficit = imb_info["deficit"]
    qgan_1_0_aug = None

    for r in AUGMENTATION_RATIOS:
        n_synth = int(round(deficit * r))
        X_synth = qgan.generate(n_synth, seed=SEED)

        # Quality & Mode-Collapse Check
        q_audit = evaluate_synthetic_quality(
            X_min, X_synth, X_tr[y_tr == MAJORITY_CLASS]
        )
        qgan_reports[str(r)] = q_audit

        # Combine into augmented training set
        X_aug_qgan = np.vstack([X_tr, X_synth])
        y_aug_qgan = np.hstack([y_tr, np.full(n_synth, MINORITY_CLASS, dtype=y_tr.dtype)])

        qgan_eval = evaluate_imbalance_strategy(
            X_aug_qgan, y_aug_qgan, X_va, y_va,
            method_name=f"QGAN_r{r:.2f}",
            augmentation_ratio=r,
            seed=SEED,
        )
        # Add quantum metadata
        qgan_eval["qgan_qubits"] = QGAN_N_QUBITS
        qgan_eval["qgan_depth"] = QGAN_CIRCUIT_DEPTH
        qgan_eval["qgan_epochs"] = QGAN_EPOCHS
        qgan_eval["qgan_training_time_s"] = qgan_train_time
        print_evaluation_summary(qgan_eval)
        all_val_metrics.append(qgan_eval)

        if r == 1.0:
            qgan_1_0_aug = (X_aug_qgan, y_aug_qgan, X_synth)

    # Save QGAN augmented data at 1.0 balance
    pd.DataFrame(qgan_1_0_aug[0]).to_csv(
        os.path.join(PART5_DATA_DIR, "X_train_augmented_qgan_r1.0.csv"), index=False
    )
    pd.DataFrame(qgan_1_0_aug[1]).to_csv(
        os.path.join(PART5_DATA_DIR, "y_train_augmented_qgan_r1.0.csv"), index=False
    )
    pd.DataFrame(qgan_1_0_aug[2]).to_csv(
        os.path.join(PART5_DATA_DIR, "X_synthetic_qgan_r1.0.csv"), index=False
    )
    print(f"  QGAN quality audit (r=1.0): Mode Collapse Status = {qgan_reports['1.0']['mode_collapse']['status']}, "
          f"Mean diff L2 = {qgan_reports['1.0']['mean_diff_l2']}, KS Match = {qgan_reports['1.0']['ks_matching_features_pct']}%")

    # ── STEP 8: QGAN Stability Across Seeds ──
    print(f"\n[STEP 8] Running QGAN multi-seed stability experiment across {len(QGAN_STABILITY_SEEDS)} seeds...")
    stability_results = []
    for s in QGAN_STABILITY_SEEDS:
        print(f"  --- Trial with seed {s} ---", flush=True)
        qgan_trial = QuantumGAN(
            n_qubits=QGAN_N_QUBITS,
            n_layers=QGAN_CIRCUIT_DEPTH,
            empirical_mean=np.mean(X_min, axis=0),
            empirical_std=np.std(X_min, axis=0),
            seed=s,
        )
        qgan_trial.train(X_min, epochs=QGAN_EPOCHS, batch_size=QGAN_BATCH_SIZE, verbose=False)
        X_synth_trial = qgan_trial.generate(deficit, seed=s)
        X_aug_trial = np.vstack([X_tr, X_synth_trial])
        y_aug_trial = np.hstack([y_tr, np.full(deficit, MINORITY_CLASS, dtype=y_tr.dtype)])

        eval_trial = evaluate_imbalance_strategy(
            X_aug_trial, y_aug_trial, X_va, y_va,
            method_name=f"QGAN_stability_seed{s}",
            augmentation_ratio=1.0,
            seed=s,
        )
        stability_results.append(eval_trial)
        print(f"    Seed {s} -> Val Acc: {eval_trial['accuracy']:.4f} | F1: {eval_trial['f1']:.4f} | "
              f"Minority Recall: {eval_trial['minority_recall']:.4f}")

    stab_df = pd.DataFrame(stability_results)
    stability_summary = {
        "seeds": QGAN_STABILITY_SEEDS,
        "accuracy_mean": round(float(stab_df["accuracy"].mean()), 4),
        "accuracy_std": round(float(stab_df["accuracy"].std()), 4),
        "f1_mean": round(float(stab_df["f1"].mean()), 4),
        "f1_std": round(float(stab_df["f1"].std()), 4),
        "minority_recall_mean": round(float(stab_df["minority_recall"].mean()), 4),
        "minority_recall_std": round(float(stab_df["minority_recall"].std()), 4),
        "roc_auc_mean": round(float(stab_df["roc_auc"].mean()), 4),
        "roc_auc_std": round(float(stab_df["roc_auc"].std()), 4),
    }
    print(f"  Stability summary: Acc = {stability_summary['accuracy_mean']} +/- {stability_summary['accuracy_std']}, "
          f"Minority Recall = {stability_summary['minority_recall_mean']} +/- {stability_summary['minority_recall_std']}")

    # ── STEP 9: Feature Space Ablation (QAOA features vs Classical MI features) ──
    print("\n[STEP 9] Running feature space ablation (QAOA features vs Classical MI features)...")
    X_tr_mi = data["mi"]["train"]
    X_va_mi = data["mi"]["val"]
    ablation_metrics = []

    # On Classical MI features
    res_mi_orig = evaluate_imbalance_strategy(X_tr_mi, y_tr, X_va_mi, y_va, method_name="MI_Original", seed=SEED)
    res_mi_cw = evaluate_imbalance_strategy(X_tr_mi, y_tr, X_va_mi, y_va, method_name="MI_ClassWeighted", class_weight="balanced", seed=SEED)
    X_smote_mi, y_smote_mi, _ = smote_oversample(X_tr_mi, y_tr, ratio=1.0, seed=SEED)
    res_mi_smote = evaluate_imbalance_strategy(X_smote_mi, y_smote_mi, X_va_mi, y_va, method_name="MI_SMOTE_r1.0", augmentation_ratio=1.0, seed=SEED)

    # QGAN on MI features
    qgan_mi = QuantumGAN(
        n_qubits=FEATURE_DIM,
        n_layers=QGAN_CIRCUIT_DEPTH,
        empirical_mean=np.mean(X_tr_mi[y_tr == MINORITY_CLASS], axis=0),
        empirical_std=np.std(X_tr_mi[y_tr == MINORITY_CLASS], axis=0),
        seed=SEED,
    )
    qgan_mi.train(X_tr_mi[y_tr == MINORITY_CLASS], epochs=QGAN_EPOCHS, batch_size=QGAN_BATCH_SIZE, verbose=False)
    X_synth_mi = qgan_mi.generate(deficit, seed=SEED)
    X_aug_mi_qgan = np.vstack([X_tr_mi, X_synth_mi])
    y_aug_mi_qgan = np.hstack([y_tr, np.full(deficit, MINORITY_CLASS, dtype=y_tr.dtype)])
    res_mi_qgan = evaluate_imbalance_strategy(X_aug_mi_qgan, y_aug_mi_qgan, X_va_mi, y_va, method_name="MI_QGAN_r1.0", augmentation_ratio=1.0, seed=SEED)

    ablation_metrics.extend([res_mi_orig, res_mi_cw, res_mi_smote, res_mi_qgan])
    for ab in ablation_metrics:
        print_evaluation_summary(ab)

    # ── STEP 10: Final Evaluation on UNTOUCHED Test Set ──
    print("\n" + "=" * 50)
    print("   FINAL TEST SET EVALUATION (EVALUATED ONCE)")
    print("=" * 50)
    test_metrics = []

    # 1. Original Imbalanced on Test
    test_orig = evaluate_imbalance_strategy(X_tr, y_tr, X_te, y_te, method_name="Test_Original_Imbalanced", seed=SEED)
    print_evaluation_summary(test_orig)
    test_metrics.append(test_orig)

    # 2. Class Weighted on Test
    test_cw = evaluate_imbalance_strategy(X_tr, y_tr, X_te, y_te, method_name="Test_Class_Weighted", class_weight="balanced", seed=SEED)
    print_evaluation_summary(test_cw)
    test_metrics.append(test_cw)

    # 3. SMOTE r=1.0 on Test
    test_smote = evaluate_imbalance_strategy(smote_1_0_aug[0], smote_1_0_aug[1], X_te, y_te, method_name="Test_SMOTE_r1.0", augmentation_ratio=1.0, seed=SEED)
    print_evaluation_summary(test_smote)
    test_metrics.append(test_smote)

    # 4. QGAN r=1.0 on Test
    test_qgan = evaluate_imbalance_strategy(qgan_1_0_aug[0], qgan_1_0_aug[1], X_te, y_te, method_name="Test_QGAN_r1.0", augmentation_ratio=1.0, seed=SEED)
    print_evaluation_summary(test_qgan)
    test_metrics.append(test_qgan)

    # ── STEP 11: Save All Reports & CSVs ──
    print("\n[STEP 11] Saving all experiment tables, quality audits, and summary reports...")
    val_df = pd.DataFrame(all_val_metrics)
    val_csv_path = os.path.join(PART5_RESULTS_DIR, "comparison_table.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"  Validation comparison table saved: {val_csv_path}")

    test_df = pd.DataFrame(test_metrics)
    test_csv_path = os.path.join(PART5_RESULTS_DIR, "final_test_comparison.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Final test comparison table saved: {test_csv_path}")

    ablation_df = pd.DataFrame(ablation_metrics)
    ablation_csv_path = os.path.join(PART5_RESULTS_DIR, "feature_space_ablation.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"  Feature space ablation table saved: {ablation_csv_path}")

    # Quality reports
    with open(os.path.join(PART5_RESULTS_DIR, "smote_quality_report.json"), "w") as f:
        json.dump(smote_reports, f, indent=2)
    with open(os.path.join(PART5_RESULTS_DIR, "qgan_quality_report.json"), "w") as f:
        json.dump(qgan_reports, f, indent=2)
    with open(os.path.join(PART5_RESULTS_DIR, "qgan_stability_report.json"), "w") as f:
        json.dump(stability_summary, f, indent=2)

    total_time = round(time.time() - t_global_start, 2)

    summary = {
        "status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_s": total_time,
        "dataset": {
            "feature_dim": FEATURE_DIM,
            "train_samples": len(y_tr),
            "val_samples": len(y_va),
            "test_samples": len(y_te),
            "minority_class": MINORITY_CLASS,
            "majority_class": MAJORITY_CLASS,
            "original_imbalance": imb_info,
        },
        "classical_strategies": {
            "class_weights": class_weights,
            "smote_ratios": AUGMENTATION_RATIOS,
        },
        "qgan": {
            "n_qubits": QGAN_N_QUBITS,
            "circuit_depth": QGAN_CIRCUIT_DEPTH,
            "epochs": QGAN_EPOCHS,
            "batch_size": QGAN_BATCH_SIZE,
            "training_time_s": qgan_train_time,
            "quality_audit_r1.0": qgan_reports["1.0"],
            "stability_summary": stability_summary,
        },
        "validation_leaderboard": val_df[["method", "accuracy", "f1", "roc_auc", "minority_recall", "minority_f1"]].to_dict(orient="records"),
        "final_test_leaderboard": test_df[["method", "accuracy", "f1", "roc_auc", "minority_recall", "minority_f1"]].to_dict(orient="records"),
    }

    summary_path = os.path.join(PART5_RESULTS_DIR, "part5_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("   PART 5 COMPLETED SUCCESSFULLY!")
    print(f"   Total runtime: {total_time}s")
    print(f"   Validation Table: {val_csv_path}")
    print(f"   Final Test Table: {test_csv_path}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
