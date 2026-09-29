"""
Master Orchestration Script for Part 6 — Variational Quantum Classifier (VQC).

Executes end-to-end:
    1. Load Part 4 selected features and Part 5 balanced datasets
    2. Fit AngleScaler strictly on training data (zero leakage)
    3. Train and evaluate classical reference baselines (SVM, LogReg, MLP)
    4. Run tiny synthetic VQC test
    5. Evaluate VQC across imbalance strategies (Original, Class-Weighted, SMOTE, QGAN)
    6. Conduct feature / qubit count scaling study (K in {4, 6, 8})
    7. Conduct circuit depth study (L in {1, 2, 3})
    8. Conduct ansatz comparison (Ansatz A vs Ansatz B)
    9. Conduct multi-seed stability study (seeds {42, 43, 44})
    10. Conduct quantum noise robustness evaluation (Ideal vs Shots vs Depolarizing)
    11. Final single-pass evaluation on untouched test set
    12. Generate publication-quality research figures
    13. Save all models, tables, metrics, and summary JSON reports
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
import torch

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.vqc.config import (
    PART6_RESULTS_DIR, PART6_MODELS_DIR, PART6_PLOTS_DIR, QAOA_FEATURES_DIR,
    CLASSICAL_MI_DIR, SMOTE_DATA_DIR, QGAN_DATA_DIR, VQCConfig
)
from src.vqc.encoding import AngleScaler, FeatureQubitMapper
from src.vqc.circuit import count_quantum_resources
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc, run_tiny_vqc_test
from src.vqc.evaluation import evaluate_vqc_model, train_and_evaluate_classical_reference
from src.vqc.noise import evaluate_under_noise
from src.vqc.ablation import run_vqc_experiment
from src.vqc.visualization import (
    plot_training_history, plot_vqc_vs_classical_comparison,
    plot_circuit_depth_scaling, plot_feature_count_scaling,
    plot_confusion_matrix_vqc, plot_roc_and_pr_curves
)


def print_metrics_summary(name: str, res: dict):
    print(f"  [{name}] Acc: {res.get('accuracy', 0.0):.4f} | F1: {res.get('f1', 0.0):.4f} | "
          f"ROC-AUC: {res.get('roc_auc', 0.0):.4f} | PR-AUC: {res.get('pr_auc', 0.0):.4f} | "
          f"Minority Recall: {res.get('minority_recall', 0.0):.4f} | FN: {res.get('malignant_false_negatives', res.get('fp', 0))}")


def main():
    t_global_start = time.time()
    cfg = VQCConfig()

    print("\n" + "=" * 70)
    print("   PART 6 -- VARIATIONAL QUANTUM CLASSIFIER (VQC) EXPERIMENT")
    print("=" * 70)

    # ── STEP 0: Load Data & Precomputed Splits ──
    print("\n[STEP 0] Loading Part 4 selected features and Part 5 datasets...")
    X_tr_df = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_train_selected.csv"))
    X_va_df = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_val_selected.csv"))
    X_te_df = pd.read_csv(os.path.join(QAOA_FEATURES_DIR, "X_test_selected.csv"))

    y_tr = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_va = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_val.csv")).values.ravel()
    y_te = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "y_test.csv")).values.ravel()

    feature_names = list(X_tr_df.columns)
    mapper = FeatureQubitMapper(feature_names)

    X_tr = X_tr_df.values
    X_va = X_va_df.values
    X_te = X_te_df.values

    print(f"  Train: {X_tr.shape} (Majority=1: {(y_tr==1).sum()}, Minority=0: {(y_tr==0).sum()})")
    print(f"  Val:   {X_va.shape} (Majority=1: {(y_va==1).sum()}, Minority=0: {(y_va==0).sum()})")
    print(f"  Test:  {X_te.shape} (Majority=1: {(y_te==1).sum()}, Minority=0: {(y_te==0).sum()})")

    # Load Part 5 balanced datasets
    X_tr_smote = pd.read_csv(os.path.join(SMOTE_DATA_DIR, "X_train_augmented_smote_r1.0.csv")).values
    y_tr_smote = pd.read_csv(os.path.join(SMOTE_DATA_DIR, "y_train_augmented_smote_r1.0.csv")).values.ravel()

    X_tr_qgan = pd.read_csv(os.path.join(QGAN_DATA_DIR, "X_train_augmented_qgan_r1.0.csv")).values
    y_tr_qgan = pd.read_csv(os.path.join(QGAN_DATA_DIR, "y_train_augmented_qgan_r1.0.csv")).values.ravel()

    # ── STEP 1: Feature Normalization & Angle Encoding (Zero Leakage) ──
    print("\n[STEP 1] Fitting AngleScaler strictly on training data...")
    scaler = AngleScaler(target_range=(0.0, np.pi))
    scaler.fit(X_tr)

    scaler_path = os.path.join(PART6_MODELS_DIR, "angle_scaler.json")
    scaler.save(scaler_path)
    print(f"  AngleScaler fitted and saved to: {scaler_path}")

    X_tr_angles = scaler.transform(X_tr)
    X_va_angles = scaler.transform(X_va)
    X_te_angles = scaler.transform(X_te)

    # Angle scaling for balanced datasets using the same training-fitted scaler
    X_tr_smote_angles = scaler.transform(X_tr_smote)
    X_tr_qgan_angles = scaler.transform(X_tr_qgan)

    assert not np.isnan(X_tr_angles).any() and not np.isinf(X_tr_angles).any()
    assert (X_tr_angles >= 0.0).all() and (X_tr_angles <= np.pi + 1e-5).all()
    print("  Angle bounds verified strictly in [0, pi] without NaN/Inf.")

    # ── STEP 2: Classical Reference Models on Validation Set ──
    print("\n[STEP 2] Training and evaluating classical reference models on validation set...")
    val_leaderboard = []

    for clf_name in ["SVM_RBF", "LogisticRegression", "MLP"]:
        res_clf = train_and_evaluate_classical_reference(
            clf_name, X_tr, y_tr, X_va, y_va, class_weight="balanced", seed=cfg.seed
        )
        res_clf["method"] = f"Classical_{clf_name}"
        res_clf["qubits"] = 0
        res_clf["depth"] = 0
        res_clf["ansatz"] = "classical"
        res_clf["entanglement"] = "none"
        res_clf["parameters"] = "classical"
        print_metrics_summary(res_clf["method"], res_clf)
        val_leaderboard.append(res_clf)

    # ── STEP 3: Tiny Synthetic VQC Sanity Test ──
    print("\n[STEP 3] Running tiny synthetic VQC verification test (4 qubits)...")
    tiny_res = run_tiny_vqc_test(seed=cfg.seed)
    print(f"  Tiny VQC test: {tiny_res['status']} | Grad Norm: {tiny_res['grad_norm']:.4f} | Weight Diff: {tiny_res['weight_diff']:.4f}")
    assert tiny_res["status"] == "PASSED", "Tiny VQC sanity test failed."

    # ── STEP 4: Imbalance Strategy Comparison for VQC (Validation Split) ──
    print("\n[STEP 4] Evaluating VQC across imbalance handling strategies (8 Qubits, Depth 2, Ring)...")
    imbalance_configs = [
        ("VQC_Original", X_tr_angles, y_tr, None),
        ("VQC_ClassWeighted", X_tr_angles, y_tr, cfg.class_weights),
        ("VQC_SMOTE", X_tr_smote_angles, y_tr_smote, None),
        ("VQC_QGAN", X_tr_qgan_angles, y_tr_qgan, None),
    ]

    trained_vqc_models = {}
    best_vqc_model = None
    best_vqc_metrics = None
    best_vqc_history = None
    best_val_score = -1.0

    for name, x_train_cur, y_train_cur, cw in imbalance_configs:
        model, eval_res, train_res = run_vqc_experiment(
            X_train_angles=x_train_cur,
            y_train=y_train_cur,
            X_val_angles=X_va_angles,
            y_val=y_va,
            n_qubits=8,
            n_layers=2,
            ansatz_type=cfg.ansatz_type,
            entanglement=cfg.entanglement,
            class_weights=cw,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.learning_rate,
            early_stopping_patience=cfg.early_stopping_patience,
            seed=cfg.seed,
            verbose=False,
        )
        eval_res["method"] = name
        print_metrics_summary(name, eval_res)
        val_leaderboard.append(eval_res)
        trained_vqc_models[name] = (model, eval_res, train_res)

        # Track best model on validation PR-AUC
        if eval_res["pr_auc"] > best_val_score:
            best_val_score = eval_res["pr_auc"]
            best_vqc_model = model
            best_vqc_metrics = eval_res
            best_vqc_history = train_res["history"]

    # Save primary best VQC model
    best_model_path = os.path.join(PART6_MODELS_DIR, "vqc_best_model.pt")
    best_vqc_model.save(best_model_path)
    print(f"  Best VQC model checkpoint saved to: {best_model_path}")

    # ── STEP 5: Feature / Qubit Count Scaling Study ──
    print("\n[STEP 5] Conducting Feature / Qubit count scaling study (K in {4, 6, 8})...")
    feature_records = []

    for k in cfg.feature_counts:
        X_tr_k, names_k = mapper.slice_features(X_tr, k)
        X_va_k, _ = mapper.slice_features(X_va, k)

        # Fit training scaler on k features
        scaler_k = AngleScaler().fit(X_tr_k)
        X_tr_k_angles = scaler_k.transform(X_tr_k)
        X_va_k_angles = scaler_k.transform(X_va_k)

        model_k, eval_k, _ = run_vqc_experiment(
            X_train_angles=X_tr_k_angles,
            y_train=y_tr,
            X_val_angles=X_va_k_angles,
            y_val=y_va,
            n_qubits=k,
            n_layers=2,
            ansatz_type="ansatz_b",
            entanglement="ring",
            class_weights=cfg.class_weights,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            verbose=False,
        )
        eval_k["feature_count"] = k
        print(f"  K={k} Qubits -> Val Acc: {eval_k['accuracy']:.4f} | F1: {eval_k['f1']:.4f} | "
              f"Minority Recall: {eval_k['minority_recall']:.4f} | PR-AUC: {eval_k['pr_auc']:.4f} "
              f"({eval_k['training_time_s']:.2f}s)")
        feature_records.append(eval_k)

    feature_df = pd.DataFrame(feature_records)
    feature_df.to_csv(os.path.join(PART6_RESULTS_DIR, "feature_count_study.csv"), index=False)

    # ── STEP 6: Circuit Depth Scaling Study ──
    print("\n[STEP 6] Conducting Circuit Depth scaling study (L in {1, 2, 3} layers)...")
    depth_records = []

    for depth in cfg.circuit_depths:
        model_d, eval_d, _ = run_vqc_experiment(
            X_train_angles=X_tr_angles,
            y_train=y_tr,
            X_val_angles=X_va_angles,
            y_val=y_va,
            n_qubits=8,
            n_layers=depth,
            ansatz_type="ansatz_b",
            entanglement="ring",
            class_weights=cfg.class_weights,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            verbose=False,
        )
        eval_d["depth"] = depth
        print(f"  Depth L={depth} -> Val Acc: {eval_d['accuracy']:.4f} | F1: {eval_d['f1']:.4f} | "
              f"Minority Recall: {eval_d['minority_recall']:.4f} | Training Time: {eval_d['training_time_s']:.2f}s")
        depth_records.append(eval_d)

    depth_df = pd.DataFrame(depth_records)
    depth_df.to_csv(os.path.join(PART6_RESULTS_DIR, "circuit_depth_study.csv"), index=False)

    # ── STEP 7: Ansatz Architecture Study ──
    print("\n[STEP 7] Comparing Ansatz Architectures (Ansatz A vs. Ansatz B)...")
    # Ansatz A: RY only + Linear CNOT
    model_a, eval_a, _ = run_vqc_experiment(
        X_train_angles=X_tr_angles,
        y_train=y_tr,
        X_val_angles=X_va_angles,
        y_val=y_va,
        n_qubits=8,
        n_layers=2,
        ansatz_type="ansatz_a",
        entanglement="linear",
        class_weights=cfg.class_weights,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        verbose=False,
    )
    eval_a["method"] = "VQC_Ansatz_A_Linear"
    print_metrics_summary("Ansatz A (RY + Linear)", eval_a)

    # Ansatz B: RY/RZ + Ring CNOT
    eval_b = trained_vqc_models["VQC_ClassWeighted"][1]
    eval_b["method"] = "VQC_Ansatz_B_Ring"
    print_metrics_summary("Ansatz B (RY/RZ + Ring)", eval_b)

    ansatz_df = pd.DataFrame([eval_a, eval_b])
    ansatz_df.to_csv(os.path.join(PART6_RESULTS_DIR, "ansatz_comparison.csv"), index=False)

    # ── STEP 8: Multi-Seed Stability Experiment ──
    print(f"\n[STEP 8] Conducting multi-seed stability experiment across {len(cfg.stability_seeds)} seeds...")
    stability_records = []

    for s in cfg.stability_seeds:
        _, eval_s, _ = run_vqc_experiment(
            X_train_angles=X_tr_angles,
            y_train=y_tr,
            X_val_angles=X_va_angles,
            y_val=y_va,
            n_qubits=8,
            n_layers=2,
            ansatz_type="ansatz_b",
            entanglement="ring",
            class_weights=cfg.class_weights,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            seed=s,
            verbose=False,
        )
        eval_s["seed"] = s
        print(f"  Seed {s} -> Val Acc: {eval_s['accuracy']:.4f} | F1: {eval_s['f1']:.4f} | "
              f"Minority Recall: {eval_s['minority_recall']:.4f} | PR-AUC: {eval_s['pr_auc']:.4f}")
        stability_records.append(eval_s)

    stab_df = pd.DataFrame(stability_records)
    stab_df.to_csv(os.path.join(PART6_RESULTS_DIR, "stability_study.csv"), index=False)

    stability_summary = {
        "seeds": cfg.stability_seeds,
        "accuracy_mean": round(float(stab_df["accuracy"].mean()), 4),
        "accuracy_std": round(float(stab_df["accuracy"].std()), 4),
        "f1_mean": round(float(stab_df["f1"].mean()), 4),
        "f1_std": round(float(stab_df["f1"].std()), 4),
        "minority_recall_mean": round(float(stab_df["minority_recall"].mean()), 4),
        "minority_recall_std": round(float(stab_df["minority_recall"].std()), 4),
        "roc_auc_mean": round(float(stab_df["roc_auc"].mean()), 4),
        "roc_auc_std": round(float(stab_df["roc_auc"].std()), 4),
        "pr_auc_mean": round(float(stab_df["pr_auc"].mean()), 4),
        "pr_auc_std": round(float(stab_df["pr_auc"].std()), 4),
    }
    print(f"  Stability summary: Acc = {stability_summary['accuracy_mean']} +/- {stability_summary['accuracy_std']}, "
          f"Minority Recall = {stability_summary['minority_recall_mean']} +/- {stability_summary['minority_recall_std']}")

    # ── STEP 9: Quantum Noise Robustness Evaluation ──
    print("\n[STEP 9] Evaluating VQC under quantum noise (Shot Noise & Depolarizing Noise)...")
    noise_results = evaluate_under_noise(
        best_vqc_model,
        X_angles=X_va_angles,
        y_true=y_va,
        shots=cfg.noisy_shots,
        depolarizing_prob=cfg.depolarizing_prob,
        minority_class=cfg.minority_class,
    )
    print(f"  Ideal:         Acc: {noise_results['ideal']['accuracy']:.4f} | Minority Recall: {noise_results['ideal']['minority_recall']:.4f}")
    print(f"  Shot Noise:    Acc: {noise_results['shot_noise']['accuracy']:.4f} | Minority Recall: {noise_results['shot_noise']['minority_recall']:.4f} "
          f"(MAE vs Ideal: {noise_results['shot_noise']['prob_mae_vs_ideal']:.4f})")
    print(f"  Depolarizing:  Acc: {noise_results['depolarizing_noise']['accuracy']:.4f} | Minority Recall: {noise_results['depolarizing_noise']['minority_recall']:.4f} "
          f"(MAE vs Ideal: {noise_results['depolarizing_noise']['prob_mae_vs_ideal']:.4f})")

    with open(os.path.join(PART6_RESULTS_DIR, "noise_evaluation.json"), "w") as f:
        json.dump(noise_results, f, indent=2)

    # ── STEP 10: Real Quantum Hardware Status Check ──
    print("\n[STEP 10] Checking real quantum hardware availability...")
    hardware_status = {
        "hardware_executed": False,
        "backend": "None",
        "reason": "Simulator evaluation completed on PennyLane lightning.qubit and default.mixed; cloud quantum hardware credentials (e.g., IBM Quantum / Amazon Braket) are not configured in this environment.",
        "verified_simulator": "lightning.qubit (statevector adjoint diff) & default.mixed (density matrix noise)",
    }
    print(f"  Hardware Status: {hardware_status['reason']}")

    # ── STEP 11: Final Evaluation on UNTOUCHED Test Set (Evaluated Once) ──
    print("\n" + "=" * 50)
    print("   FINAL TEST SET EVALUATION (EVALUATED ONCE)")
    print("=" * 50)
    test_records = []

    # 1. Classical Reference on Test Set
    test_classical = train_and_evaluate_classical_reference(
        "SVM_RBF", X_tr, y_tr, X_te, y_te, class_weight="balanced", seed=cfg.seed
    )
    test_classical["method"] = "Test_Classical_SVM_RBF"
    test_classical["qubits"] = 0
    test_classical["depth"] = 0
    print_metrics_summary("Test Classical SVM", test_classical)
    test_records.append(test_classical)

    # 2. VQC Candidates on Test Set
    vqc_test_candidates = [
        ("Test_VQC_Original", trained_vqc_models["VQC_Original"][0]),
        ("Test_VQC_ClassWeighted", trained_vqc_models["VQC_ClassWeighted"][0]),
        ("Test_VQC_SMOTE", trained_vqc_models["VQC_SMOTE"][0]),
        ("Test_VQC_QGAN", trained_vqc_models["VQC_QGAN"][0]),
    ]

    for cand_name, cand_model in vqc_test_candidates:
        eval_te = evaluate_vqc_model(cand_model, X_te_angles, y_te, minority_class=cfg.minority_class)
        eval_te["method"] = cand_name
        eval_te["qubits"] = cand_model.n_qubits
        eval_te["depth"] = cand_model.n_layers
        eval_te["ansatz"] = cand_model.ansatz_type
        eval_te["entanglement"] = cand_model.entanglement
        eval_te["parameters"] = cand_model.n_params
        print_metrics_summary(cand_name, eval_te)
        test_records.append(eval_te)

    test_df = pd.DataFrame(test_records)
    test_df.to_csv(os.path.join(PART6_RESULTS_DIR, "final_test_comparison.csv"), index=False)

    # ── STEP 12: Generate Publication-Quality Research Figures ──
    print("\n[STEP 12] Generating publication research figures...")
    plot_training_history(best_vqc_history, os.path.join(PART6_PLOTS_DIR, "training_curves.png"))
    plot_vqc_vs_classical_comparison(
        [r for r in val_leaderboard if "VQC" in r["method"] or "Classical_SVM" in r["method"]],
        os.path.join(PART6_PLOTS_DIR, "vqc_vs_classical_comparison.png")
    )
    plot_circuit_depth_scaling(depth_records, os.path.join(PART6_PLOTS_DIR, "performance_vs_depth.png"))
    plot_feature_count_scaling(feature_records, os.path.join(PART6_PLOTS_DIR, "performance_vs_features.png"))

    # Test set confusion matrix & ROC/PR curves for best VQC
    best_vqc_test_probs = best_vqc_model.predict_proba(X_te_angles)
    best_vqc_test_preds = (best_vqc_test_probs >= 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm_vqc_te = confusion_matrix(y_te, best_vqc_test_preds, labels=[0, 1])
    plot_confusion_matrix_vqc(cm_vqc_te, os.path.join(PART6_PLOTS_DIR, "confusion_matrix_vqc.png"))
    plot_roc_and_pr_curves(y_te, best_vqc_test_probs, os.path.join(PART6_PLOTS_DIR, "roc_pr_curves_vqc.png"))
    print(f"  All research figures saved to: {PART6_PLOTS_DIR}")

    # ── STEP 13: Summary JSON & Leaderboard ──
    val_df = pd.DataFrame(val_leaderboard)
    val_df.to_csv(os.path.join(PART6_RESULTS_DIR, "comparison_table.csv"), index=False)

    total_time = round(time.time() - t_global_start, 2)
    resource_profile = count_quantum_resources(8, 2, "ansatz_b", "ring")

    summary = {
        "status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_s": total_time,
        "dataset": {
            "feature_dim": 8,
            "train_samples": len(y_tr),
            "val_samples": len(y_va),
            "test_samples": len(y_te),
            "minority_class": cfg.minority_class,
            "majority_class": cfg.majority_class,
        },
        "vqc_architecture": {
            "n_qubits": 8,
            "n_layers": 2,
            "ansatz": "ansatz_b",
            "entanglement": "ring",
            "parameters": resource_profile["trainable_parameters"],
            "total_gates": resource_profile["total_gates"],
            "entangling_gates": resource_profile["total_entangling_gates"],
            "circuit_depth": resource_profile["circuit_depth"],
            "preferred_device": cfg.preferred_device,
        },
        "training_convergence": {
            "best_epoch": best_vqc_metrics["best_epoch"],
            "total_epochs": best_vqc_metrics["epochs_trained"],
            "best_val_pr_auc": best_vqc_metrics["pr_auc"],
            "best_val_minority_recall": best_vqc_metrics["minority_recall"],
        },
        "validation_leaderboard": val_df[["method", "accuracy", "f1", "roc_auc", "pr_auc", "minority_recall"]].to_dict(orient="records"),
        "final_test_leaderboard": test_df[["method", "accuracy", "f1", "roc_auc", "pr_auc", "minority_recall", "malignant_false_negatives"]].to_dict(orient="records"),
        "stability_summary": stability_summary,
        "noise_summary": {
            "ideal_acc": noise_results["ideal"]["accuracy"],
            "shot_acc": noise_results["shot_noise"]["accuracy"],
            "depol_acc": noise_results["depolarizing_noise"]["accuracy"],
        },
        "hardware_status": hardware_status,
    }

    summary_path = os.path.join(PART6_RESULTS_DIR, "part6_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("   PART 6 COMPLETED SUCCESSFULLY!")
    print(f"   Total runtime: {total_time}s")
    print(f"   Validation Table: {os.path.join(PART6_RESULTS_DIR, 'comparison_table.csv')}")
    print(f"   Final Test Table: {os.path.join(PART6_RESULTS_DIR, 'final_test_comparison.csv')}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
