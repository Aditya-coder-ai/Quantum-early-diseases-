"""
Comprehensive Ablation Studies and Robustness Experiments for Part 9.
Implements:
1. Controlled Ablations A–E (Isolating Feature Selection, Imbalance, Quantum Classifier)
2. Feature Count Ablation (4, 6, 8, 10, 12 features)
3. Circuit Depth Ablation (1, 2, 3 variational layers)
4. Data Efficiency Experiment (25%, 50%, 75%, 100% training data)
5. Noise & Perturbation Robustness (Feature noise perturbations & Quantum shot/depolarizing noise)
"""
from __future__ import annotations

import copy
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.svm import SVC

from src.comparison.config import ComparisonConfig, ModelConfig
from src.comparison.benchmark import ModelBenchmarkEngine
from src.comparison.metrics import compute_extended_metrics
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc
from src.vqc.encoding import AngleScaler
from src.vqc.noise import build_noisy_vqc_qnode


class AblationSuite:
    """
    Executes controlled research ablations isolating every component's contribution.
    """
    def __init__(self, engine: ModelBenchmarkEngine):
        self.engine = engine
        self.config = engine.config

    def run_all_ablations(self, seed: int = 42) -> Dict[str, Any]:
        """Runs the complete suite of Part 9 ablations."""
        print("\n" + "=" * 70)
        print("  RUNNING PART 9 ABLATION & ROBUSTNESS EXPERIMENTS")
        print("=" * 70)

        abl_matrix = self.run_ablation_matrix(seed)
        feat_ablation = self.run_feature_count_ablation(seed)
        depth_ablation = self.run_circuit_depth_ablation(seed)
        data_eff = self.run_data_efficiency_experiment(seed)
        robustness = self.run_robustness_experiments(seed)

        return {
            "ablation_matrix": abl_matrix,
            "feature_count_ablation": feat_ablation,
            "circuit_depth_ablation": depth_ablation,
            "data_efficiency": data_eff,
            "robustness": robustness,
        }

    # ── 1. Controlled Matrix Ablations A–E ────────────────────────────────────
    def run_ablation_matrix(self, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Executes:
        A: Classical FS + Classical Classifier (MI -> SVM)
        B: QAOA FS + Classical Classifier (QAOA -> SVM)
        C: QAOA FS + VQC (Original Unweighted)
        D: QAOA FS + SMOTE + VQC
        E: QAOA FS + QGAN + VQC
        """
        print("\n>>> Running Controlled Matrix Ablations A–E <<<")
        ablation_cfgs = [
            ("Ablation_A_ClassicalFS_Classical", ModelConfig(
                model_id="ablation_a",
                model_name="A: Classical FS + SVM",
                model_family="compact_classical",
                input_space="selected_8_classical",
                feature_selection_method="mutual_info",
                n_features=8,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            )),
            ("Ablation_B_QAOAFS_Classical", ModelConfig(
                model_id="ablation_b",
                model_name="B: QAOA FS + SVM",
                model_family="compact_classical",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                n_features=8,
                hyperparameters={"C": 1.0, "kernel": "rbf", "probability": True}
            )),
            ("Ablation_C_QAOAFS_VQC_Original", ModelConfig(
                model_id="ablation_c",
                model_name="C: QAOA FS + VQC (Original)",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="none",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            )),
            ("Ablation_D_QAOAFS_SMOTE_VQC", ModelConfig(
                model_id="ablation_d",
                model_name="D: QAOA FS + SMOTE + VQC",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="smote",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            )),
            ("Ablation_E_QAOAFS_QGAN_VQC", ModelConfig(
                model_id="ablation_e",
                model_name="E: QAOA FS + QGAN + VQC",
                model_family="hybrid_vqc",
                input_space="selected_8_qaoa",
                feature_selection_method="qaoa",
                imbalance_method="qgan",
                n_features=8,
                n_qubits=8,
                circuit_depth=2
            )),
        ]

        results = []
        for code, mcfg in ablation_cfgs:
            res = self.engine.train_and_evaluate(mcfg, seed=seed)
            res["ablation_code"] = code
            results.append(res)
            print(f"  {code:<32} -> Recall: {res['recall']:.4f} | PR-AUC: {res['pr_auc']:.4f} | FN: {res['false_negatives']}")
        return results

    # ── 2. Feature Count Ablation (4, 6, 8, 10, 12) ──────────────────────────
    def run_feature_count_ablation(self, seed: int = 42) -> List[Dict[str, Any]]:
        """Evaluates compact feature counts k in [4, 6, 8, 10, 12] on Classical SVM and Hybrid VQC."""
        print("\n>>> Running Feature Count Ablation (k = 4, 6, 8, 10, 12) <<<")
        priority_indices = [0, 1, 3, 4, 6, 8, 9, 14, 11, 15, 12, 13]
        results = []

        k_list = [4, 6, 8] if self.config.quick_mode else self.config.ablation_feature_counts

        for k in k_list:
            sel_idx = priority_indices[:k]
            X_tr = self.engine.latent_splits["X_train_latent"][:, sel_idx]
            X_va = self.engine.latent_splits["X_val_latent"][:, sel_idx]
            X_te = self.engine.latent_splits["X_test_latent"][:, sel_idx]
            y_tr = self.engine.splits["y_train"]
            y_va = self.engine.splits["y_val"]
            y_te = self.engine.splits["y_test"]

            # Classical SVM on k features
            t0 = time.perf_counter()
            svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
            svm.fit(X_tr, y_tr)
            train_t_svm = time.perf_counter() - t0
            p_svm = svm.predict_proba(X_te)[:, 1]
            pred_svm = (p_svm >= 0.5).astype(int)
            m_svm = compute_extended_metrics(y_te, pred_svm, p_svm, model_name=f"Classical-SVM (k={k})")

            results.append({
                "model_type": "Classical_SVM",
                "k_features": k,
                "reduction_pct": round((1.0 - k / 30.0) * 100.0, 2),
                "accuracy": m_svm["accuracy"],
                "recall": m_svm["recall"],
                "pr_auc": m_svm["pr_auc"],
                "f1": m_svm["f1"],
                "fn": m_svm["fn"],
                "training_time_s": round(train_t_svm, 4),
                "n_qubits": 0,
            })

            # Hybrid VQC on k qubits
            scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr)
            X_tr_ang = scaler.transform(X_tr)
            X_va_ang = scaler.transform(X_va)
            X_te_ang = scaler.transform(X_te)

            epochs = 6 if self.config.quick_mode else 12
            vqc = VariationalQuantumClassifier(
                n_qubits=k,
                n_layers=2,
                preferred_device=self.config.preferred_device,
                fallback_device=self.config.fallback_device,
                seed=seed,
            )
            t0 = time.perf_counter()
            train_vqc(vqc, X_tr_ang, y_tr, X_va_ang, y_va, epochs=epochs, batch_size=16, lr=0.02, verbose=False)
            train_t_vqc = time.perf_counter() - t0

            p_vqc = vqc.predict_proba(X_te_ang)
            pred_vqc = (p_vqc >= 0.5).astype(int)
            m_vqc = compute_extended_metrics(y_te, pred_vqc, p_vqc, model_name=f"Hybrid-VQC (k={k})")

            results.append({
                "model_type": "Hybrid_VQC",
                "k_features": k,
                "reduction_pct": round((1.0 - k / 30.0) * 100.0, 2),
                "accuracy": m_vqc["accuracy"],
                "recall": m_vqc["recall"],
                "pr_auc": m_vqc["pr_auc"],
                "f1": m_vqc["f1"],
                "fn": m_vqc["fn"],
                "training_time_s": round(train_t_vqc, 4),
                "n_qubits": k,
            })
            print(f"  k={k:<2} -> SVM Recall: {m_svm['recall']:.4f} | VQC Recall: {m_vqc['recall']:.4f} | VQC Time: {train_t_vqc:.2f}s")
        return results

    # ── 3. Circuit Depth Ablation (1, 2, 3 Layers) ───────────────────────────
    def run_circuit_depth_ablation(self, seed: int = 42) -> List[Dict[str, Any]]:
        """Evaluates VQC depth L in [1, 2, 3] on 8 QAOA-selected features."""
        print("\n>>> Running Circuit Depth Ablation (L = 1, 2, 3) <<<")
        X_tr = self.engine.qaoa_selected_splits["X_train_qaoa"]
        X_va = self.engine.qaoa_selected_splits["X_val_qaoa"]
        X_te = self.engine.qaoa_selected_splits["X_test_qaoa"]
        y_tr = self.engine.splits["y_train"]
        y_va = self.engine.splits["y_val"]
        y_te = self.engine.splits["y_test"]

        scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr)
        X_tr_ang = scaler.transform(X_tr)
        X_va_ang = scaler.transform(X_va)
        X_te_ang = scaler.transform(X_te)

        results = []
        depth_list = [1, 2] if self.config.quick_mode else self.config.ablation_circuit_depths

        for layers in depth_list:
            epochs = 6 if self.config.quick_mode else 12
            vqc = VariationalQuantumClassifier(
                n_qubits=8,
                n_layers=layers,
                ansatz_type=self.config.vqc_ansatz,
                entanglement=self.config.vqc_entanglement,
                preferred_device=self.config.preferred_device,
                fallback_device=self.config.fallback_device,
                seed=seed,
            )
            t0 = time.perf_counter()
            train_vqc(vqc, X_tr_ang, y_tr, X_va_ang, y_va, epochs=epochs, batch_size=16, lr=0.02, verbose=False)
            train_t = time.perf_counter() - t0

            p_vqc = vqc.predict_proba(X_te_ang)
            pred_vqc = (p_vqc >= 0.5).astype(int)
            m = compute_extended_metrics(y_te, pred_vqc, p_vqc, model_name=f"VQC (L={layers})")

            results.append({
                "circuit_depth": layers,
                "n_qubits": 8,
                "trainable_parameters": vqc.n_params,
                "quantum_gates": vqc.resource_profile.get("gate_count", layers * 16),
                "accuracy": m["accuracy"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "fn": m["fn"],
                "training_time_s": round(train_t, 4),
            })
            print(f"  Layers={layers} -> Params: {vqc.n_params} | Recall: {m['recall']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Time: {train_t:.2f}s")
        return results

    # ── 4. Data Efficiency Experiment (25%, 50%, 75%, 100%) ──────────────────
    def run_data_efficiency_experiment(self, seed: int = 42) -> List[Dict[str, Any]]:
        """Tests learning curves under restricted training set fractions."""
        print("\n>>> Running Data Efficiency Experiment (25%, 50%, 75%, 100%) <<<")
        X_tr = self.engine.qaoa_selected_splits["X_train_qaoa"]
        X_va = self.engine.qaoa_selected_splits["X_val_qaoa"]
        X_te = self.engine.qaoa_selected_splits["X_test_qaoa"]
        y_tr = self.engine.splits["y_train"]
        y_va = self.engine.splits["y_val"]
        y_te = self.engine.splits["y_test"]

        scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr)
        X_tr_ang = scaler.transform(X_tr)
        X_va_ang = scaler.transform(X_va)
        X_te_ang = scaler.transform(X_te)

        results = []
        fractions = [0.5, 1.0] if self.config.quick_mode else self.config.ablation_data_fractions

        n_total = len(X_tr)
        rng = np.random.RandomState(seed)

        for frac in fractions:
            n_sub = int(round(n_total * frac))
            idx = rng.choice(n_total, size=n_sub, replace=False)

            # Subsampled training data
            sub_X_tr = X_tr[idx]
            sub_y_tr = y_tr[idx]
            sub_X_tr_ang = X_tr_ang[idx]

            # Classical SVM
            svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
            svm.fit(sub_X_tr, sub_y_tr)
            p_svm = svm.predict_proba(X_te)[:, 1]
            pred_svm = (p_svm >= 0.5).astype(int)
            m_svm = compute_extended_metrics(y_te, pred_svm, p_svm)

            # Hybrid VQC
            epochs = 6 if self.config.quick_mode else 12
            vqc = VariationalQuantumClassifier(n_qubits=8, n_layers=2, seed=seed)
            train_vqc(vqc, sub_X_tr_ang, sub_y_tr, X_va_ang, y_va, epochs=epochs, batch_size=16, lr=0.02, verbose=False)
            p_vqc = vqc.predict_proba(X_te_ang)
            pred_vqc = (p_vqc >= 0.5).astype(int)
            m_vqc = compute_extended_metrics(y_te, pred_vqc, p_vqc)

            results.append({
                "data_fraction": frac,
                "n_samples": n_sub,
                "svm_recall": m_svm["recall"],
                "svm_pr_auc": m_svm["pr_auc"],
                "vqc_recall": m_vqc["recall"],
                "vqc_pr_auc": m_vqc["pr_auc"],
            })
            print(f"  Frac={int(frac*100)}% (N={n_sub}) -> SVM Recall: {m_svm['recall']:.4f} | VQC Recall: {m_vqc['recall']:.4f}")
        return results

    # ── 5. Noise & Perturbation Robustness ────────────────────────────────────
    def run_robustness_experiments(self, seed: int = 42) -> Dict[str, Any]:
        """Evaluates input feature perturbation and quantum noise simulation."""
        print("\n>>> Running Robustness Experiments (Input Noise & Quantum Noise) <<<")
        X_tr = self.engine.qaoa_selected_splits["X_train_qaoa"]
        X_va = self.engine.qaoa_selected_splits["X_val_qaoa"]
        X_te = self.engine.qaoa_selected_splits["X_test_qaoa"]
        y_tr = self.engine.splits["y_train"]
        y_va = self.engine.splits["y_val"]
        y_te = self.engine.splits["y_test"]

        # Train reference models
        svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
        svm.fit(X_tr, y_tr)

        scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr)
        X_tr_ang = scaler.transform(X_tr)
        X_va_ang = scaler.transform(X_va)
        X_te_ang = scaler.transform(X_te)

        epochs = 6 if self.config.quick_mode else 12
        vqc = VariationalQuantumClassifier(n_qubits=8, n_layers=2, seed=seed)
        train_vqc(vqc, X_tr_ang, y_tr, X_va_ang, y_va, epochs=epochs, batch_size=16, lr=0.02, verbose=False)

        # Part A: Input feature noise perturbation
        sigmas = [0.0, 0.05, 0.1] if self.config.quick_mode else self.config.robustness_noise_sigmas
        perturbation_results = []
        rng = np.random.RandomState(seed)

        for sig in sigmas:
            noise = rng.normal(0.0, sig, size=X_te.shape) if sig > 0 else np.zeros_like(X_te)
            X_te_noisy = X_te + noise
            X_te_ang_noisy = scaler.transform(X_te_noisy)

            # SVM evaluation
            p_svm = svm.predict_proba(X_te_noisy)[:, 1]
            m_svm = compute_extended_metrics(y_te, (p_svm >= 0.5).astype(int), p_svm)

            # VQC evaluation
            p_vqc = vqc.predict_proba(X_te_ang_noisy)
            m_vqc = compute_extended_metrics(y_te, (p_vqc >= 0.5).astype(int), p_vqc)

            perturbation_results.append({
                "noise_sigma": sig,
                "svm_accuracy": m_svm["accuracy"],
                "svm_recall": m_svm["recall"],
                "svm_pr_auc": m_svm["pr_auc"],
                "vqc_accuracy": m_vqc["accuracy"],
                "vqc_recall": m_vqc["recall"],
                "vqc_pr_auc": m_vqc["pr_auc"],
            })
            print(f"  Sigma={sig:.2f} -> SVM Recall: {m_svm['recall']:.4f} | VQC Recall: {m_vqc['recall']:.4f}")

        # Part B: Quantum Simulator Noise Analysis
        # Evaluate clean vs finite shot noise
        quantum_noise_results = []
        # Clean statevector
        p_clean = vqc.predict_proba(X_te_ang)
        m_clean = compute_extended_metrics(y_te, (p_clean >= 0.5).astype(int), p_clean)
        quantum_noise_results.append({
            "simulation_mode": "Ideal Statevector (Analytic)",
            "shots": None,
            "depolarizing_prob": 0.0,
            "accuracy": m_clean["accuracy"],
            "recall": m_clean["recall"],
            "pr_auc": m_clean["pr_auc"],
        })

        shots_list = [1024] if self.config.quick_mode else self.config.quantum_noise_shots
        for s in shots_list:
            noisy_qnode = build_noisy_vqc_qnode(n_qubits=8, n_layers=2, noise_type="shots", shots=s)
            with torch.no_grad():
                weights = vqc.weights
                t_in = torch.tensor(X_te_ang, dtype=torch.float32)
                expvals = [float(noisy_qnode(t_in[i], weights)) for i in range(len(t_in))]
                expvals = torch.tensor(expvals, dtype=torch.float32)
                logits = vqc.head(expvals.unsqueeze(1)).squeeze(1)
                p_shots = torch.sigmoid(logits).numpy()

            m_shot = compute_extended_metrics(y_te, (p_shots >= 0.5).astype(int), p_shots)
            quantum_noise_results.append({
                "simulation_mode": f"Finite Shot Noise ({s} shots)",
                "shots": s,
                "depolarizing_prob": 0.0,
                "accuracy": m_shot["accuracy"],
                "recall": m_shot["recall"],
                "pr_auc": m_shot["pr_auc"],
            })
            print(f"  Shots={s} -> Recall: {m_shot['recall']:.4f} | PR-AUC: {m_shot['pr_auc']:.4f}")

        return {
            "feature_perturbation": perturbation_results,
            "quantum_simulation_noise": quantum_noise_results,
        }
