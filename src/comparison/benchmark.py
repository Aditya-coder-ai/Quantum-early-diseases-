"""
Core Benchmark Engine for Part 9: Classical vs Hybrid Comparison.
Executes standardized training, threshold optimization, inference profiling,
and evaluation across classical, deep learning, compact classical, and hybrid VQC models.
"""
from __future__ import annotations

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import torch
import torch.nn as nn

from configs.config import PROJECT_ROOT
from src.comparison.config import ModelConfig, ComparisonConfig
from src.comparison.metrics import (
    compute_extended_metrics, tune_decision_threshold,
    compute_calibration_curve_data, ResourceProfiler
)
from src.comparison.fairness import FairnessAuditor
from src.imbalance.classical import smote_oversample, compute_class_weights
from src.imbalance.qgan import QuantumGAN
from src.vqc.encoding import AngleScaler
from src.vqc.model import VariationalQuantumClassifier
from src.vqc.training import train_vqc
from src.models.autoencoder import Autoencoder


class ModelBenchmarkEngine:
    """
    Standardized benchmark execution engine enforcing:
    1. Identical held-out test split for all models.
    2. Strict zero-leakage training.
    3. Proper validation-only threshold tuning.
    4. Hardware & simulator profiling.
    """
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.dataset_dir = os.path.join(PROJECT_ROOT, config.dataset_dir)
        self.artifacts_dir = os.path.join(PROJECT_ROOT, config.artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "models"), exist_ok=True)

        # Load canonical data splits
        self.splits = self._load_canonical_splits()
        self.auditor = FairnessAuditor(
            canonical_test_x=self.splits["X_test_raw"],
            canonical_test_y=self.splits["y_test"]
        )

        # Precompute/load latent and selected representations
        self.latent_splits = self._load_or_compute_latent()
        self.qaoa_selected_splits = self._load_or_compute_qaoa_selected()
        self.classical_selected_splits = self._compute_classical_selected()

    def _load_canonical_splits(self) -> Dict[str, np.ndarray]:
        """Loads canonical preprocessed splits from data/processed."""
        X_tr = pd.read_csv(os.path.join(self.dataset_dir, "X_train.csv")).values
        X_va = pd.read_csv(os.path.join(self.dataset_dir, "X_val.csv")).values
        X_te = pd.read_csv(os.path.join(self.dataset_dir, "X_test.csv")).values
        y_tr = pd.read_csv(os.path.join(self.dataset_dir, "y_train.csv")).values.ravel()
        y_va = pd.read_csv(os.path.join(self.dataset_dir, "y_val.csv")).values.ravel()
        y_te = pd.read_csv(os.path.join(self.dataset_dir, "y_test.csv")).values.ravel()

        return {
            "X_train_raw": X_tr, "X_val_raw": X_va, "X_test_raw": X_te,
            "y_train": y_tr, "y_val": y_va, "y_test": y_te
        }

    def _load_or_compute_latent(self) -> Dict[str, np.ndarray]:
        """Extracts 16D latent representations using the trained PyTorch Autoencoder or precomputed files."""
        latent_tr = os.path.join(PROJECT_ROOT, "results", "latent_features_train.csv")
        latent_va = os.path.join(PROJECT_ROOT, "results", "latent_features_val.csv")
        latent_te = os.path.join(PROJECT_ROOT, "results", "latent_features_test.csv")
        if os.path.exists(latent_tr) and os.path.exists(latent_va) and os.path.exists(latent_te):
            return {
                "X_train_latent": pd.read_csv(latent_tr).values,
                "X_val_latent": pd.read_csv(latent_va).values,
                "X_test_latent": pd.read_csv(latent_te).values,
            }

        ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder.pth")
        ae = Autoencoder(input_dim=30, hidden_dims=[64, 32], latent_dim=16)

        if os.path.exists(ae_path):
            state = torch.load(ae_path, map_location="cpu", weights_only=True)
            ae.load_state_dict(state)
        ae.eval()

        with torch.no_grad():
            z_tr = ae.encode(torch.tensor(self.splits["X_train_raw"], dtype=torch.float32)).numpy()
            z_va = ae.encode(torch.tensor(self.splits["X_val_raw"], dtype=torch.float32)).numpy()
            z_te = ae.encode(torch.tensor(self.splits["X_test_raw"], dtype=torch.float32)).numpy()

        return {"X_train_latent": z_tr, "X_val_latent": z_va, "X_test_latent": z_te}

    def _load_or_compute_qaoa_selected(self) -> Dict[str, np.ndarray]:
        """Loads or slices QAOA 8-qubit selected latent features."""
        # 1. Check features/selected/qaoa
        qaoa_dir = os.path.join(PROJECT_ROOT, self.config.features_dir, "qaoa")
        tr_file = os.path.join(qaoa_dir, "X_train_selected.csv")
        if os.path.exists(tr_file):
            s_tr = pd.read_csv(tr_file).values
            s_va = pd.read_csv(os.path.join(qaoa_dir, "X_val_selected.csv")).values
            s_te = pd.read_csv(os.path.join(qaoa_dir, "X_test_selected.csv")).values
            return {"X_train_qaoa": s_tr, "X_val_qaoa": s_va, "X_test_qaoa": s_te}

        # 2. Check results/selected_features_*.csv
        sel_tr = os.path.join(PROJECT_ROOT, "results", "selected_features_train.csv")
        sel_va = os.path.join(PROJECT_ROOT, "results", "selected_features_val.csv")
        sel_te = os.path.join(PROJECT_ROOT, "results", "selected_features_test.csv")
        if os.path.exists(sel_tr) and os.path.exists(sel_va) and os.path.exists(sel_te):
            return {
                "X_train_qaoa": pd.read_csv(sel_tr).values,
                "X_val_qaoa": pd.read_csv(sel_va).values,
                "X_test_qaoa": pd.read_csv(sel_te).values,
            }

        # 3. Fallback to canonical indices [0, 1, 3, 4, 6, 8, 9, 14]
        q_idx = [0, 1, 3, 4, 6, 8, 9, 14]
        z_tr = self.latent_splits["X_train_latent"][:, q_idx]
        z_va = self.latent_splits["X_val_latent"][:, q_idx]
        z_te = self.latent_splits["X_test_latent"][:, q_idx]
        return {"X_train_qaoa": z_tr, "X_val_qaoa": z_va, "X_test_qaoa": z_te}

    def _compute_classical_selected(self) -> Dict[str, np.ndarray]:
        """Computes classical 8-feature subset via Mutual Information strictly on train split."""
        selector = SelectKBest(mutual_info_classif, k=8)
        z_tr = self.latent_splits["X_train_latent"]
        y_tr = self.splits["y_train"]

        s_tr = selector.fit_transform(z_tr, y_tr)
        s_va = selector.transform(self.latent_splits["X_val_latent"])
        s_te = selector.transform(self.latent_splits["X_test_latent"])
        return {"X_train_mi": s_tr, "X_val_mi": s_va, "X_test_mi": s_te}

    def _get_input_arrays(self, input_space: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maps input_space string to feature matrices."""
        if input_space == "raw_30":
            return self.splits["X_train_raw"], self.splits["X_val_raw"], self.splits["X_test_raw"]
        elif input_space == "latent_16":
            return self.latent_splits["X_train_latent"], self.latent_splits["X_val_latent"], self.latent_splits["X_test_latent"]
        elif input_space == "selected_8_classical":
            return self.classical_selected_splits["X_train_mi"], self.classical_selected_splits["X_val_mi"], self.classical_selected_splits["X_test_mi"]
        elif input_space == "selected_8_qaoa":
            return self.qaoa_selected_splits["X_train_qaoa"], self.qaoa_selected_splits["X_val_qaoa"], self.qaoa_selected_splits["X_test_qaoa"]
        else:
            raise ValueError(f"Unknown input space: {input_space}")

    def train_and_evaluate(self, model_cfg: ModelConfig, seed: int = 42) -> Dict[str, Any]:
        """
        Executes a single controlled training and evaluation lifecycle for a model configuration.
        """
        profiler = ResourceProfiler()
        profiler.start()

        X_tr, X_va, X_te = self._get_input_arrays(model_cfg.input_space)
        y_tr = self.splits["y_train"]
        y_va = self.splits["y_val"]
        y_te = self.splits["y_test"]

        # Run scientific fairness audit
        self.auditor.audit_test_split(model_cfg.model_id, X_te, y_te)
        self.auditor.audit_imbalance_isolation(model_cfg.model_id, len(X_va), len(X_te))

        # Handle class imbalance on training split ONLY
        X_tr_fit, y_tr_fit = X_tr, y_tr
        class_weights = None

        if model_cfg.imbalance_method == "smote":
            X_tr_fit, y_tr_fit, _ = smote_oversample(X_tr, y_tr, ratio=1.0, seed=seed)
        elif model_cfg.imbalance_method == "class_weight":
            class_weights = compute_class_weights(y_tr)
        elif model_cfg.imbalance_method == "qgan":
            deficit = int(round((np.sum(y_tr == 1) - np.sum(y_tr == 0)) * 1.0))
            X_min = X_tr[y_tr == 0]
            qgan = QuantumGAN(n_qubits=X_tr.shape[1], n_layers=2, seed=seed)
            qgan.train(X_min, epochs=15, batch_size=16, verbose=False)
            X_synth = qgan.generate(deficit, seed=seed)
            X_tr_fit = np.vstack([X_tr, X_synth])
            y_tr_fit = np.hstack([y_tr, np.full(deficit, 0, dtype=y_tr.dtype)])

        # ── 1. Train Model ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        trained_model, quantum_profile = self._fit_model(model_cfg, X_tr_fit, y_tr_fit, X_va, y_va, class_weights, seed)
        train_time_s = time.perf_counter() - t0

        # ── 2. Validation & Threshold Tuning ─────────────────────────────────
        val_probs = self._predict_proba(trained_model, model_cfg, X_va, X_tr_fit)
        
        if self.config.threshold_tuning:
            best_th, val_metrics, _ = tune_decision_threshold(
                y_val=y_va,
                y_val_probs=val_probs,
                metric=self.config.threshold_metric,
                min_th=self.config.min_threshold,
                max_th=self.config.max_threshold,
                n_steps=self.config.threshold_steps
            )
            self.auditor.audit_threshold_protocol(model_cfg.model_id, best_th, tuned_on_val=True)
        else:
            best_th = 0.5
            self.auditor.audit_threshold_protocol(model_cfg.model_id, 0.5, tuned_on_val=False)

        # ── 3. Final Test Evaluation (Evaluated ONCE with fixed threshold) ────
        t_infer_start = time.perf_counter()
        # Repeat 3 times to compute robust latency average
        for _ in range(3):
            _ = self._predict_proba(trained_model, model_cfg, X_te, X_tr_fit)
        t_infer_tot = time.perf_counter() - t_infer_start
        test_latency_ms = (t_infer_tot / (3 * len(X_te))) * 1000.0

        test_probs = self._predict_proba(trained_model, model_cfg, X_te, X_tr_fit)
        test_preds = (test_probs >= best_th).astype(int)

        test_metrics = compute_extended_metrics(
            y_true=y_te,
            y_pred=test_preds,
            y_probs=test_probs,
            disease_class=0,
            model_name=model_cfg.model_name,
            threshold=best_th
        )

        cal_data = compute_calibration_curve_data(y_true=y_te, y_probs=test_probs, disease_class=0)
        prof_res = profiler.stop()

        # Save model
        save_path = os.path.join(self.artifacts_dir, "models", f"{model_cfg.model_id}_seed{seed}.joblib")
        if model_cfg.model_family == "hybrid_vqc":
            save_path = os.path.join(self.artifacts_dir, "models", f"{model_cfg.model_id}_seed{seed}.pt")
            trained_model.save(save_path)
        else:
            joblib.dump(trained_model, save_path)

        result = {
            "model_id": model_cfg.model_id,
            "model_name": model_cfg.model_name,
            "model_family": model_cfg.model_family,
            "input_space": model_cfg.input_space,
            "feature_selection": model_cfg.feature_selection_method,
            "imbalance_method": model_cfg.imbalance_method,
            "n_features": model_cfg.n_features,
            "feature_reduction_pct": round((1.0 - (model_cfg.n_features / 30.0)) * 100.0, 2),
            "seed": seed,
            "best_threshold": round(best_th, 4),
            "training_time_s": round(train_time_s, 4),
            "inference_latency_ms": round(test_latency_ms, 4),
            "peak_memory_mb": prof_res["peak_memory_mb"],
            "model_path": save_path,
            # Quantum Resource Metrics
            "n_qubits": quantum_profile.get("n_qubits", 0),
            "circuit_depth": quantum_profile.get("circuit_depth", 0),
            "quantum_gates": quantum_profile.get("gate_count", 0),
            "trainable_parameters": quantum_profile.get("trainable_parameters", 0),
            "shots": quantum_profile.get("shots", None),
            "backend": quantum_profile.get("backend", "Classical CPU"),
            # Clinical Test Metrics
            **test_metrics,
            # Raw vectors for curve plotting
            "test_probs": test_probs.tolist(),
            "test_preds": test_preds.tolist(),
            "calibration_curve": cal_data,
        }
        return result

    def _fit_model(
        self,
        cfg: ModelConfig,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        class_weights: Optional[Dict[int, float]],
        seed: int
    ) -> Tuple[Any, Dict[str, Any]]:
        """Instantiates and fits the designated model."""
        hp = cfg.hyperparameters.copy()

        m_id = cfg.model_id.lower()

        if "logreg" in m_id or "logistic" in m_id:
            clf = LogisticRegression(random_state=seed, **hp)
            clf.fit(X_tr, y_tr)
            params = X_tr.shape[1] + 1
            return clf, {"trainable_parameters": params, "backend": "scikit-learn (CPU)"}

        elif "svm" in m_id or m_id.startswith("ablation_a") or m_id.startswith("ablation_b"):
            clf = SVC(random_state=seed, **hp)
            clf.fit(X_tr, y_tr)
            params = len(clf.support_vectors_) * X_tr.shape[1]
            return clf, {"trainable_parameters": params, "backend": "scikit-learn (CPU)"}

        elif "rf" in m_id or "forest" in m_id:
            clf = RandomForestClassifier(random_state=seed, **hp)
            clf.fit(X_tr, y_tr)
            params = sum(tree.tree_.node_count for tree in clf.estimators_)
            return clf, {"trainable_parameters": params, "backend": "scikit-learn (CPU)"}

        elif "mlp" in m_id or "neural" in m_id:
            clf = MLPClassifier(random_state=seed, **hp)
            clf.fit(X_tr, y_tr)
            params = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
            return clf, {"trainable_parameters": params, "backend": "scikit-learn MLP (CPU)"}

        elif cfg.model_family == "hybrid_vqc" or "vqc" in m_id or m_id.startswith("ablation_"):
            scaler = AngleScaler(target_range=(0.0, np.pi))
            scaler.fit(X_tr)
            X_tr_ang = scaler.transform(X_tr)
            X_va_ang = scaler.transform(X_va)

            epochs = 8 if self.config.quick_mode else self.config.vqc_epochs
            patience = 4 if self.config.quick_mode else self.config.vqc_patience

            vqc = VariationalQuantumClassifier(
                n_qubits=cfg.n_qubits,
                n_layers=cfg.circuit_depth or self.config.vqc_layers,
                ansatz_type=self.config.vqc_ansatz,
                entanglement=self.config.vqc_entanglement,
                preferred_device=self.config.preferred_device,
                fallback_device=self.config.fallback_device,
                diff_method=self.config.diff_method,
                shots=self.config.vqc_shots,
                seed=seed,
            )

            train_vqc(
                vqc,
                X_train=X_tr_ang,
                y_train=y_tr,
                X_val=X_va_ang,
                y_val=y_va,
                epochs=epochs,
                batch_size=self.config.vqc_batch_size,
                lr=self.config.vqc_lr,
                class_weights=class_weights,
                early_stopping_patience=patience,
                verbose=False,
            )

            res = vqc.resource_profile.copy()
            res.update({
                "n_qubits": cfg.n_qubits,
                "circuit_depth": cfg.circuit_depth or self.config.vqc_layers,
                "trainable_parameters": vqc.n_params,
                "backend": f"PennyLane {self.config.fallback_device} (Simulation)",
                "shots": self.config.vqc_shots or "Analytic (statevector)"
            })
            # Store scaler on the model instance for inference
            vqc._angle_scaler = scaler
            return vqc, res

        else:
            raise ValueError(f"Unsupported model: {cfg.model_id}")

    def _predict_proba(self, model: Any, cfg: ModelConfig, X: np.ndarray, X_tr_fit: np.ndarray) -> np.ndarray:
        """Standardized probability output for Class 1 (Benign)."""
        if cfg.model_family == "hybrid_vqc":
            scaler = getattr(model, "_angle_scaler", None)
            if scaler is None:
                scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr_fit)
            X_ang = scaler.transform(X)
            return model.predict_proba(X_ang)
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                return 1.0 / (1.0 + np.exp(-df))
            else:
                return model.predict(X).astype(float)
