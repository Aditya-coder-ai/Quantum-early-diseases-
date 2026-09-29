"""
Part 7: Master Hybrid Pipeline Orchestrator.
Coordinates end-to-end execution from raw medical dataset to quantum-classified predictions,
enforcing stage contracts, performance profiling, leakage auditing, and artifact serialization.
"""
from __future__ import annotations

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.config import PROJECT_ROOT
from src.pipeline.config import PipelineConfig
from src.pipeline.state import PipelineState, PipelineContext
from src.pipeline.contracts import PipelineReportContract
from src.pipeline.validation import perform_full_leakage_audit
from src.pipeline.stages import (
    run_data_stage, run_preprocessing_stage, run_feature_extraction_stage,
    run_feature_selection_stage, run_imbalance_stage, run_quantum_encoding_stage,
    run_vqc_training_stage, run_test_evaluation_stage
)
from src.vqc.circuit import count_quantum_resources


class HybridPipeline:
    """
    Master Orchestrator for the Hybrid Classical-Quantum Early Disease Detection System.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.fingerprint = self.config.get_fingerprint()
        self.experiment_id = f"{self.config.experiment_name}_{self.fingerprint}"
        self.artifact_dir = os.path.join(self.config.artifacts_dir, self.experiment_id)
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.context = PipelineContext(
            experiment_id=self.experiment_id,
            config_hash=self.fingerprint,
        )

    def run(self) -> PipelineReportContract:
        """
        Execute the complete hybrid classical-quantum pipeline end-to-end.
        """
        print("\n" + "=" * 75)
        print(f"   STARTING HYBRID CLASSICAL-QUANTUM PIPELINE [{self.experiment_id}]")
        print("=" * 75)

        try:
            # ── STAGE 1: Data Loading & Validation Gate ──
            t0 = time.time()
            self.context.transition_to(PipelineState.INITIALIZED, "Initializing data pipeline...")
            raw, split = run_data_stage(self.config)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_1_data_loading", t_stage)
            self.context.transition_to(
                PipelineState.DATA_VALIDATED,
                f"Data loaded & split: Train={split.X_train.shape}, Val={split.X_val.shape}, Test={split.X_test.shape} ({t_stage:.2f}s)"
            )

            # ── STAGE 2: Preprocessing ──
            t0 = time.time()
            prep = run_preprocessing_stage(split, self.config, self.artifact_dir)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_2_preprocessing", t_stage)
            self.context.transition_to(
                PipelineState.PREPROCESSED,
                f"Zero-leakage preprocessing fitted on train and applied to splits ({t_stage:.2f}s)"
            )

            # ── STAGE 3: Feature Extraction (Autoencoder) ──
            t0 = time.time()
            latent = run_feature_extraction_stage(prep, split, self.config, self.artifact_dir)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_3_feature_extraction", t_stage)
            self.context.transition_to(
                PipelineState.FEATURES_EXTRACTED,
                f"Autoencoder compressed 30D features to {latent.latent_dim}D latent space ({t_stage:.2f}s)"
            )

            # ── STAGE 4: Feature Selection (QAOA vs. Classical) ──
            t0 = time.time()
            selected = run_feature_selection_stage(latent, split, self.config, self.artifact_dir)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_4_feature_selection", t_stage)
            self.context.transition_to(
                PipelineState.FEATURES_SELECTED,
                f"Feature selection ({self.config.feature_selection_method}) selected {selected.k} features: {selected.selected_indices} ({t_stage:.2f}s)"
            )

            # ── STAGE 5: Class Imbalance Handling ──
            t0 = time.time()
            imb = run_imbalance_stage(selected, split, self.config)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_5_imbalance_handling", t_stage)
            self.context.transition_to(
                PipelineState.IMBALANCE_HANDLED,
                f"Imbalance handling ({self.config.imbalance_method}) train size: {len(imb.X_train_balanced)} (Val/Test untouched) ({t_stage:.2f}s)"
            )

            # ── STAGE 6: Quantum Feature Encoding ──
            t0 = time.time()
            q_enc = run_quantum_encoding_stage(imb, self.config, self.artifact_dir)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_6_quantum_encoding", t_stage)
            self.context.transition_to(
                PipelineState.FEATURES_SELECTED,
                f"Quantum AngleScaler mapped features to [0, pi] on {q_enc.n_qubits} qubits ({t_stage:.2f}s)"
            )

            # ── STAGE 7: VQC Training & Validation ──
            t0 = time.time()
            vqc_model, val_contract, train_res = run_vqc_training_stage(q_enc, imb, self.config, self.artifact_dir)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_7_vqc_training", t_stage)
            self.context.transition_to(
                PipelineState.VALIDATED,
                f"VQC trained ({train_res['total_epochs_trained']} epochs): Val Acc={val_contract.metrics['accuracy']:.4f}, "
                f"PR-AUC={val_contract.metrics['pr_auc']:.4f}, Minority Recall={val_contract.metrics['minority_recall']:.4f} ({t_stage:.2f}s)"
            )

            # ── STAGE 8: Final Test Evaluation (Evaluated Once) ──
            t0 = time.time()
            test_contract, classical_test = run_test_evaluation_stage(vqc_model, q_enc, imb, self.config)
            t_stage = time.time() - t0
            self.context.record_stage_time("stage_8_test_evaluation", t_stage)
            self.context.transition_to(
                PipelineState.TESTED,
                f"Test evaluation completed: VQC Acc={test_contract.metrics['accuracy']:.4f} vs Classical SVM Acc={classical_test['accuracy']:.4f} ({t_stage:.2f}s)"
            )

            # ── LEAKAGE AUDIT ──
            leakage_audit = perform_full_leakage_audit(raw, split, prep, latent, selected, imb, q_enc)
            for check_name, passed in leakage_audit.items():
                if not passed:
                    raise RuntimeError(f"Data Leakage Audit Failed on check: {check_name}")

            # ── SERIALIZE FINAL REPORT & ARTIFACTS ──
            self.config.save_yaml(os.path.join(self.artifact_dir, "config.yaml"))
            resources = count_quantum_resources(
                self.config.num_selected_features, self.config.vqc_layers,
                self.config.vqc_ansatz, self.config.vqc_entanglement
            )

            comparison_records = [
                {"model": "Classical_Reference_SVM", **classical_test},
                {"model": "Hybrid_VQC", **test_contract.metrics},
            ]
            comp_df = pd.DataFrame(comparison_records)
            comp_df.to_csv(os.path.join(self.artifact_dir, "test_comparison.csv"), index=False)

            total_runtime = self.context.get_total_runtime_s()

            report = PipelineReportContract(
                experiment_id=self.experiment_id,
                config_hash=self.fingerprint,
                status="COMPLETED",
                total_runtime_s=total_runtime,
                stage_timings=self.context.stage_timings,
                validation_metrics=val_contract.metrics,
                test_metrics=test_contract.metrics,
                resource_profile=resources,
                artifacts={
                    "preprocessing": prep.scaler_artifact_path,
                    "autoencoder": latent.model_artifact_path,
                    "feature_selection": selected.artifact_path,
                    "angle_scaler": q_enc.angle_scaler_path,
                    "vqc_checkpoint": os.path.join(self.artifact_dir, "vqc_model.pt"),
                    "config": os.path.join(self.artifact_dir, "config.yaml"),
                    "comparison_table": os.path.join(self.artifact_dir, "test_comparison.csv"),
                },
                leakage_audit=leakage_audit,
            )

            # Save report JSON
            def to_serializable(obj):
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                if isinstance(obj, (np.floating, float)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {str(k): to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [to_serializable(v) for v in obj]
                return obj

            report_dict = {
                "experiment_id": report.experiment_id,
                "config_hash": report.config_hash,
                "status": report.status,
                "total_runtime_s": report.total_runtime_s,
                "stage_timings": report.stage_timings,
                "validation_metrics": report.validation_metrics,
                "test_metrics": report.test_metrics,
                "classical_test_benchmark": classical_test,
                "resource_profile": report.resource_profile,
                "artifacts": report.artifacts,
                "leakage_audit": report.leakage_audit,
            }
            clean_dict = to_serializable(report_dict)
            with open(os.path.join(self.artifact_dir, "final_results.json"), "w") as f:
                json.dump(clean_dict, f, indent=2)

            final_results_row = {
                "experiment_id": self.experiment_id,
                "configuration": self.config.experiment_name,
                "feature_selection": self.config.feature_selection_method,
                "imbalance_method": self.config.imbalance_method,
                "feature_count": self.config.num_selected_features,
                "qubits": resources["n_qubits"],
                "circuit_depth": resources["circuit_depth"],
                "accuracy": test_contract.metrics["accuracy"],
                "precision": test_contract.metrics["precision"],
                "recall": test_contract.metrics["recall"],
                "specificity": test_contract.metrics["specificity"],
                "f1": test_contract.metrics["f1"],
                "roc_auc": test_contract.metrics["roc_auc"],
                "pr_auc": test_contract.metrics["pr_auc"],
                "training_time": self.context.stage_timings.get("stage_7_vqc_training", 0.0),
                "inference_time": test_contract.metrics.get("inference_time_s", 0.0),
            }
            res_df = pd.DataFrame([final_results_row])
            res_df.to_csv(os.path.join(self.artifact_dir, "final_results.csv"), index=False)
            results_dir = os.path.join(PROJECT_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            res_df.to_csv(os.path.join(results_dir, "final_results.csv"), index=False)

            self.context.transition_to(PipelineState.COMPLETED, f"Pipeline finished successfully in {total_runtime:.2f}s!")

            print("\n" + "=" * 75)
            print(f"   HYBRID PIPELINE COMPLETED SUCCESSFULLY [{self.experiment_id}]")
            print(f"   Total Runtime: {total_runtime:.2f}s")
            print(f"   Artifacts saved to: {self.artifact_dir}")
            print(f"   VQC Test Acc: {test_contract.metrics['accuracy']:.4f} | Minority Recall: {test_contract.metrics['minority_recall']:.4f}")
            print(f"   Classical Test Acc: {classical_test['accuracy']:.4f} | Minority Recall: {classical_test['minority_recall']:.4f}")
            print("=" * 75)

            return report

        except Exception as e:
            self.context.fail("pipeline_execution", e)
            raise e
