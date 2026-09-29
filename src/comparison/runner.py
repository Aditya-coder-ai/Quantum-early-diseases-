"""
Master Comparison Runner for Part 9: Classical vs Hybrid Comparison.
Coordinates benchmark execution, ablations, multi-seed aggregation, statistical testing,
explainability comparison, visualization, and programmatic reporting.
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from configs.config import PROJECT_ROOT
from src.comparison.config import ComparisonConfig, ModelConfig
from src.comparison.benchmark import ModelBenchmarkEngine
from src.comparison.ablations import AblationSuite
from src.comparison.statistical import aggregate_multiseed_metrics, run_paired_statistical_tests
from src.comparison.explainability_comparison import compare_classical_vs_hybrid_explanations
from src.comparison.visualization import ComparisonVisualizer
from src.comparison.reporter import ComparisonReporter
from src.vqc.encoding import AngleScaler


class ComparisonRunner:
    """
    Automated scientific coordinator executing the full Part 9 experimental comparison loop.
    """
    def __init__(self, config: Optional[ComparisonConfig] = None):
        self.config = config or ComparisonConfig()
        self.artifacts_dir = os.path.join(PROJECT_ROOT, self.config.artifacts_dir)
        self.plots_dir = os.path.join(self.artifacts_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.engine = ModelBenchmarkEngine(self.config)
        self.ablation_suite = AblationSuite(self.engine)
        self.visualizer = ComparisonVisualizer(self.plots_dir)
        self.reporter = ComparisonReporter(self.artifacts_dir)

    def run(self) -> Dict[str, Any]:
        """Executes the complete comparative study."""
        t_start = time.perf_counter()
        print("\n" + "=" * 75)
        print("   STARTING PART 9: CLASSICAL VS HYBRID COMPARATIVE STUDY")
        print("=" * 75)
        print(f"Models to evaluate: {len(self.config.models)}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Quick Mode: {self.config.quick_mode}")

        # ── 1. Benchmark Execution across Seeds ──────────────────────────────
        all_seed_results: List[Dict[str, Any]] = []
        primary_results: List[Dict[str, Any]] = []

        # Determine seeds to run (primary seed first, plus others if not quick mode)
        run_seeds = [self.config.primary_seed] if self.config.quick_mode else self.config.seeds

        for seed in run_seeds:
            print(f"\n>>> Running Benchmark Suite for Seed: {seed} <<<")
            for model_cfg in self.config.models:
                print(f"  Training {model_cfg.model_name:<30}...", end="", flush=True)
                res = self.engine.train_and_evaluate(model_cfg, seed=seed)
                all_seed_results.append(res)
                if seed == self.config.primary_seed:
                    primary_results.append(res)
                print(f" Done in {res['training_time_s']:.2f}s | Recall: {res['recall']:.4f} | PR-AUC: {res['pr_auc']:.4f} | FN: {res['false_negatives']}")

        # ── 2. Multi-Seed Statistical Aggregation ────────────────────────────
        print("\n>>> Aggregating Multi-Seed Performance Metrics <<<")
        multiseed_summary = aggregate_multiseed_metrics(all_seed_results)

        # ── 3. Paired Statistical Testing (Classical vs Hybrid) ──────────────
        print(">>> Performing Paired Statistical Significance Testing <<<")
        # Find Classical baseline (SVM) and Hybrid VQC from primary seed
        classical_run = next((r for r in primary_results if r["model_id"] == "compact_svm_qaoa_fs"), primary_results[0])
        hybrid_run = next((r for r in primary_results if r["model_id"] == "hybrid_vqc_smote"), primary_results[-1])

        y_test = self.engine.splits["y_test"]
        stats_results = run_paired_statistical_tests(
            y_true=y_test,
            baseline_probs=np.array(classical_run["test_probs"]),
            baseline_preds=np.array(classical_run["test_preds"]),
            hybrid_probs=np.array(hybrid_run["test_probs"]),
            hybrid_preds=np.array(hybrid_run["test_preds"]),
            baseline_name=classical_run["model_name"],
            hybrid_name=hybrid_run["model_name"],
            disease_class=0
        )
        print(f"  Paired t-test p-value: {stats_results['paired_t_test']['p_value']}")
        print(f"  Wilcoxon p-value:     {stats_results['wilcoxon_test']['p_value']}")
        print(f"  McNemar p-value:      {stats_results['mcnemar_test']['p_value']}")

        # ── 4. Ablation & Robustness Studies ─────────────────────────────────
        ablation_results = self.ablation_suite.run_all_ablations(seed=self.config.primary_seed)

        # ── 5. Explainability Comparison ─────────────────────────────────────
        # Load 8-feature classical reference model and hybrid VQC model
        import joblib
        compact_candidates = [r for r in primary_results if r.get("n_features") == 8 and r.get("model_family") != "hybrid_vqc"]
        if compact_candidates:
            classical_model = joblib.load(compact_candidates[0]["model_path"])
        else:
            from sklearn.svm import SVC
            classical_model = SVC(C=1.0, kernel="rbf", probability=True, random_state=self.config.primary_seed)
            classical_model.fit(self.engine.qaoa_selected_splits["X_train_qaoa"], self.engine.splits["y_train"])

        from src.vqc.model import VariationalQuantumClassifier
        hybrid_vqc = VariationalQuantumClassifier(n_qubits=8, n_layers=2, seed=self.config.primary_seed)
        import torch
        if os.path.exists(hybrid_run["model_path"]):
            hybrid_vqc.load(hybrid_run["model_path"])

        X_tr_qaoa = self.engine.qaoa_selected_splits["X_train_qaoa"]
        X_te_qaoa = self.engine.qaoa_selected_splits["X_test_qaoa"]
        angle_scaler = AngleScaler(target_range=(0.0, np.pi)).fit(X_tr_qaoa)
        feature_names = [f"latent_{i}" for i in [0, 1, 3, 4, 6, 8, 9, 14]]

        n_bg = 20 if self.config.quick_mode else 30
        n_exp = 10 if self.config.quick_mode else 20

        explainability_results = compare_classical_vs_hybrid_explanations(
            classical_model=classical_model,
            hybrid_vqc=hybrid_vqc,
            angle_scaler=angle_scaler,
            X_train=X_tr_qaoa,
            X_test=X_te_qaoa,
            feature_names=feature_names,
            n_background=n_bg,
            n_explain_samples=n_exp,
            seed=self.config.primary_seed
        )
        print(f"  Spearman Rank Correlation: rho = {explainability_results['spearman_rank_correlation']:.4f}")

        # ── 6. Visualizations Generation (All 13 Plots) ──────────────────────
        print("\n>>> Generating All 13 Publication-Quality Research Figures <<<")
        plot_paths = self.visualizer.generate_all_plots(
            benchmark_results=primary_results,
            multiseed_summary=multiseed_summary,
            ablation_results=ablation_results,
            explainability_results=explainability_results,
            y_test=y_test
        )
        for p in plot_paths:
            print(f"  Rendered: {os.path.basename(p)}")

        # ── 7. Reporting & Artifact Export ───────────────────────────────────
        print("\n>>> Generating Comprehensive Markdown & CSV Reports <<<")
        fairness_summary = self.engine.auditor.get_summary()
        report_path = self.reporter.generate_and_save(
            benchmark_results=primary_results,
            multiseed_summary=multiseed_summary,
            statistical_tests=stats_results,
            ablation_results=ablation_results,
            explainability_results=explainability_results,
            fairness_summary=fairness_summary,
            plot_paths=plot_paths
        )

        total_time = time.perf_counter() - t_start
        print("\n" + "=" * 75)
        print(f"   PART 9 COMPARATIVE STUDY COMPLETED IN {total_time:.2f} SECONDS")
        print("=" * 75)

        return {
            "status": "COMPLETE",
            "total_time_s": total_time,
            "report_path": report_path,
            "benchmark_results": primary_results,
            "multiseed_summary": multiseed_summary,
            "statistical_tests": stats_results,
            "ablation_results": ablation_results,
            "fairness_summary": fairness_summary,
            "plot_paths": plot_paths,
        }
