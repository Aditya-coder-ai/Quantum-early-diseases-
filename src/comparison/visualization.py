"""
Publication-Quality Visualization Suite for Part 9: Classical vs Hybrid Comparison.
Generates all 13 required scientific figures with clinical annotations.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, precision_recall_curve


class ComparisonVisualizer:
    """
    Renders high-resolution comparative figures in experiments/comparison/plots/.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font="sans-serif")
        self.palette = sns.color_palette("deep")

    def generate_all_plots(
        self,
        benchmark_results: List[Dict[str, Any]],
        multiseed_summary: Dict[str, Any],
        ablation_results: Dict[str, Any],
        explainability_results: Dict[str, Any],
        y_test: np.ndarray,
    ) -> List[str]:
        """Orchestrates generation of all 13 research figures."""
        saved_plots = []
        saved_plots.append(self.plot_model_performance(benchmark_results))
        saved_plots.append(self.plot_recall_comparison(benchmark_results))
        saved_plots.append(self.plot_pr_auc_comparison(benchmark_results))
        saved_plots.append(self.plot_roc_curves(benchmark_results, y_test))
        saved_plots.append(self.plot_pr_curves(benchmark_results, y_test))
        saved_plots.append(self.plot_confusion_matrices(benchmark_results))
        saved_plots.append(self.plot_calibration_curves(benchmark_results))
        saved_plots.append(self.plot_feature_count_vs_performance(ablation_results.get("feature_count_ablation", [])))
        saved_plots.append(self.plot_runtime_comparison(benchmark_results))
        saved_plots.append(self.plot_feature_reduction(benchmark_results))
        saved_plots.append(self.plot_vqc_depth(ablation_results.get("circuit_depth_ablation", [])))
        saved_plots.append(self.plot_noise_robustness(ablation_results.get("robustness", {})))
        saved_plots.append(self.plot_explainability_comparison(explainability_results))
        return saved_plots

    # 1. Model Performance Comparison
    def plot_model_performance(self, results: List[Dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = [r["model_name"] for r in results]
        metrics = ["recall", "specificity", "f1", "roc_auc", "pr_auc"]
        labels = ["Recall (Sens.)", "Specificity", "F1 Score", "ROC-AUC", "PR-AUC"]
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, (m_key, m_label) in enumerate(zip(metrics, labels)):
            vals = [r.get(m_key, 0.0) or 0.0 for r in results]
            ax.bar(x + (i - 2) * width, vals, width, label=m_label)

        ax.set_title("Comprehensive Model Performance Comparison across Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim([0.7, 1.05])
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_performance_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 2. Recall Comparison (Emphasizing False Negatives)
    def plot_recall_comparison(self, results: List[Dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        models = [r["model_name"] for r in results]
        recalls = [r["recall"] for r in results]
        fns = [r["false_negatives"] for r in results]

        bars = ax.bar(models, recalls, color=sns.color_palette("Blues_r", len(models)))
        ax.set_title("Malignant Recall (Clinical Sensitivity) & False Negative Count", fontsize=13, fontweight="bold")
        ax.set_ylabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylim([0.8, 1.05])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)

        for bar, fn in zip(bars, fns):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}\n(FN={fn})",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "recall_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 3. PR-AUC Comparison
    def plot_pr_auc_comparison(self, results: List[Dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        models = [r["model_name"] for r in results]
        pr_aucs = [r.get("pr_auc", 0.0) or 0.0 for r in results]

        bars = ax.bar(models, pr_aucs, color=sns.color_palette("crest", len(models)))
        ax.set_title("Precision-Recall AUC Comparison (Minority Malignant Class)", fontsize=13, fontweight="bold")
        ax.set_ylabel("PR-AUC", fontsize=11)
        ax.set_ylim([0.85, 1.02])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "pr_auc_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 4. ROC Curves
    def plot_roc_curves(self, results: List[Dict[str, Any]], y_test: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(8, 7))
        y_dis = (np.asarray(y_test) == 0).astype(int)

        for r in results:
            if "test_probs" in r:
                p_dis = 1.0 - np.asarray(r["test_probs"])
                fpr, tpr, _ = roc_curve(y_dis, p_dis)
                auc_val = r.get("roc_auc", 0.0)
                ax.plot(fpr, tpr, label=f"{r['model_name']} (AUC = {auc_val:.3f})", lw=2)

        ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)", lw=1.5)
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "roc_curves_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 5. Precision-Recall Curves
    def plot_pr_curves(self, results: List[Dict[str, Any]], y_test: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(8, 7))
        y_dis = (np.asarray(y_test) == 0).astype(int)
        baseline_rate = float(np.mean(y_dis))

        for r in results:
            if "test_probs" in r:
                p_dis = 1.0 - np.asarray(r["test_probs"])
                prec, rec, _ = precision_recall_curve(y_dis, p_dis)
                pr_auc = r.get("pr_auc", 0.0)
                ax.plot(rec, prec, label=f"{r['model_name']} (PR-AUC = {pr_auc:.3f})", lw=2)

        ax.axhline(baseline_rate, color="gray", linestyle="--", label=f"No Skill (Prevalence = {baseline_rate:.2f})")
        ax.set_title("Precision-Recall (PR) Curves for Early Malignancy", fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall (Clinical Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([0.3, 1.05])
        ax.legend(loc="lower left", fontsize=8, frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "precision_recall_curves_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 6. Confusion Matrices
    def plot_confusion_matrices(self, results: List[Dict[str, Any]]) -> str:
        # Pick 4 representative models
        selected_ids = ["logreg_raw", "rf_raw", "compact_svm_qaoa_fs", "hybrid_vqc_smote"]
        candidates = [r for r in results if r["model_id"] in selected_ids]
        if len(candidates) < 4:
            candidates = results[:4]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, (ax, r) in enumerate(zip(axes, candidates)):
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        annot_kws={"size": 13, "weight": "bold"})
            ax.set_title(f"{r['model_name']}\n(FN={r['fn']} False Negatives)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label (0: Benign, 1: Malignant)", fontsize=9)
            ax.set_ylabel("True Label (0: Benign, 1: Malignant)", fontsize=9)
            ax.set_xticklabels(["Benign", "Malignant"])
            ax.set_yticklabels(["Benign", "Malignant"])

        plt.tight_layout()
        path = os.path.join(self.output_dir, "confusion_matrices_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 7. Calibration Curves
    def plot_calibration_curves(self, results: List[Dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.5)

        for r in results:
            cal = r.get("calibration_curve")
            if cal and len(cal.get("prob_true", [])) > 0:
                ax.plot(cal["prob_pred"], cal["prob_true"], "s-", label=f"{r['model_name']} (Brier={r.get('brier_score', 0):.3f})", lw=1.8)

        ax.set_title("Reliability Diagrams (Calibration Curves)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability of Malignancy", fontsize=11)
        ax.set_ylabel("Empirical Fraction of Malignant Cases", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "calibration_curves_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 8. Feature Count vs Performance
    def plot_feature_count_vs_performance(self, feat_ablations: List[Dict[str, Any]]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if not feat_ablations:
            ax1.text(0.5, 0.5, "No feature count data", ha='center')
        else:
            svm_data = [x for x in feat_ablations if x["model_type"] == "Classical_SVM"]
            vqc_data = [x for x in feat_ablations if x["model_type"] == "Hybrid_VQC"]

            if svm_data and vqc_data:
                k_svm = [x["k_features"] for x in svm_data]
                rec_svm = [x["recall"] for x in svm_data]
                pr_svm = [x["pr_auc"] for x in svm_data]

                k_vqc = [x["k_features"] for x in vqc_data]
                rec_vqc = [x["recall"] for x in vqc_data]
                pr_vqc = [x["pr_auc"] for x in vqc_data]

                ax1.plot(k_svm, rec_svm, "o-", label="Classical SVM", lw=2, color="tab:blue")
                ax1.plot(k_vqc, rec_vqc, "s-", label="Hybrid VQC", lw=2, color="tab:green")
                ax1.set_title("Recall vs Feature Count (k)", fontsize=12, fontweight="bold")
                ax1.set_xlabel("Selected Features (k)", fontsize=10)
                ax1.set_ylabel("Malignant Recall", fontsize=10)
                ax1.legend()

                ax2.plot(k_svm, pr_svm, "o-", label="Classical SVM", lw=2, color="tab:blue")
                ax2.plot(k_vqc, pr_vqc, "s-", label="Hybrid VQC", lw=2, color="tab:green")
                ax2.set_title("PR-AUC vs Feature Count (k)", fontsize=12, fontweight="bold")
                ax2.set_xlabel("Selected Features (k)", fontsize=10)
                ax2.set_ylabel("PR-AUC", fontsize=10)
                ax2.legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_count_vs_performance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 9. Runtime Comparison
    def plot_runtime_comparison(self, results: List[Dict[str, Any]]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        models = [r["model_name"] for r in results]
        train_times = [r.get("training_time_s", 0.0) for r in results]
        latencies = [r.get("inference_latency_ms", 0.0) for r in results]

        y_pos = np.arange(len(models))

        ax1.barh(y_pos, train_times, color=sns.color_palette("mako", len(models)))
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models, fontsize=9)
        ax1.set_xscale("log")
        ax1.set_title("Training Duration (Log Scale Seconds)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Seconds (log scale)", fontsize=10)

        ax2.barh(y_pos, latencies, color=sns.color_palette("flare", len(models)))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(models, fontsize=9)
        ax2.set_xscale("log")
        ax2.set_title("Inference Latency per Sample (Log Scale ms)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Milliseconds / Sample (log scale)", fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "runtime_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 10. Feature Reduction Comparison
    def plot_feature_reduction(self, results: List[Dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        reductions = [r.get("feature_reduction_pct", 0.0) for r in results]
        recalls = [r["recall"] for r in results]
        models = [r["model_name"] for r in results]

        scatter = ax.scatter(reductions, recalls, s=150, c=recalls, cmap="viridis", edgecolors="black", zorder=3)
        ax.set_title("Feature Space Compression vs Sensitivity Retention", fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature Reduction from 30 Raw Dimensions (%)", fontsize=11)
        ax.set_ylabel("Malignant Recall (Sensitivity)", fontsize=11)
        ax.set_ylim([0.85, 1.05])
        plt.colorbar(scatter, label="Recall Score")

        for red, rec, name in zip(reductions, recalls, models):
            ax.annotate(name, (red + 1.0, rec), fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_reduction_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 11. VQC Depth vs Performance
    def plot_vqc_depth(self, depth_data: List[Dict[str, Any]]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        if not depth_data:
            ax1.text(0.5, 0.5, "No depth data", ha='center')
        else:
            layers = [x["circuit_depth"] for x in depth_data]
            rec = [x["recall"] for x in depth_data]
            pr = [x["pr_auc"] for x in depth_data]
            params = [x["trainable_parameters"] for x in depth_data]

            ax1.plot(layers, rec, "o-", label="Recall", color="tab:blue", lw=2)
            ax1.plot(layers, pr, "s-", label="PR-AUC", color="tab:green", lw=2)
            ax1.set_title("Performance vs Circuit Depth (Layers)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Variational Layers", fontsize=10)
            ax1.set_ylabel("Score", fontsize=10)
            ax1.set_xticks(layers)
            ax1.legend()

            ax2.bar(layers, params, color="tab:purple", width=0.4)
            ax2.set_title("Parameter Count vs Circuit Depth", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Variational Layers", fontsize=10)
            ax2.set_ylabel("Trainable Weights", fontsize=10)
            ax2.set_xticks(layers)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "vqc_depth_vs_performance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 12. Noise Robustness Comparison
    def plot_noise_robustness(self, rob_data: Dict[str, Any]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pert_data = rob_data.get("feature_perturbation", [])
        q_noise_data = rob_data.get("quantum_simulation_noise", [])

        if pert_data:
            sigmas = [x["noise_sigma"] for x in pert_data]
            svm_rec = [x["svm_recall"] for x in pert_data]
            vqc_rec = [x["vqc_recall"] for x in pert_data]

            ax1.plot(sigmas, svm_rec, "o-", label="Classical SVM", lw=2, color="tab:blue")
            ax1.plot(sigmas, vqc_rec, "s-", label="Hybrid VQC", lw=2, color="tab:green")
            ax1.set_title("Input Perturbation Robustness (Noise Sigma)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Gaussian Noise Sigma (Std Dev)", fontsize=10)
            ax1.set_ylabel("Malignant Recall", fontsize=10)
            ax1.set_ylim([0.7, 1.05])
            ax1.legend()

        if q_noise_data:
            labels = [x["simulation_mode"] for x in q_noise_data]
            recs = [x["recall"] for x in q_noise_data]
            ax2.bar(labels, recs, color=sns.color_palette("muted", len(labels)), width=0.5)
            ax2.set_title("Quantum Simulation Noise Impact on Recall", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Malignant Recall", fontsize=10)
            ax2.set_ylim([0.75, 1.05])
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "noise_robustness_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 13. Explainability Comparison
    def plot_explainability_comparison(self, exp_data: Dict[str, Any]) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        table = exp_data.get("comparison_table", [])
        if not table:
            ax.text(0.5, 0.5, "No explainability comparison data", ha='center')
        else:
            features = [x["feature_name"] for x in table]
            cls_vals = [x["classical_mean_abs_shap"] for x in table]
            vqc_vals = [x["vqc_mean_abs_shap"] for x in table]

            y = np.arange(len(features))
            height = 0.35

            ax.barh(y - height/2, cls_vals, height, label="Classical SVM Mean |SHAP|", color="tab:blue")
            ax.barh(y + height/2, vqc_vals, height, label="Hybrid VQC Mean |SHAP|", color="tab:green")

            ax.set_yticks(y)
            ax.set_yticklabels(features, fontsize=10)
            ax.invert_yaxis()
            ax.set_title(f"Classical vs Hybrid VQC Feature Attribution (Spearman rho = {exp_data.get('spearman_rank_correlation', 0):.3f})",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11)
            ax.legend(frameon=True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "explainability_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path
