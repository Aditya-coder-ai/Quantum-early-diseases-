"""
Statistical Significance Analysis and Multi-Seed Aggregation for Part 9.
Computes Mean ± Std, 95% Confidence Intervals, Paired t-tests, Wilcoxon Signed-Rank, and McNemar Tests.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Optional
from collections import defaultdict


def aggregate_multiseed_metrics(
    seed_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates metrics across multiple seeds for each model.
    Computes mean, standard deviation, and 95% confidence intervals.
    """
    grouped = defaultdict(list)
    for r in seed_results:
        grouped[r["model_id"]].append(r)

    summary = {}
    metric_keys = [
        "accuracy", "balanced_accuracy", "recall", "specificity",
        "precision", "f1", "roc_auc", "pr_auc", "false_negatives",
        "training_time_s", "inference_latency_ms"
    ]

    for model_id, runs in grouped.items():
        model_name = runs[0].get("model_name", model_id)
        model_family = runs[0].get("model_family", "unknown")
        n_seeds = len(runs)

        stat_dict: Dict[str, Any] = {
            "model_id": model_id,
            "model_name": model_name,
            "model_family": model_family,
            "n_seeds": n_seeds,
            "metrics": {}
        }

        for k in metric_keys:
            vals = [float(run[k]) for run in runs if run.get(k) is not None]
            if not vals:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            
            # 95% Confidence Interval (using t-distribution if n > 1)
            if len(vals) > 1:
                ci = float(stats.t.ppf(0.975, df=len(vals) - 1) * (std_val / np.sqrt(len(vals))))
            else:
                ci = 0.0

            stat_dict["metrics"][k] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "ci95": round(ci, 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "formatted": f"{mean_val:.4f} ± {std_val:.4f}"
            }

        summary[model_id] = stat_dict

    return summary


def run_paired_statistical_tests(
    y_true: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_preds: np.ndarray,
    hybrid_probs: np.ndarray,
    hybrid_preds: np.ndarray,
    baseline_name: str = "Classical_Baseline",
    hybrid_name: str = "Hybrid_VQC",
    disease_class: int = 0
) -> Dict[str, Any]:
    """
    Executes rigorous paired statistical comparison tests between Classical baseline and Hybrid model.
    1. Paired t-test on Brier error residuals.
    2. Wilcoxon signed-rank test on absolute probability errors.
    3. McNemar's test for paired binary classification contingencies.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_dis = (y_true == disease_class).astype(int)

    base_p_dis = 1.0 - np.asarray(baseline_probs, dtype=float).ravel()
    hyb_p_dis = 1.0 - np.asarray(hybrid_probs, dtype=float).ravel()

    # Brier residuals (squared error for probability of disease)
    base_err = (base_p_dis - y_dis) ** 2
    hyb_err = (hyb_p_dis - y_dis) ** 2

    # 1. Paired t-test
    t_stat, t_pval = stats.ttest_rel(base_err, hyb_err)

    # 2. Wilcoxon Signed-Rank Test
    diff = base_err - hyb_err
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        w_stat, w_pval = stats.wilcoxon(diff)
    else:
        w_stat, w_pval = 0.0, 1.0

    # 3. McNemar's Test for Paired Classifications
    # b: Baseline correct (1), Hybrid incorrect (0)
    # c: Baseline incorrect (0), Hybrid correct (1)
    base_correct = (baseline_preds.ravel() == y_true)
    hyb_correct = (hybrid_preds.ravel() == y_true)

    n00 = int(np.sum((~base_correct) & (~hyb_correct)))
    n01 = int(np.sum((~base_correct) & hyb_correct))  # c
    n10 = int(np.sum(base_correct & (~hyb_correct)))  # b
    n11 = int(np.sum(base_correct & hyb_correct))

    b = n10
    c = n01

    if (b + c) > 0:
        # Edwards' continuity correction
        mcnemar_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        mcnemar_pval = float(stats.chi2.sf(mcnemar_stat, df=1))
    else:
        mcnemar_stat = 0.0
        mcnemar_pval = 1.0

    return {
        "comparison": f"{baseline_name} vs {hybrid_name}",
        "sample_size": len(y_true),
        "paired_t_test": {
            "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
            "p_value": round(float(t_pval), 6) if not np.isnan(t_pval) else 1.0,
            "statistically_significant_p05": bool(t_pval < 0.05) if not np.isnan(t_pval) else False
        },
        "wilcoxon_test": {
            "statistic": round(float(w_stat), 4),
            "p_value": round(float(w_pval), 6),
            "statistically_significant_p05": bool(w_pval < 0.05)
        },
        "mcnemar_test": {
            "contingency_table": {
                "both_correct": n11,
                "both_wrong": n00,
                f"{baseline_name}_only_correct (b)": b,
                f"{hybrid_name}_only_correct (c)": c,
            },
            "chi2_statistic": round(mcnemar_stat, 4),
            "p_value": round(mcnemar_pval, 6),
            "statistically_significant_p05": bool(mcnemar_pval < 0.05)
        }
    }
