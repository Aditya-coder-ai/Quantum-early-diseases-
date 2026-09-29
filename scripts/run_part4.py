"""
Master Orchestration Script for Part 4 -- Feature Selection.

Executes end-to-end:
    1. Load Part 3 latent features (16-D)
    2. Build feature registry
    3. Run classical feature selection (MI + RFE) at multiple budgets
    4. Build feature-selection objective (QUBO)
    5. Run QAOA feature selection
    6. Brute-force exact solution (sanity check)
    7. QAOA stability test
    8. Fair downstream comparison (same classifier, same splits)
    9. Save all artifacts
    10. Generate comparison report
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import functools
import warnings

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import (
    PROJECT_ROOT, DATA_PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED,
)
from src.feature_selection.config import (
    SEED, INPUT_FEATURE_DIM, FEATURE_BUDGETS, CLASSICAL_METHODS,
    QAOA_NUM_LAYERS, QAOA_SHOTS, QAOA_MAX_ITERATIONS,
    QAOA_PENALTY_LAMBDA, QAOA_CARDINALITY_PENALTY,
    PART4_RESULTS_DIR, PART4_ARTIFACTS_DIR, ARTIFACT_DIRS,
    QAOA_STABILITY_RUNS, QAOA_STABILITY_SEEDS,
)
from src.feature_selection.registry import (
    build_latent_feature_registry, save_registry, validate_registry,
    get_feature_names, get_selected_names,
)
from src.feature_selection.objective import FeatureSelectionObjective
from src.feature_selection.classical import (
    run_classical_selection, save_classical_results,
)
from src.feature_selection.qaoa import (
    run_qaoa, run_qaoa_stability, qaoa_vs_exact, save_qaoa_results,
)
from src.feature_selection.evaluation import (
    run_comparison_evaluation,
)


def load_latent_data():
    """Load Part 3 latent features and labels."""
    X_train = pd.read_csv(
        os.path.join(RESULTS_DIR, "latent_features_train.csv")
    ).values
    X_val = pd.read_csv(
        os.path.join(RESULTS_DIR, "latent_features_val.csv")
    ).values
    X_test = pd.read_csv(
        os.path.join(RESULTS_DIR, "latent_features_test.csv")
    ).values
    y_train = pd.read_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_train.csv")
    ).squeeze().values
    y_val = pd.read_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_val.csv")
    ).squeeze().values
    y_test = pd.read_csv(
        os.path.join(DATA_PROCESSED_DIR, "y_test.csv")
    ).squeeze().values

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def main():
    t_global_start = time.time()

    print("\n" + "=" * 70)
    print("   PART 4 -- CLASSICAL + QAOA FEATURE SELECTION")
    print("=" * 70)

    # ── STEP 0: Load and verify data ──
    print("\n[STEP 0] Loading Part 3 latent features...")
    data = load_latent_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    D = X_train.shape[1]
    assert D == INPUT_FEATURE_DIM, f"Expected {INPUT_FEATURE_DIM}D, got {D}"
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── STEP 1: Build feature registry ──
    print("\n[STEP 1] Building feature registry...")
    registry = build_latent_feature_registry(D)
    validate_registry(registry, D)
    reg_path = save_registry(registry)
    print(f"  Registry saved: {reg_path}")
    print(f"  Features: {get_feature_names(registry)}")

    # ── STEP 2: Classical feature selection ──
    print("\n[STEP 2] Running classical feature selection...")
    print(f"  Methods: {CLASSICAL_METHODS}")
    print(f"  Budgets: {FEATURE_BUDGETS}")

    classical_results = run_classical_selection(
        X_train, y_train,
        methods=CLASSICAL_METHODS,
        budgets=FEATURE_BUDGETS,
        registry=registry,
        seed=SEED,
    )
    classical_paths = save_classical_results(classical_results)
    print(f"  Classical results saved: {classical_paths}")

    # ── STEP 3: Build objective and QUBO ──
    print("\n[STEP 3] Building feature-selection objective...")

    # Use k=8 as the primary QAOA budget (matches SELECTED_DIM)
    qaoa_k = 8
    objective = FeatureSelectionObjective(
        X_train, y_train,
        target_k=qaoa_k,
        alpha=QAOA_PENALTY_LAMBDA,
        beta=QAOA_CARDINALITY_PENALTY,
        seed=SEED,
    )
    obj_path = objective.save_components()
    print(f"  Objective saved: {obj_path}")
    print(f"  Relevance (normed): {np.round(objective.relevance_normed, 3)}")

    Q = objective.build_qubo()
    qubo_verify = objective.verify_qubo(Q)
    assert qubo_verify["passed"], f"QUBO verification failed: {qubo_verify}"
    print(f"  QUBO shape: {Q.shape}, verified: {qubo_verify['passed']}")

    # ── STEP 4: Brute-force exact solution (sanity check) ──
    print("\n[STEP 4] Computing exact brute-force solution (D=16, 65536 subsets)...")
    t0 = time.time()
    exact_result = objective.brute_force_optimal()
    exact_time = time.time() - t0
    exact_result["computation_time_s"] = round(exact_time, 2)
    print(f"  Exact optimum: {exact_result['selected_indices']}")
    print(f"  Exact objective: {exact_result['best_objective']:.4f}")
    print(f"  N selected: {exact_result['n_selected']}")
    print(f"  Enumerated: {exact_result['n_evaluated']} subsets in {exact_time:.2f}s")

    # Save exact result
    exact_path = os.path.join(PART4_ARTIFACTS_DIR, "exact_brute_force.json")
    with open(exact_path, "w") as f:
        json.dump(exact_result, f, indent=2)

    # ── STEP 5: QAOA feature selection ──
    print("\n[STEP 5] Running QAOA feature selection...")
    qaoa_result = run_qaoa(
        objective,
        target_k=qaoa_k,
        p=QAOA_NUM_LAYERS,
        max_iterations=QAOA_MAX_ITERATIONS,
        shots=QAOA_SHOTS,
        seed=SEED,
    )

    # Attach feature names
    qaoa_result["selected_feature_names"] = get_selected_names(
        registry, qaoa_result["selected_indices"]
    )
    qaoa_result["target_k"] = qaoa_k

    qaoa_path = save_qaoa_results(qaoa_result, tag="primary")
    print(f"  QAOA result saved: {qaoa_path}")

    # ── STEP 6: QAOA vs Exact sanity check ──
    print("\n[STEP 6] QAOA vs Exact comparison...")
    sanity = qaoa_vs_exact(qaoa_result, exact_result)
    print(f"  QAOA objective:  {sanity['qaoa_objective']:.4f}")
    print(f"  Exact objective: {sanity['exact_objective']:.4f}")
    print(f"  Gap:             {sanity['objective_gap']:.4f}")
    print(f"  Approx ratio:    {sanity['approximation_ratio']:.4f}")
    print(f"  Overlap:         {sanity['overlap_count']}/{len(sanity['qaoa_indices'])}")
    print(f"  Jaccard:         {sanity['jaccard_similarity']:.4f}")

    sanity_path = os.path.join(PART4_ARTIFACTS_DIR, "qaoa_vs_exact.json")
    with open(sanity_path, "w") as f:
        json.dump(sanity, f, indent=2)

    # ── STEP 7: QAOA stability test ──
    print("\n[STEP 7] Running QAOA stability test...")
    stability = run_qaoa_stability(
        objective,
        target_k=qaoa_k,
        seeds=QAOA_STABILITY_SEEDS,
        p=QAOA_NUM_LAYERS,
        max_iterations=QAOA_MAX_ITERATIONS,
        shots=QAOA_SHOTS,
    )
    stability_summary = {
        "n_runs": stability["n_runs"],
        "seeds": stability["seeds"],
        "objectives": stability["objectives"],
        "best_objective": stability["best_objective"],
        "mean_objective": stability["mean_objective"],
        "std_objective": stability["std_objective"],
        "worst_objective": stability["worst_objective"],
        "subset_frequency": stability["subset_frequency"],
    }
    stability_path = os.path.join(PART4_ARTIFACTS_DIR, "qaoa_stability.json")
    with open(stability_path, "w") as f:
        json.dump(stability_summary, f, indent=2)
    print(f"  Stability saved: {stability_path}")
    print(f"  Best obj: {stability['best_objective']:.4f}")
    print(f"  Mean obj: {stability['mean_objective']:.4f} +/- {stability['std_objective']:.4f}")
    print(f"  Subset freq: {stability['subset_frequency']}")

    # ── STEP 8: Fair downstream comparison ──
    print("\n[STEP 8] Fair downstream comparison (same classifier)...")

    # Collect QAOA results at different budgets
    # For QAOA we only have the primary k=8 result
    qaoa_for_eval = [qaoa_result]

    comparison_metrics = run_comparison_evaluation(
        X_train, y_train, X_val, y_val,
        classical_results=classical_results,
        qaoa_results=qaoa_for_eval,
        budgets=FEATURE_BUDGETS,
        seed=SEED,
    )

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_metrics)
    comparison_csv = os.path.join(PART4_RESULTS_DIR, "comparison_table.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n  Comparison table saved: {comparison_csv}")

    # ── STEP 9: Save feature subsets for Part 5/6 ──
    print("\n[STEP 9] Saving selected feature subsets for downstream use...")

    # Save QAOA-selected features for all splits
    qaoa_indices = qaoa_result["selected_indices"]
    for split_name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
        X_sel = X_split[:, qaoa_indices]
        cols = [f"selected_{i}" for i in range(len(qaoa_indices))]
        out_path = os.path.join(ARTIFACT_DIRS["qaoa"], f"X_{split_name}_selected.csv")
        pd.DataFrame(X_sel, columns=cols).to_csv(out_path, index=False)
        print(f"  QAOA {split_name}: {X_sel.shape} -> {out_path}")

    # Save best classical (MI k=8) features for all splits
    if "mutual_info" in classical_results and 8 in classical_results["mutual_info"]:
        mi_indices = classical_results["mutual_info"][8]["selected_indices"]
        for split_name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
            X_sel = X_split[:, mi_indices]
            cols = [f"selected_{i}" for i in range(len(mi_indices))]
            out_path = os.path.join(ARTIFACT_DIRS["mutual_info"], f"X_{split_name}_selected.csv")
            pd.DataFrame(X_sel, columns=cols).to_csv(out_path, index=False)

    # ── STEP 10: Summary Report ──
    total_time = time.time() - t_global_start

    summary = {
        "status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "input": {
            "feature_dim": D,
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
            "feature_source": "Part 3 autoencoder latent (16D)",
        },
        "classical_selection": {
            "methods": CLASSICAL_METHODS,
            "budgets": FEATURE_BUDGETS,
        },
        "qaoa": {
            "n_qubits": qaoa_result["n_qubits"],
            "layers": qaoa_result["qaoa_layers"],
            "shots": qaoa_result["shots"],
            "optimizer": qaoa_result["optimizer"],
            "max_iterations": qaoa_result["max_iterations"],
            "target_k": qaoa_k,
            "selected_indices": qaoa_result["selected_indices"],
            "selected_names": qaoa_result["selected_feature_names"],
            "objective_value": qaoa_result["objective_value"],
            "total_time_s": qaoa_result["total_time_s"],
        },
        "sanity_check": sanity,
        "stability": stability_summary,
        "seed": SEED,
    }

    summary_path = os.path.join(PART4_RESULTS_DIR, "part4_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("   PART 4 COMPLETED SUCCESSFULLY!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Summary: {summary_path}")
    print(f"   Comparison: {comparison_csv}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
