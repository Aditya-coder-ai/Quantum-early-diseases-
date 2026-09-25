"""
Quantum Feature Selection module.

Selects 8 features from the 16-dimensional latent representation
using a quantum-inspired optimization approach.

Strategy:
- Define a QUBO-like objective combining predictive value and redundancy
- Use PennyLane's QAOA-inspired variational circuit to find optimal selection
- Compare against classical feature selection (SelectKBest, mutual information)

For the initial implementation, we use a variational approach on a quantum
simulator that explores the feature selection landscape.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    RANDOM_SEED, RESULTS_DIR, DATA_PROCESSED_DIR,
    SELECTED_DIM, QFS_LAMBDA
)

# PennyLane import
import pennylane as qml
from pennylane import numpy as pnp


def load_latent_data() -> dict:
    """Load latent features and labels."""
    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = pd.read_csv(
            os.path.join(RESULTS_DIR, f"latent_features_{split}.csv")).values
        data[f"y_{split}"] = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, f"y_{split}.csv")).values.ravel()
    return data


# ──────────────────────────────────────────────────────────────────────
# Classical Feature Selection Baselines
# ──────────────────────────────────────────────────────────────────────

def classical_feature_selection(
    X_train: np.ndarray, y_train: np.ndarray, k: int = SELECTED_DIM
) -> dict:
    """
    Run classical feature selection methods.
    
    Returns dict with selected indices from each method.
    """
    results = {}
    
    # Method 1: SelectKBest with ANOVA F-value
    skb_f = SelectKBest(f_classif, k=k)
    skb_f.fit(X_train, y_train)
    results["f_classif"] = {
        "indices": skb_f.get_support(indices=True).tolist(),
        "scores": skb_f.scores_.tolist(),
    }
    
    # Method 2: Mutual information
    skb_mi = SelectKBest(mutual_info_classif, k=k)
    skb_mi.fit(X_train, y_train)
    results["mutual_info"] = {
        "indices": skb_mi.get_support(indices=True).tolist(),
        "scores": skb_mi.scores_.tolist(),
    }
    
    print(f"[CFS] F-classif selected: {results['f_classif']['indices']}")
    print(f"[CFS] Mutual info selected: {results['mutual_info']['indices']}")
    
    return results


# ──────────────────────────────────────────────────────────────────────
# Quantum Feature Selection
# ──────────────────────────────────────────────────────────────────────

def compute_feature_importance_matrix(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute:
    1. Individual feature importance scores (predictive value)
    2. Feature correlation matrix (redundancy)
    
    These form the objective function for quantum optimization.
    """
    n_features = X_train.shape[1]
    
    # Individual importance via univariate classification performance
    importance = np.zeros(n_features)
    for i in range(n_features):
        X_single = X_train[:, i:i+1]
        lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
        scores = cross_val_score(lr, X_single, y_train, cv=3, scoring="roc_auc")
        importance[i] = np.mean(scores)
    
    # Pairwise correlation (absolute) as redundancy measure
    correlation = np.abs(np.corrcoef(X_train.T))
    np.fill_diagonal(correlation, 0)  # No self-redundancy
    
    return importance, correlation


def build_qubo_matrix(
    importance: np.ndarray, correlation: np.ndarray,
    k: int = SELECTED_DIM, lam: float = QFS_LAMBDA
) -> np.ndarray:
    """
    Build a QUBO-like matrix for feature selection.
    
    Objective to MAXIMIZE:
        sum_i(importance_i * x_i) - lam * sum_{i<j}(correlation_{ij} * x_i * x_j)
    
    Subject to selecting approximately k features (soft constraint via penalty).
    
    The QUBO matrix Q is defined such that:
        f(x) = x^T Q x
    
    where Q_ii = importance_i - penalty * (2*k - 1) (linear term + constraint)
          Q_ij = -lam * correlation_ij - 2*penalty (for i != j, constraint term)
    """
    n = len(importance)
    penalty = 0.5  # Penalty for deviating from k features
    
    Q = np.zeros((n, n))
    
    # Diagonal: importance (benefit) + cardinality constraint
    for i in range(n):
        Q[i, i] = importance[i] - penalty * (2 * k - 1)
    
    # Off-diagonal: redundancy penalty + cardinality constraint
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = -lam * correlation[i, j] + 2 * penalty
            Q[j, i] = Q[i, j]
    
    return Q


def quantum_feature_selection(
    X_train: np.ndarray, y_train: np.ndarray,
    k: int = SELECTED_DIM, n_iterations: int = 100
) -> dict:
    """
    Quantum-inspired feature selection using a variational quantum circuit.
    
    Uses a parameterized quantum circuit to explore the binary selection
    space {0,1}^16 and find a high-quality subset of k features.
    """
    print(f"\n[QFS] Starting quantum feature selection...")
    print(f"[QFS] Input features: {X_train.shape[1]}, Target: {k}")
    
    n_features = X_train.shape[1]
    assert n_features == 16, f"Expected 16 features, got {n_features}"
    
    t_start = time.time()
    
    # Step 1: Compute importance and correlation
    importance, correlation = compute_feature_importance_matrix(X_train, y_train)
    print(f"[QFS] Feature importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    
    # Step 2: Build QUBO matrix
    Q = build_qubo_matrix(importance, correlation, k)
    
    # Step 3: Variational quantum optimization
    n_qubits = n_features  # One qubit per feature
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(params):
        """
        Variational circuit for QUBO optimization.
        Each qubit represents a feature selection variable.
        Measurement in Z basis: |0> -> not selected, |1> -> selected.
        """
        # Layer 1: Individual rotations (feature biases)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        
        # Layer 2: Entanglement (capture correlations)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Layer 3: Additional rotations
        for i in range(n_qubits):
            qml.RY(params[n_qubits + i], wires=i)
            qml.RZ(params[2 * n_qubits + i], wires=i)
        
        # Measure expectation of each qubit (selection probability)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    def cost_function(params):
        """
        Cost function combining QUBO objective with the variational circuit.
        
        The circuit outputs expectation values in [-1, 1].
        We map them to selection probabilities in [0, 1]:
            p_i = (1 - <Z_i>) / 2
        
        Then compute the expected QUBO objective.
        """
        expectations = circuit(params)
        # Convert Z expectations to selection probabilities
        probs = [(1 - e) / 2 for e in expectations]
        
        # Compute expected QUBO value
        cost = 0.0
        for i in range(n_qubits):
            cost += Q[i, i] * probs[i]
            for j in range(i + 1, n_qubits):
                cost += Q[i, j] * probs[i] * probs[j]
        
        # We want to MAXIMIZE the QUBO, so we minimize -cost
        # Also add a soft constraint to encourage exactly k features
        selected_count = sum(probs)
        cardinality_penalty = 0.5 * (selected_count - k) ** 2
        
        return -cost + cardinality_penalty
    
    # Initialize parameters
    np.random.seed(RANDOM_SEED)
    params = pnp.array(
        np.random.uniform(-np.pi, np.pi, 3 * n_qubits),
        requires_grad=True
    )
    
    # Optimize
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    best_cost = float("inf")
    best_params = params.copy()
    costs = []
    
    for step in range(n_iterations):
        params, cost_val = opt.step_and_cost(cost_function, params)
        costs.append(float(cost_val))
        
        if cost_val < best_cost:
            best_cost = cost_val
            best_params = params.copy()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}/{n_iterations}: cost={cost_val:.6f} "
                  f"(best={best_cost:.6f})")
    
    # Extract final selection
    final_expectations = circuit(best_params)
    selection_probs = [(1 - float(e)) / 2 for e in final_expectations]
    
    # Select top-k features by probability
    sorted_indices = np.argsort(selection_probs)[::-1]
    selected_indices = sorted(sorted_indices[:k].tolist())
    
    elapsed = round(time.time() - t_start, 2)
    
    result = {
        "method": "quantum_variational",
        "selected_indices": selected_indices,
        "selection_probabilities": [round(p, 4) for p in selection_probs],
        "n_selected": len(selected_indices),
        "best_cost": float(best_cost),
        "feature_importance": importance.tolist(),
        "n_iterations": n_iterations,
        "n_qubits": n_qubits,
        "random_seed": RANDOM_SEED,
        "time_seconds": elapsed,
        "cost_history": costs,
    }
    
    print(f"\n[QFS] Quantum selection complete in {elapsed}s")
    print(f"[QFS] Selected features: {selected_indices}")
    print(f"[QFS] Selection probs: {[round(p, 3) for p in selection_probs]}")
    
    return result


def apply_feature_selection(
    X: np.ndarray, indices: list[int]
) -> np.ndarray:
    """Apply feature selection to data by index."""
    X_selected = X[:, indices]
    assert X_selected.shape[1] == len(indices), \
        f"Expected {len(indices)} features, got {X_selected.shape[1]}"
    return X_selected


def run_feature_selection_pipeline():
    """Complete feature selection pipeline with quantum and classical methods."""
    print("\n" + "="*60)
    print("FEATURE SELECTION PIPELINE")
    print("="*60)
    
    data = load_latent_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Classical methods
    print("\n--- Classical Feature Selection ---")
    classical_results = classical_feature_selection(X_train, y_train)
    
    # Quantum method
    print("\n--- Quantum Feature Selection ---")
    quantum_result = quantum_feature_selection(X_train, y_train)
    
    # Save quantum results
    qfs_path = os.path.join(RESULTS_DIR, "selected_features_quantum.json")
    with open(qfs_path, "w") as f:
        json.dump(quantum_result, f, indent=2)
    print(f"[QFS] Results saved to {qfs_path}")
    
    # Save classical results
    cfs_path = os.path.join(RESULTS_DIR, "selected_features_classical.json")
    with open(cfs_path, "w") as f:
        json.dump(classical_results, f, indent=2, default=lambda x: x.tolist() 
                  if hasattr(x, 'tolist') else x)
    
    # Save selected features for all splits
    selected_idx = quantum_result["selected_indices"]
    for split in ["train", "val", "test"]:
        X_full = data[f"X_{split}"]
        X_sel = apply_feature_selection(X_full, selected_idx)
        assert X_sel.shape[1] == SELECTED_DIM, \
            f"Expected {SELECTED_DIM} dims, got {X_sel.shape[1]}"
        
        cols = [f"selected_{i}" for i in range(SELECTED_DIM)]
        pd.DataFrame(X_sel, columns=cols).to_csv(
            os.path.join(RESULTS_DIR, f"selected_features_{split}.csv"),
            index=False
        )
    
    print(f"\n[QFS] Selected features (8-dim) saved for all splits")
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("FEATURE SELECTION COMPARISON")
    print(f"{'='*60}")
    print(f"  F-classif:     {classical_results['f_classif']['indices']}")
    print(f"  Mutual info:   {classical_results['mutual_info']['indices']}")
    print(f"  Quantum:       {quantum_result['selected_indices']}")
    
    overlap_fc = set(quantum_result['selected_indices']) & \
                 set(classical_results['f_classif']['indices'])
    overlap_mi = set(quantum_result['selected_indices']) & \
                 set(classical_results['mutual_info']['indices'])
    print(f"  Overlap Q-vs-F:  {len(overlap_fc)}/{SELECTED_DIM}")
    print(f"  Overlap Q-vs-MI: {len(overlap_mi)}/{SELECTED_DIM}")
    
    return quantum_result, classical_results


# Alias for pipeline runner
run_feature_selection_experiment = run_feature_selection_pipeline


if __name__ == "__main__":
    run_feature_selection_pipeline()
