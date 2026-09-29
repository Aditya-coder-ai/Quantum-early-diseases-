"""
Part 4 -- QAOA-based Quantum Feature Selection.

Implements the full QAOA pipeline:
    1. QUBO -> Ising Hamiltonian conversion
    2. Cost Hamiltonian construction
    3. Mixer Hamiltonian construction (transverse field)
    4. QAOA circuit with p layers
    5. Classical optimiser loop
    6. Measurement and bitstring decoding
    7. Feature-subset extraction

Uses PennyLane (the project's existing quantum framework).
"""
from __future__ import annotations

import json
import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

import pennylane as qml
from pennylane import numpy as pnp

from src.feature_selection.config import (
    SEED, QAOA_NUM_LAYERS, QAOA_SHOTS, QAOA_MAX_ITERATIONS,
    QAOA_OPTIMIZER, ARTIFACT_DIRS,
)
from src.feature_selection.objective import FeatureSelectionObjective
from src.feature_selection.registry import FeatureEntry, get_selected_names

logger = logging.getLogger(__name__)


# =====================================================================
# QUBO -> Ising Conversion
# =====================================================================

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a QUBO matrix Q to Ising form.

    QUBO variable:  x_i in {0, 1}
    Ising variable:  s_i in {-1, +1}
    Mapping:  x_i = (1 - s_i) / 2    (so x_i=1 when s_i=-1)

    Substituting into f(x) = x^T Q x:
        f(s) = (1/4) sum_{i,j} Q_{ij}(1 - s_i)(1 - s_j)
             = (1/4) sum_{i,j} Q_{ij} [1 - s_i - s_j + s_i*s_j]

    For symmetric Q this gives:
        offset = (1/4) sum_{i,j} Q_{ij}
        h_i    = -(1/4) [Q_{ii} + sum_j Q_{ij}]   (collecting -s_i terms)
               Wait -- let me be precise.

    Expanding term by term for symmetric Q:
        (1/4) sum_{i,j} Q_{ij}                         --> offset
        -(1/4) sum_{i,j} Q_{ij} s_i                    --> -(1/2) sum_i s_i * sum_j Q_{ij}
        -(1/4) sum_{i,j} Q_{ij} s_j                    --> same as above (by symmetry) = -(1/2) sum_j s_j * sum_i Q_{ij}
        +(1/4) sum_{i,j} Q_{ij} s_i s_j                --> couplings + diagonal

    But -s_i and -s_j terms are the same by symmetry of Q, so:
        h_i = -(1/2) sum_j Q_{ij}  +  (1/4) Q_{ii}     (the +Q_{ii}/4 comes from s_i*s_i = 1 in the coupling term)
        J_{ij} = (1/4) Q_{ij}    for i != j
        offset needs correction for the s_i*s_i = 1 diagonal terms

    Let me use the standard textbook formula directly:

    Returns:
        J:  (n, n) coupling matrix (upper triangular)
        h:  (n,) local field vector
        offset:  constant energy offset
    """
    n = Q.shape[0]
    assert Q.shape == (n, n), f"Q must be square, got {Q.shape}"

    # Ensure symmetric
    Qs = (Q + Q.T) / 2.0

    # Standard conversion (see Glover et al., "Quantum Bridge Analytics"):
    # f(x) = x^T Q x = sum_i Q_{ii} x_i + sum_{i<j} 2*Qs_{ij} x_i x_j
    # Using x_i = (1 - s_i)/2:
    #   x_i = (1-s_i)/2,  x_i^2 = x_i (binary), x_i x_j = (1-s_i)(1-s_j)/4
    #
    # Linear terms:   Q_{ii} * (1-s_i)/2
    # Quadratic terms: 2*Qs_{ij} * (1-s_i)(1-s_j)/4 = Qs_{ij}/2 * (1 - s_i - s_j + s_i*s_j)
    #
    # Collecting:
    #   offset = sum_i Q_{ii}/2 + sum_{i<j} Qs_{ij}/2
    #   h_i    = -Q_{ii}/2 - sum_{j!=i} Qs_{ij}/2
    #   J_{ij} = Qs_{ij}/2     for i < j

    # But wait, we need to double-check: sum_{i<j} Qs_{ij}/2 * s_i * s_j
    # and the linear contributions from the quadratic terms.

    # Let me just do it numerically, term by term:
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    # From diagonal terms: Q_{ii} * x_i = Q_{ii} * (1 - s_i)/2
    #   = Q_{ii}/2 - Q_{ii}/2 * s_i
    for i in range(n):
        offset += Qs[i, i] / 2.0
        h[i] -= Qs[i, i] / 2.0

    # From off-diagonal terms (i<j): 2*Qs_{ij} * x_i * x_j
    #   = 2*Qs_{ij} * (1-s_i)/2 * (1-s_j)/2
    #   = Qs_{ij}/2 * (1 - s_i - s_j + s_i*s_j)
    #   = Qs_{ij}/2 - Qs_{ij}/2 * s_i - Qs_{ij}/2 * s_j + Qs_{ij}/2 * s_i*s_j
    for i in range(n):
        for j in range(i + 1, n):
            val = Qs[i, j]  # This is half the full coupling (we stored half on each side)
            # But x^T Q x with symmetric Q gives: Q[i,j]*x_i*x_j + Q[j,i]*x_j*x_i = 2*Qs[i,j]*x_i*x_j
            # So the off-diagonal contribution is 2*val * x_i * x_j
            # = 2*val * (1-si)/2 * (1-sj)/2 = val/2 * (1 - si - sj + si*sj)
            offset += val / 2.0
            h[i] -= val / 2.0
            h[j] -= val / 2.0
            J[i, j] = val / 2.0

    return J, h, float(offset)


def verify_ising_conversion(
    Q: np.ndarray, J: np.ndarray, h: np.ndarray, offset: float,
    n_samples: int = 50, seed: int = SEED
) -> dict:
    """
    Verify Ising conversion by comparing QUBO(x) vs Ising(s) + offset
    for random binary vectors.
    """
    rng = np.random.RandomState(seed)
    n = Q.shape[0]
    max_error = 0.0
    errors = []
    for _ in range(n_samples):
        x = rng.randint(0, 2, n).astype(float)
        s = 1 - 2 * x  # x=0 -> s=+1, x=1 -> s=-1

        qubo_val = float(x @ Q @ x)

        ising_val = offset
        for i in range(n):
            ising_val += h[i] * s[i]
            for j in range(i + 1, n):
                ising_val += J[i, j] * s[i] * s[j]

        error = abs(qubo_val - ising_val)
        errors.append(error)
        max_error = max(max_error, error)

    return {
        "max_abs_error": float(max_error),
        "mean_abs_error": float(np.mean(errors)),
        "passed": max_error < 1e-8,
    }


# =====================================================================
# QAOA Circuit
# =====================================================================

def build_cost_hamiltonian(
    J: np.ndarray, h: np.ndarray
) -> qml.Hamiltonian:
    """
    Build the PennyLane cost Hamiltonian from Ising parameters.

    H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j
    """
    n = J.shape[0]
    coeffs = []
    ops = []

    # Local fields
    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

    # Couplings
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)


def build_mixer_hamiltonian(n_qubits: int) -> qml.Hamiltonian:
    """
    Standard transverse-field mixer: H_M = sum_i X_i
    """
    coeffs = [1.0] * n_qubits
    ops = [qml.PauliX(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, ops)


def create_qaoa_circuit(
    cost_h: qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n_qubits: int,
    p: int = QAOA_NUM_LAYERS,
    shots: int = QAOA_SHOTS,
):
    """
    Create QAOA QNode that returns the expectation of the cost Hamiltonian.

    Parameters:
        cost_h:  Cost Hamiltonian H_C
        mixer_h: Mixer Hamiltonian H_M
        n_qubits: Number of qubits
        p:       Number of QAOA layers
        shots:   Measurement shots (None = exact)

    Returns:
        (qnode, dev)
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qaoa_circuit(gamma, beta):
        """QAOA circuit with p layers."""
        # Initial state: |+>^n
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # p QAOA layers
        for layer in range(p):
            # Cost unitary: exp(-i gamma_l H_C)
            qml.ApproxTimeEvolution(cost_h, gamma[layer], 1)
            # Mixer unitary: exp(-i beta_l H_M)
            qml.ApproxTimeEvolution(mixer_h, beta[layer], 1)

        return qml.expval(cost_h)

    return qaoa_circuit, dev


def create_qaoa_sample_circuit(
    cost_h: qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n_qubits: int,
    p: int = QAOA_NUM_LAYERS,
    shots: int = QAOA_SHOTS,
):
    """
    Create QAOA circuit that returns computational-basis samples.
    Used for bitstring extraction after optimisation.
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def qaoa_sample(gamma, beta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for layer in range(p):
            qml.ApproxTimeEvolution(cost_h, gamma[layer], 1)
            qml.ApproxTimeEvolution(mixer_h, beta[layer], 1)
        return qml.sample()

    return qaoa_sample, dev


def compute_basis_energies(
    J: np.ndarray, h: np.ndarray, n_qubits: int
) -> np.ndarray:
    """
    Precompute Ising energies for all 2^n computational basis states.

    Basis states are ordered: |0...00>, |0...01>, ..., |1...11>.
    Bits: b_i in {0, 1}, spin: s_i = 1 - 2*b_i in {+1, -1}.
    Energy: E = sum_i h_i s_i + sum_{i<j} J_{ij} s_i s_j.
    """
    N = 1 << n_qubits
    shifts = np.arange(n_qubits - 1, -1, -1)
    idx = np.arange(N)[:, None]
    bits = (idx >> shifts) & 1
    s = 1.0 - 2.0 * bits  # (N, n_qubits) in {+1, -1}

    energies = s @ h  # (N,)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if abs(J[i, j]) > 1e-12:
                energies += J[i, j] * (s[:, i] * s[:, j])
    return energies


def create_qaoa_prob_circuit(
    cost_h: qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n_qubits: int,
    p: int = QAOA_NUM_LAYERS,
):
    """
    Create QAOA QNode that returns computational basis probabilities.
    Uses diff_method='backprop' on default.qubit for fast analytic gradient.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def qaoa_prob_circuit(gamma, beta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for layer in range(p):
            qml.ApproxTimeEvolution(cost_h, gamma[layer], 1)
            qml.ApproxTimeEvolution(mixer_h, beta[layer], 1)
        return qml.probs(wires=range(n_qubits))

    return qaoa_prob_circuit, dev


# =====================================================================
# Bitstring Decoding
# =====================================================================

def decode_bitstring(bits: np.ndarray) -> List[int]:
    """
    Decode a measurement bitstring to selected feature indices.
    PennyLane convention: 0 = |0>, 1 = |1>.
    We select features where measurement = 1.
    """
    return [int(i) for i, b in enumerate(bits) if int(b) == 1]


def decode_samples(
    samples: np.ndarray
) -> Tuple[np.ndarray, List[int], Dict[str, int]]:
    """
    Decode a batch of samples (shots x n_qubits).

    Returns:
        best_bitstring:  the most frequently occurring bitstring
        best_indices:    decoded feature indices
        bitstring_counts: dict mapping bitstring -> count
    """
    # Handle 1-D case (single shot)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    # Count bitstrings
    counts: Dict[str, int] = {}
    for row in samples:
        key = "".join(str(int(b)) for b in row)
        counts[key] = counts.get(key, 0) + 1

    # Find most common
    best_key = max(counts, key=counts.get)
    best_bits = np.array([int(c) for c in best_key])
    best_indices = decode_bitstring(best_bits)

    return best_bits, best_indices, counts


# =====================================================================
# QAOA Optimizer Loop
# =====================================================================

def run_qaoa(
    objective: FeatureSelectionObjective,
    target_k: int,
    p: int = QAOA_NUM_LAYERS,
    max_iterations: int = QAOA_MAX_ITERATIONS,
    shots: int = QAOA_SHOTS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Full QAOA feature selection pipeline.

    Steps:
        1. Build QUBO from objective
        2. Convert QUBO to Ising
        3. Build cost and mixer Hamiltonians
        4. Optimise QAOA parameters via analytic backpropagation
        5. Sample from optimised statevector and decode
        6. Validate solution and compute metrics

    Returns:
        Dictionary with all QAOA results and metadata.
    """
    t_start = time.time()
    n_qubits = objective.D

    print(f"\n[QAOA] Starting QAOA feature selection", flush=True)
    print(f"[QAOA] n_qubits={n_qubits}, p={p}, shots={shots}, "
          f"max_iter={max_iterations}, seed={seed}", flush=True)

    # Step 1: Build QUBO
    t0 = time.time()
    Q = objective.build_qubo()
    qubo_time = time.time() - t0

    # Step 2: Verify QUBO
    qubo_verify = objective.verify_qubo(Q)
    assert qubo_verify["passed"], f"QUBO verification failed: {qubo_verify}"
    print(f"[QAOA] QUBO verified (max_error={qubo_verify['max_abs_error']:.2e})", flush=True)

    # Step 3: Convert to Ising
    J, h, offset = qubo_to_ising(Q)
    ising_verify = verify_ising_conversion(Q, J, h, offset, seed=seed)
    assert ising_verify["passed"], f"Ising conversion failed: {ising_verify}"
    print(f"[QAOA] Ising conversion verified (max_error={ising_verify['max_abs_error']:.2e})", flush=True)

    # Step 4: Build Hamiltonians
    cost_h = build_cost_hamiltonian(J, h)
    mixer_h = build_mixer_hamiltonian(n_qubits)

    # Step 5: Precompute basis energies and create fast expectation circuit
    energies_np = compute_basis_energies(J, h, n_qubits)
    energies = pnp.array(energies_np)

    qaoa_circuit, dev = create_qaoa_prob_circuit(
        cost_h, mixer_h, n_qubits, p=p
    )

    def cost_fn(gamma, beta):
        probs = qaoa_circuit(gamma, beta)
        return pnp.dot(probs, energies)

    # Step 6: Optimise parameters
    np.random.seed(seed)
    gamma_init = pnp.array(
        np.random.uniform(0, 2 * np.pi, p), requires_grad=True
    )
    beta_init = pnp.array(
        np.random.uniform(0, np.pi, p), requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=0.1)
    gamma, beta = gamma_init.copy(), beta_init.copy()
    best_cost = float("inf")
    best_gamma, best_beta = gamma.copy(), beta.copy()
    cost_history = []

    t_opt_start = time.time()
    for step in range(max_iterations):
        (gamma, beta), cost_val = opt.step_and_cost(
            cost_fn, gamma, beta
        )
        cost_val_f = float(cost_val)
        cost_history.append(cost_val_f)

        if cost_val_f < best_cost:
            best_cost = cost_val_f
            best_gamma = gamma.copy()
            best_beta = beta.copy()

        if (step + 1) % 10 == 0 or step == 0 or (step + 1) == max_iterations:
            print(f"  Step {step+1:3d}/{max_iterations}: "
                  f"cost={cost_val_f:.6f} (best={best_cost:.6f})", flush=True)

    opt_time = time.time() - t_opt_start
    print(f"[QAOA] Optimisation done in {opt_time:.2f}s", flush=True)

    # Step 7: Sample from optimised probability distribution
    opt_probs = np.array(qaoa_circuit(best_gamma, best_beta))
    opt_probs = np.clip(opt_probs, 0.0, 1.0)
    opt_probs /= opt_probs.sum()

    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(opt_probs), size=shots, p=opt_probs)
    counts: Dict[str, int] = {}
    for idx_val in sampled_indices:
        bs = format(idx_val, f"0{n_qubits}b")
        counts[bs] = counts.get(bs, 0) + 1

    # Most probable bitstring
    best_key = max(counts, key=counts.get)
    best_bits = np.array([int(c) for c in best_key])
    best_indices = decode_bitstring(best_bits)

    # Fallback to argmax if empty
    if len(best_indices) == 0:
        argmax_idx = int(np.argmax(opt_probs))
        best_key = format(argmax_idx, f"0{n_qubits}b")
        best_bits = np.array([int(c) for c in best_key])
        best_indices = decode_bitstring(best_bits)

    # Step 8: Get circuit depth
    circuit_info = {}
    try:
        specs = qml.specs(qaoa_circuit)(best_gamma, best_beta)
        circuit_info["gate_count"] = int(specs.get("num_device_wires", n_qubits))
        circuit_info["depth"] = int(specs.get("depth", 0))
    except Exception:
        circuit_info["gate_count"] = n_qubits + p * (n_qubits + len(cost_h.ops))
        circuit_info["depth"] = 2 * p + 1

    # Step 9: Evaluate solution objective
    x_solution = best_bits.astype(float)
    solution_objective = objective.evaluate(x_solution)
    solution_minimisation = objective.evaluate_minimisation(x_solution)

    total_time = time.time() - t_start

    # Build result
    result = {
        "method": "QAOA",
        "selected_indices": sorted(best_indices),
        "n_selected": len(best_indices),
        "bitstring": "".join(str(int(b)) for b in best_bits),
        "objective_value": float(solution_objective),
        "objective_minimisation": float(solution_minimisation),
        "qaoa_cost": float(best_cost),
        "n_qubits": n_qubits,
        "qaoa_layers": p,
        "shots": shots,
        "optimizer": "Adam",
        "max_iterations": max_iterations,
        "seed": seed,
        "cost_history": cost_history,
        "qubo_construction_time_s": round(qubo_time, 4),
        "optimization_time_s": round(opt_time, 4),
        "total_time_s": round(total_time, 4),
        "qubo_verify": qubo_verify,
        "ising_verify": ising_verify,
        "circuit_info": circuit_info,
        "top_bitstrings": dict(
            sorted(counts.items(), key=lambda x: -x[1])[:10]
        ),
    }

    print(f"\n[QAOA] Selected features: {sorted(best_indices)}", flush=True)
    print(f"[QAOA] Objective value: {solution_objective:.4f}", flush=True)
    print(f"[QAOA] Total time: {total_time:.2f}s", flush=True)

    return result


# =====================================================================
# QAOA Stability Test
# =====================================================================

def run_qaoa_stability(
    objective: FeatureSelectionObjective,
    target_k: int,
    seeds: List[int],
    p: int = QAOA_NUM_LAYERS,
    max_iterations: int = QAOA_MAX_ITERATIONS,
    shots: int = QAOA_SHOTS,
) -> Dict[str, Any]:
    """
    Run QAOA multiple times with different seeds.
    Measures stability of the solution.
    """
    print(f"\n[QAOA-STABILITY] Running {len(seeds)} QAOA trials", flush=True)
    results = []
    objectives = []
    selected_sets = []

    for i, s in enumerate(seeds):
        print(f"\n--- Stability run {i+1}/{len(seeds)} (seed={s}) ---", flush=True)
        r = run_qaoa(objective, target_k, p=p,
                     max_iterations=max_iterations, shots=shots, seed=s)
        results.append(r)
        objectives.append(r["objective_value"])
        selected_sets.append(tuple(sorted(r["selected_indices"])))

    # Frequency of selected subsets
    from collections import Counter
    subset_freq = Counter(selected_sets)

    return {
        "n_runs": len(seeds),
        "seeds": seeds,
        "objectives": objectives,
        "best_objective": float(np.max(objectives)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "worst_objective": float(np.min(objectives)),
        "subset_frequency": {str(k): v for k, v in subset_freq.most_common()},
        "all_results": results,
    }


# =====================================================================
# Sanity Check vs Exact Solution
# =====================================================================

def qaoa_vs_exact(
    qaoa_result: Dict[str, Any],
    exact_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare QAOA solution against brute-force exact optimum."""
    qaoa_obj = qaoa_result["objective_value"]
    exact_obj = exact_result["best_objective"]
    qaoa_indices = set(qaoa_result["selected_indices"])
    exact_indices = set(exact_result["selected_indices"])

    overlap = qaoa_indices & exact_indices
    approximation_ratio = qaoa_obj / exact_obj if exact_obj != 0 else float("nan")

    return {
        "qaoa_objective": qaoa_obj,
        "exact_objective": exact_obj,
        "objective_gap": exact_obj - qaoa_obj,
        "approximation_ratio": approximation_ratio,
        "qaoa_indices": sorted(list(qaoa_indices)),
        "exact_indices": sorted(list(exact_indices)),
        "overlap_count": len(overlap),
        "overlap_indices": sorted(list(overlap)),
        "jaccard_similarity": len(overlap) / len(qaoa_indices | exact_indices)
                              if len(qaoa_indices | exact_indices) > 0 else 0.0,
    }


# =====================================================================
# Save / Load
# =====================================================================

def save_qaoa_results(result: Dict[str, Any], tag: str = "default") -> str:
    """Save QAOA results to artifact directory."""
    out_dir = ARTIFACT_DIRS["qaoa"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"qaoa_selection_{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path
