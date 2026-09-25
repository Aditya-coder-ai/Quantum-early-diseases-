"""
End-to-End Execution Pipeline for MIndMatrix:
Hybrid Classical-Quantum Medical Disease Detection System.

Orchestrates:
1. Data loading and validation
2. Leak-free preprocessing & stratified splitting
3. Classical baselines (LogReg, SVM, Random Forest, MLP)
4. Classical feature extractor (Autoencoder -> 16-dim latent space)
5. Latent space validation
6. Quantum & Classical feature selection (16-dim -> 8-dim)
7. Hybrid VQC training & test evaluation
8. Full 5-stage ablation study (A through E)
9. Research figure visualization suite
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import generate_dataset_card
from src.preprocessing.pipeline import run_preprocessing_pipeline
from src.models.classical_baselines import run_all_baselines
from src.models.autoencoder import train_autoencoder_pipeline
from src.models.latent_validation import run_latent_validation
from src.quantum.feature_selection import run_feature_selection_experiment
from src.training.train_vqc import train_vqc
from src.evaluation.ablation import run_ablation_study
from src.evaluation.visualization import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Run MIndMatrix Hybrid Classical-Quantum Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset prep")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip classical baselines")
    parser.add_argument("--skip-encoder", action="store_true", help="Skip autoencoder training")
    parser.add_argument("--skip-fs", action="store_true", help="Skip feature selection")
    parser.add_argument("--skip-vqc", action="store_true", help="Skip VQC training")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("   MINDMATRIX: HYBRID CLASSICAL-QUANTUM PIPELINE EXECUTION")
    print("=" * 70)

    # Stage 1: Data Preparation & Preprocessing
    if not args.skip_data:
        print("\n[STAGE 1/8] Data Preparation & Zero-Leakage Preprocessing...")
        generate_dataset_card()
        run_preprocessing_pipeline()

    # Stage 2: Classical Baselines
    if not args.skip_baselines:
        print("\n[STAGE 2/8] Training Classical Baselines...")
        run_all_baselines()

    # Stage 3: Autoencoder Latent Feature Extractor
    if not args.skip_encoder:
        print("\n[STAGE 3/8] Training Classical Feature Extractor (Autoencoder 30->16)...")
        train_autoencoder_pipeline()

    # Stage 4: Latent Space Validation
    print("\n[STAGE 4/8] Validating 16-Dimensional Latent Representation...")
    run_latent_validation()

    # Stage 5: Quantum Feature Selection
    if not args.skip_fs:
        print("\n[STAGE 5/8] Quantum Feature Selection (16->8)...")
        run_feature_selection_experiment()

    # Stage 6: Hybrid VQC Training
    if not args.skip_vqc:
        print("\n[STAGE 6/8] Training Hybrid Variational Quantum Circuit...")
        train_vqc()

    # Stage 7: Ablation Study
    if not args.skip_ablation:
        print("\n[STAGE 7/8] Running Ablation Analysis (A to E)...")
        run_ablation_study()

    # Stage 8: Visualization
    if not args.skip_plots:
        print("\n[STAGE 8/8] Generating Research Plots & Visualizations...")
        generate_all_plots()

    total_time = round(time.time() - t_start, 2)
    print("\n" + "=" * 70)
    print(f"   PIPELINE EXECUTION COMPLETE IN {total_time}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
