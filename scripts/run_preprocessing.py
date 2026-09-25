"""
MIndMatrix - Part 2 Data Preprocessing Runner
Executes the leak-free data preprocessing, validation, and serialization pipeline.
"""
import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.pipeline import run_preprocessing_pipeline

if __name__ == "__main__":
    print("[RUNNER] Executing Part 2: Preprocessing and Data Pipeline...")
    results = run_preprocessing_pipeline()
    print("[RUNNER] Preprocessing finished successfully!")
