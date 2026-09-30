# Loop Engineering Log

## Phase 0: Reproducibility & Audit Snapshot

### Iteration 1
- **Phase:** Phase 0 — Reproducibility & Audit Snapshot
- **Iteration:** 1
- **Hypothesis:** Documentation and historical result files contain conflicting metric reports, ambiguous VQC parameter counts, duplicate part numbering, and inconsistent batch limits because numbers were reported from different experimental runs without a centralized audit script or locked dependency environment. Pinning dependencies and implementing an automated discrepancy scanner will identify all conflicting figures.
- **Baseline:**
  - Dependencies unpinned in lock format (only loose `requirements.txt`).
  - Python 3.14 used without fallback CI matrix for Python 3.11/3.12.
  - VQC reported with conflicting accuracies (94.19%, 90.7%, 93.0%, 90.0%).
  - SVM reported with multiple accuracies (97.67%, 96.51%, 91.86%, 94.19%, 93.49%).
  - VQC L=2 parameters ambiguously reported as 16 vs 32.
  - Test badge in README claimed 188/188 passed; Part 8 docs claimed 17 tests passed; actual collected tests were 207.
  - Duplicate "Part 10" existed: Part 10 (Privacy & Security) and Part 10 (FastAPI Inference Service).
  - Batch limit conflict: `MAX_BATCH_SIZE = 1000` in `src/security/validation.py` vs `100` in `app/config.py`.
- **Change:**
  1. Generated `requirements.lock` pinning all 21 core scientific dependencies.
  2. Created `.github/workflows/ci.yml` defining automated CI matrix across Python 3.11 and 3.12 to mitigate Python 3.14 C-extension and wheel risks.
  3. Built `scripts/audit_numbers.py` to scan documents, extract conflicting metric claims, and output structured discrepancy inventories.
  4. Added `tests/test_audit.py` to test and prevent regressions in audit scanning.
  5. Generated `results/audit_discrepancies.json` and `docs/audit_inventory.md` containing the complete discrepancy catalog.
- **Verify:**
  - `python -m pytest tests/test_audit.py` -> 1/1 PASSED in 0.06s.
  - `python -m pytest -q -k "not test_18_end_to_end_runner_quick_mode"` -> 206/206 PASSED, 1 deselected in 348.16s.
  - `python -m pytest -q tests/test_part9_comparison.py::test_18_end_to_end_runner_quick_mode` -> 1/1 PASSED in 440.36s.
  - Total test suite: 208/208 tests passed. Zero regressions.
- **Measure:**
  - VQC accuracy contradictions cataloged: 60 unique recorded occurrences across splits/seeds.
  - SVM accuracy contradictions cataloged: 41 unique recorded occurrences across representation spaces.
  - Parameter mismatch documented: Ansatz A (16) vs Ansatz B (32).
  - Duplicate part numbers flagged: 2 instances of Part 10.
  - Batch limit mismatch verified: 1000 vs 100.
  - Test count verified: 208 collected tests vs 188 badge.
- **Decide:** KEEP. Acceptance criteria for Phase 0 completely satisfied. Proceeding to Phase 1.
- **Files Touched:**
  - `requirements.lock`
  - `.github/workflows/ci.yml`
  - `scripts/audit_numbers.py`
  - `tests/test_audit.py`
  - `results/audit_discrepancies.json`
  - `docs/audit_inventory.md`
  - `LOOP_LOG.md`
- **Commit Hash:** Pending commit `loop(phase0/iter1): reproducibility lock and audit snapshot`
