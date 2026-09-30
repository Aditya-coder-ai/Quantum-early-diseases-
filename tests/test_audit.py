"""
Tests for Phase 0 Scientific Audit and Number Scanner.
"""
import os
import json
import pytest
from scripts.audit_numbers import audit_project_numbers, PROJECT_ROOT


def test_audit_numbers_finds_contradictions():
    inventory = audit_project_numbers()
    assert "vqc_accuracy_conflicts" in inventory
    assert "svm_accuracy_conflicts" in inventory
    assert "vqc_param_conflicts" in inventory
    assert "test_count_mismatches" in inventory
    assert "duplicate_part_numbers" in inventory
    assert "batch_limit_conflicts" in inventory

    # Verify batch limit mismatch is detected
    assert inventory["batch_limit_conflicts"]["src_security_validation_limit"] == "1000"
    assert inventory["batch_limit_conflicts"]["app_config_limit"] == "100"

    # Verify duplicate part 10 is detected
    assert len(inventory["duplicate_part_numbers"]["occurrences"]) >= 2

    # Verify param count conflict is documented
    assert inventory["vqc_param_conflicts"]["reported_parameters"] == [16, 32]
