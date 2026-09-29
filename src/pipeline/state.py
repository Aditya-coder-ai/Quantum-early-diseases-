"""
Part 7: Pipeline State & Execution Context.
Manages state transitions, structured logging, stage profiling, and error propagation.
"""
from __future__ import annotations

import time
import psutil
import os
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class PipelineState(str, Enum):
    INITIALIZED = "INITIALIZED"
    DATA_LOADED = "DATA_LOADED"
    DATA_VALIDATED = "DATA_VALIDATED"
    PREPROCESSED = "PREPROCESSED"
    FEATURES_EXTRACTED = "FEATURES_EXTRACTED"
    FEATURES_SELECTED = "FEATURES_SELECTED"
    IMBALANCE_HANDLED = "IMBALANCE_HANDLED"
    VQC_TRAINED = "VQC_TRAINED"
    VALIDATED = "VALIDATED"
    TESTED = "TESTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PipelineContext:
    """Execution context and telemetry tracker across all pipeline stages."""
    experiment_id: str
    config_hash: str
    state: PipelineState = PipelineState.INITIALIZED
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_artifacts: Dict[str, str] = field(default_factory=dict)
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)

    def transition_to(self, new_state: PipelineState, message: str = "") -> None:
        """Record state transition with structured timestamped log."""
        self.state = new_state
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": new_state.value,
            "message": message,
            "memory_rss_mb": self.get_memory_usage_mb(),
        }
        self.logs.append(entry)
        if message:
            print(f"[{new_state.value}] {message}")

    def record_stage_time(self, stage_name: str, duration_s: float) -> None:
        """Record execution duration of a pipeline stage."""
        self.stage_timings[stage_name] = round(duration_s, 4)

    def fail(self, stage_name: str, exception: Exception) -> None:
        """Mark pipeline as FAILED and record error trace."""
        self.state = PipelineState.FAILED
        self.error = f"Stage '{stage_name}' failed: {str(exception)}"
        self.logs.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": PipelineState.FAILED.value,
            "error": self.error,
        })
        print(f"\n[PIPELINE FAILED] {self.error}")

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current process RSS memory in MB."""
        try:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def get_total_runtime_s(self) -> float:
        """Compute total runtime elapsed."""
        return round(time.time() - self.start_time, 2)
