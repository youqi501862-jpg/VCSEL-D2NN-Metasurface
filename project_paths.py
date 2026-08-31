"""Central pathlib-based paths for the D2NN project.

This module is intentionally independent from legacy training scripts. New
engineering utilities may import it without changing existing experiment code.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    configured = os.environ.get("D2NN_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent


ROOT = _resolve_root()
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
COMSOL_RESULTS_DIR = ROOT / "comsol_results"
PRESENTATION_ASSETS_DIR = ROOT / "assets"
MICRO_DEVICE_QAT_DIR = OUTPUTS_DIR / "micro_device_qat"
MICRO_DEVICE_ROBUSTNESS_DIR = OUTPUTS_DIR / "micro_device_error_robustness"
VCSEL_OUTPUTS_DIR = OUTPUTS_DIR
VCSEL_QAT_DIR = VCSEL_OUTPUTS_DIR / "vcsel_near_v2_qat"
VCSEL_ROBUSTNESS_DIR = VCSEL_OUTPUTS_DIR / "vcsel_near_v2_error_robustness"


def ensure_dirs() -> None:
    """Create directories used by new reports and derived presentation assets."""

    for directory in (
        OUTPUTS_DIR,
        REPORTS_DIR,
        COMSOL_RESULTS_DIR,
        PRESENTATION_ASSETS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
