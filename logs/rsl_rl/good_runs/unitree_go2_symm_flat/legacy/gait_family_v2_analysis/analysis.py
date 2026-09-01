# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the shared gait-family analysis engine for the curated Go2 study."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ENGINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "dobot_x1_symm_flat"
    / "legacy"
    / "gait_family_v2_analysis"
    / "analysis.py"
)


def _load_engine():
    """Load the shared analysis engine from the sibling X1 archive."""
    spec = importlib.util.spec_from_file_location("gait_famili_v2_shared_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared analysis engine at {ENGINE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Regenerate the Go2 comparison from its local study manifest."""
    engine = _load_engine()
    engine.main(Path(__file__).with_name("study.json"))


if __name__ == "__main__":
    main()
