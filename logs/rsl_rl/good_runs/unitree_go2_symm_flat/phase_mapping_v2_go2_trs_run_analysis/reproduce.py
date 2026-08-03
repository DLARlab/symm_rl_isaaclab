# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce the manifest-defined Unitree Go2 Phase Mapping V2 study."""

from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[5] / "scripts" / "symm_locomotion"
sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_matched_trs_study import main  # noqa: E402

if __name__ == "__main__":
    main([str(Path(__file__).with_name("study.json"))])
