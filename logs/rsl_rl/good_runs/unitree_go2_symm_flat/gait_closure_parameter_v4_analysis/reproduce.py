# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce this gait-closure comparison from its resolved manifest."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, *HERE.parents):
    script_dir = candidate / "scripts" / "symm_locomotion"
    if (script_dir / "comparison.py").is_file():
        sys.path.insert(0, str(script_dir))
        break
else:
    raise FileNotFoundError("Could not locate scripts/symm_locomotion from this analysis folder.")

from comparison import main

if __name__ == "__main__":
    raise SystemExit(main(["--manifest", str(HERE / "study.json"), "--output_dir", str(HERE)]))
