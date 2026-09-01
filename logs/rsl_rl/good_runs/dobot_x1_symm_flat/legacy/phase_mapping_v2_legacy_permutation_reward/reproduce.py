# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explain how to reproduce the archived Dobot X1 Phase Mapping V2 study."""

from __future__ import annotations

import sys

ARCHIVE_NOTICE = """\
This is an archival notice, not a current-checkout reproduction entry point.

The Phase Mapping V2 analysis engine was retired after this historical result
was archived. To reproduce the original analysis, use a detached checkout of:

  5546e5ca3fa4c9268ed3e2c0c78a2a73b6b0056e

and run the wrapper at its original path:

  logs/rsl_rl/good_runs/dobot_x1_symm_flat/phase_mapping_v2_x1_trs_run_analysis/reproduce.py

That wrapper uses the historical analyzer:

  scripts/symm_locomotion/analyze_matched_trs_study.py

The CSV, JSON, figures, manifest, and report beside this notice remain the
preserved historical outputs.
"""


def main() -> int:
    """Print the archival reproduction instructions and return a failure code."""
    print(ARCHIVE_NOTICE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
