# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility launcher for the recent-run listing command."""

import sys
import warnings

from symm_cli import main

if __name__ == "__main__":
    warnings.warn(
        "compare.py is deprecated and will be removed in a future release; use symm_cli.py compare or the "
        "symm_locomotion platform launcher with the compare subcommand.",
        FutureWarning,
        stacklevel=1,
    )
    raise SystemExit(main(["compare", *sys.argv[1:]]))
