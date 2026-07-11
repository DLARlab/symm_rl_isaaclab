# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run symmetric quadruped TRS ablations."""

import sys

from symm_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ablation", *sys.argv[1:]]))
