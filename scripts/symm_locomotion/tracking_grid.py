# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a symmetric quadruped checkpoint on the command tracking grid."""

import sys

from symm_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["tracking-grid", *sys.argv[1:]]))

