# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a Go2 Symm Isaac Lab checkpoint."""

from __future__ import annotations

import sys

from go2_symm_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["play", *sys.argv[1:]]))
