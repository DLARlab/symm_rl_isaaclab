#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SCRIPT.py [args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND_SCRIPT="$1"
shift

exec "${PYTHON:-python3}" "$SCRIPT_DIR/$COMMAND_SCRIPT" "$@"
