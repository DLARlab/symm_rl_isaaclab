#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  mkdir -p "$REPO_ROOT/logs/go2_symm"
  LOG_FILE="$REPO_ROOT/logs/go2_symm/ablation_$(date +%Y%m%d_%H%M%S).log"
  nohup python3 "$SCRIPT_DIR/ablation.py" "$@" > "$LOG_FILE" 2>&1 &
  echo "Started Go2 Symm ablation in background."
  echo "PID: $!"
  echo "Log: $LOG_FILE"
  exit 0
fi

exec python3 "$SCRIPT_DIR/ablation.py" "$@"
