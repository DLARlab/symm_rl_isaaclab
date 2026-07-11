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
  mkdir -p "$REPO_ROOT/logs/symm_locomotion"
  LOG_FILE="$REPO_ROOT/logs/symm_locomotion/ablation_$(date +%Y%m%d_%H%M%S).log"
  nohup "${PYTHON:-python3}" "$SCRIPT_DIR/ablation.py" "$@" > "$LOG_FILE" 2>&1 &
  echo "Started symmetric locomotion ablation in background."
  echo "PID: $!"
  echo "Log: $LOG_FILE"
  exit 0
fi

exec "$SCRIPT_DIR/_run.sh" ablation.py "$@"
