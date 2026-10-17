#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Edit one robot|run_name|model entry per task. Tasks run from top to bottom.
# Robots: go2, x1, x1_go2bend. Run names are relative to each robot's log folder.
tasks=(
  "x1_go2bend|2026-10-16_21-42-35_x1_go2bend_no_trs|1499"
)

# Shared settings. "all" means only gaits in each checkpoint's saved training configuration.
mode="both"  # both, video, or metrics
gait="all"

for task in "${tasks[@]}"; do
  IFS='|' read -r robot run model extra <<< "$task"
  if [[ -z "$robot" || -z "$run" || -z "$model" || -n "$extra" ]]; then
    printf 'Invalid evaluation task (expected robot|run_name|model): %s\n' "$task" >&2
    exit 2
  fi
  printf '[evaluation_batch] robot=%s run=%s model=%s\n' "$robot" "$run" "$model"
  bash "$REPO_ROOT/evaluation.sh" \
    --robot "$robot" \
    --run "$run" \
    --model "$model" \
    --mode "$mode" \
    --gait "$gait"
done
