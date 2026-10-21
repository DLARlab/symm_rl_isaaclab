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
  "x1|2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1|9999"
  # "go2|2026-10-19_09-33-24_go2_no_trs|9999"
)

# Shared settings. "all" means only gaits in each checkpoint's saved training configuration.
mode="metrics"  # both, video, or metrics
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
