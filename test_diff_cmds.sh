#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

# Override this checkpoint with --checkpoint PATH when invoking the script.
checkpoint="logs/rsl_rl/dobot_x1_symm_flat/2026-09-16_17-05-18_x1_with_trs_mirror0p1_criticcoef=0/model_9999.pt"
x_command="1"
y_command="0.5"
yaw_command="0.6"

gait_names=(
  "trot"
  "bound"
  "half-bound-left"
  "half-bound-right"
  "rotary-gallop"
  "transverse-gallop"
)

for gait_name in "${gait_names[@]}"; do
  bash scripts/symm_locomotion/_run.sh diff_cmds.py \
    --robot x1 \
    --checkpoint "$checkpoint" \
    --gait "$gait_name" \
    --envs_per_command 100 \
    --duration 20 \
    --forward_speed "$x_command" \
    --lateral_speed "$y_command" \
    --yaw_rate "$yaw_command" \
    env.policy_observation_history.history_length=20 \
    "$@"
done
