#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

exec bash scripts/symm_locomotion/tracking_grid.sh \
  --robot x1 \
  --checkpoint logs/rsl_rl/dobot_x1_symm_flat/2026-08-26_01-05-50_x1_with_trs_mirror0p1/model_5999.pt \
  --envs_per_command 50 \
  --warmup_time 2.0 \
  --measurement_time 18.0 \
  --output tracking_errors.csv \
  env.policy_observation_history.history_length=30 \
  "$@"

