#!/usr/bin/env bash

set -euo pipefail

export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

exec bash scripts/symm_locomotion/record.sh \
  --robot x1 \
  --checkpoint logs/rsl_rl/dobot_x1_symm_flat/2026-08-18_05-18-33_x1_no_trs/model_4999.pt \
  --gif \
  "$@"
