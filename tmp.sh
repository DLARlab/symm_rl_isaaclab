#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

robot=x1_go2bend
run=2026-10-21_08-45-56_x1_go2bend_no_trs
model=9999

OMNI_KIT_ACCEPT_EULA=YES ./test.sh --robot "$robot" --run "$run" --model "$model"
OMNI_KIT_ACCEPT_EULA=YES ./evaluation.sh --robot "$robot" --run "$run" --model "$model" --mode metrics --gait all