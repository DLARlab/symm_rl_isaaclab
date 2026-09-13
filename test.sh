#!/usr/bin/env bash

set -euo pipefail

export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

checkpoint="logs/rsl_rl/dobot_x1_symm_flat/2026-09-11_03-08-04_x1_with_trs_mirror0p1/model_9999.pt"
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
gait_thetas=(
  "0.0,0.5,0.5,0.0"
  "0.0,0.0,0.5,0.5"
  "0.13,-0.13,0.5,0.5"
  "-0.13,0.13,0.5,0.5"
  "-0.13,0.13,0.63,0.37"
  "0.13,-0.13,0.63,0.37"
)

checkpoint_filename="${checkpoint##*/}"
checkpoint_name="${checkpoint_filename%.pt}"
evaluation_root="$(dirname "$checkpoint")/eval"

for gait_index in "${!gait_names[@]}"; do
  gait_name="${gait_names[$gait_index]}"
  gait_theta="${gait_thetas[$gait_index]}"
  output_dir="${evaluation_root}/${checkpoint_name}_${gait_name}_x${x_command}_y${y_command}_yaw${yaw_command}"

  bash scripts/symm_locomotion/record.sh \
    --robot x1 \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --gif \
    --tracking-speed "$x_command" \
    --tracking-lateral-speed "$y_command" \
    --tracking-yaw-rate "$yaw_command" \
    "env.commands.base_velocity.init_foot_thetas=[[$gait_theta]]" \
    env.commands.base_velocity.add_noise_theta=False \
    env.policy_observation_history.history_length=20 \
    "$@"
done
