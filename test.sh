#!/usr/bin/env bash

set -euo pipefail

export LD_LIBRARY_PATH="/home/dlar58/anaconda3/envs/symm_rl_isaaclab/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

# Add one "robot|checkpoint_path" per line (robot: x1 or go2).
# Each checkpoint runs all six gaits below.
# --plots also saves figure11_foot_periodicity.png and foot_periodicity.csv
# in each gait's output directory (force and speed in both stance and swing).
checkpoints=(
  # "x1|logs/rsl_rl/dobot_x1_symm_flat/2026-10-10_06-57-37_x1_no_trs/model_9999.pt"
  # "go2|logs/rsl_rl/unitree_go2_symm_flat/2026-10-07_03-32-43_go2_with_trs_mirror0p1/model_9999.pt"
  # "go2|logs/rsl_rl/unitree_go2_symm_flat/2026-10-11_04-06-59_go2_no_trs/model_9999.pt"
)
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

for checkpoint_entry in "${checkpoints[@]}"; do
  robot="${checkpoint_entry%%|*}"
  checkpoint="${checkpoint_entry#*|}"
  if [[ "$checkpoint_entry" != *'|'* || -z "$robot" || -z "$checkpoint" ]]; then
    printf 'Invalid checkpoint entry (expected robot|checkpoint_path): %s\n' "$checkpoint_entry" >&2
    exit 1
  fi
  checkpoint_filename="${checkpoint##*/}"
  checkpoint_name="${checkpoint_filename%.pt}"
  evaluation_root="$(dirname "$checkpoint")/eval"

  for gait_index in "${!gait_names[@]}"; do
    gait_name="${gait_names[$gait_index]}"
    gait_theta="${gait_thetas[$gait_index]}"
    output_dir="${evaluation_root}/${checkpoint_name}_${gait_name}_x${x_command}_y${y_command}_yaw${yaw_command}"

    bash scripts/symm_locomotion/record.sh \
      --robot "$robot" \
      --checkpoint "$checkpoint" \
      --output-dir "$output_dir" \
      --plots \
      --gif \
      --tracking-speed "$x_command" \
      --tracking-lateral-speed "$y_command" \
      --tracking-yaw-rate "$yaw_command" \
      "env.commands.base_velocity.init_foot_thetas=[[$gait_theta]]" \
      env.commands.base_velocity.add_noise_theta=False \
      env.policy_observation_history.history_length=20 \
      "$@"
  done
done
