#!/usr/bin/env bash

conda activate symm_rl_isaaclab
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# Add one "robot|run_name_or_path" per line (robot: x1 or go2).
runs=(
  "x1|2026-10-06_15-09-50_x1_with_trs_mirror0p1/"
  "x1|2026-10-06_23-51-30_x1_no_trs/"
  "go2|2026-10-07_03-32-43_go2_with_trs_mirror0p1/"
  "go2|2026-10-07_10-24-16_go2_no_trs/"
)
model="9999"
protocol="light"

for run_entry in "${runs[@]}"; do
  robot="${run_entry%%|*}"
  run="${run_entry#*|}"
  if [[ "$run_entry" != *'|'* || -z "$robot" || -z "$run" ]]; then
    printf 'Invalid run entry (expected robot|run_name_or_path): %s\n' "$run_entry" >&2
    exit 1
  fi

  bash scripts/symm_locomotion/evaluation.sh \
    --robot "$robot" \
    --run "$run" \
    --model "$model" \
    --protocol "$protocol" \
    "$@"
done
