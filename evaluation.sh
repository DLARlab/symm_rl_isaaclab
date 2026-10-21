#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export OMNI_KIT_ACCEPT_EULA=YES
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  NVIDIA_LIB_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
  if [[ -d "$NVIDIA_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR:${LD_LIBRARY_PATH:-}"
  fi
fi

# Defaults: latest Go2 checkpoint, all saved training gaits, videos/data and metrics.
# Statistics: 100 parallel trials per gait/command, 250 settling + 1000 measured steps.
# Videos: one environment, retaining the existing video duration.
# Example: bash evaluation.sh --robot x1_go2bend --run RUN --model 1499
# --mode video|metrics|both; --gait all|trot|bound|halfbound|rotarygallop|traversegallop
exec bash "$REPO_ROOT/scripts/symm_locomotion/_run.sh" evaluate_trained.py "$@"
