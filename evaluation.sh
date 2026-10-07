conda activate symm_rl_isaaclab
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

bash scripts/symm_locomotion/evaluation.sh \
  --robot x1 \
  --run 2026-10-03_03-58-10_x1_with_trs_mirror0p1 \
  --model 9999 \
  --protocol light


bash scripts/symm_locomotion/evaluation.sh \
  --robot x1 \
  --run 2026-10-03_07-39-49_x1_no_trs \
  --model 9999 \
  --protocol light