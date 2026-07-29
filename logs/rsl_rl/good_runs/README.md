# Curated symmetric-locomotion runs

This directory contains the checkpoints selected as development milestones for
the Unitree Go2 and Dobot X1 symmetric-locomotion tasks. Routine training output
belongs under `logs/rsl_rl/<experiment>/`; only intentionally selected runs
belong here.

The historical comparison of the legacy 60D policies and the updated 72D
policies through mirror/value `0.20 / 0.10`, including reward and controller
changes, leg-usage measurements, and safe restoration procedures, is:

- [60D to 72D milestone](MILESTONE_60D_TO_72D.md)

The focused comparison of the newest no-TRS and mirror/value `0.30 / 0.15`
policies is:

- [TRS 0.30/0.15 front/hind leg usage](TRS_0P30_0P15_LEG_USAGE.md)
- [TRS 0.30/0.15 training efficiency](TRS_0P30_0P15_TRAINING_EFFICIENCY.md)

The full `3 coefficient pairs x 3 warm-ups` TRS scan, including the exported
TensorBoard learning curves for each robot, is:

- [Full TRS grid report](trs_grid_analysis/REPORT.md)
- [Go2 TensorBoard efficiency plot](trs_grid_analysis/tensorboard_reward_efficiency_go2.svg)
- [X1 TensorBoard efficiency plot](trs_grid_analysis/tensorboard_reward_efficiency_x1.svg)

## Run generations

| Robot | Legacy 60D runs | Updated 72D runs |
| --- | --- | --- |
| Go2 | `2026-07-07` no TRS; `2026-07-13` more TRS | `2026-07-19` no/low TRS; `2026-07-20` high TRS |
| X1 | `2026-07-11` no TRS; `2026-07-13` more TRS | `2026-07-19` no/low TRS; `2026-07-20` high TRS |

Each run's archived `params/env.yaml` and `params/agent.yaml` are the authority
for its resolved configuration. A captured `git/symm_rl_isaaclab.diff` records
tracked source changes present at training time, but it is not necessarily a
complete source snapshot because untracked files listed in its status section
were not embedded in the patch.

Recorded rollouts store raw data in `plots/play/sim_data.npz`. Figures 8-10
show absolute per-leg motor torque, absolute mechanical power, and absolute
ground-reaction-force components. Raw signed channels remain in the NPZ files.

## Curation rules

- Keep the original timestamped directory name and resolved YAML files.
- Keep checkpoints and exported policies in Git LFS.
- Do not treat rollout samples as independent training replicates.
- Record the exact command window and metric definition for every comparison.
- Prefer a new run directory over modifying an archived checkpoint or export.
