# Symmetric Locomotion Scripts

This folder is the shared launcher surface for the symmetric quadruped tasks.
Use it for both Unitree Go2 and Dobot X1, and add future robots here instead
of creating another robot-specific script tree.

Supported robot keys:

```text
go2    Unitree Go2
x1     Dobot X1
```

Aliases are also accepted: `unitree-go2`, `unitree_go2`, `dobot`,
`dobot-x1`, and `dobot_x1`.

Complete the repository setup in the root [README](../../README.md#requirements)
before using these launchers. In particular, rerun `isaaclab.bat -i core` or
`./isaaclab.sh -i core` after copying or moving the checkout so editable
package paths do not continue pointing at the old workspace.

## Quick Commands

Windows PowerShell:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 20000 --num-envs 512 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 20000 --num-envs 512 --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1
```

Ubuntu/bash:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --iterations 20000 --num-envs 512 --no-trs
bash scripts/symm_locomotion/train.sh --robot x1 --iterations 20000 --num-envs 512 --no-trs
bash scripts/symm_locomotion/play.sh --robot go2 --checkpoint latest
bash scripts/symm_locomotion/play.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/record.sh --robot x1 --checkpoint latest --gif
bash scripts/symm_locomotion/compare.sh --robots go2 x1
bash scripts/symm_locomotion/tensorboard.sh --robots go2 x1
```

Direct Python style still works from an activated environment:

```bash
python scripts/symm_locomotion/train.py --robot go2 --iterations 20000 --no-trs
python scripts/symm_locomotion/play.py --robot x1 --checkpoint latest
```

The generic launcher accepts the command as its first argument:

```bash
bash scripts/symm_locomotion/symm_locomotion.sh train --robot go2 --smoke --dry-run
```

```powershell
.\scripts\symm_locomotion\symm_locomotion.ps1 train --robot x1 --smoke --dry-run
```

## Common Options

Every command accepts:

```text
--robot go2|x1
--conda-env symm_rl_isaaclab
--use-conda-run
--no-conda-run
--dry-run
```

Training options:

```text
--iterations / --max-iterations
--num-envs
--run-name
--seed
--mirror
--tr-value-coef
--tr-warmup-iterations
--tr-min-abs-cmd-vel
--no-trs
--smoke
```

Training and ablation runs default to 20,000 iterations across 512 environments.

`--tr-min-abs-cmd-vel` defaults to `0.0`, so TRS losses also train on
zero-velocity commands used for in-place behavior.

`--no-trs` disables symmetry data augmentation, mirror loss, and TRS value
loss by forwarding these Hydra overrides:

```text
agent.algorithm.symmetry_cfg.use_data_augmentation=False
agent.algorithm.symmetry_cfg.use_mirror_loss=False
agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0
agent.algorithm.symmetry_cfg.value_loss_coeff=0.0
```

`--smoke` uses one environment and one training iteration.

Play and record resolve `--checkpoint latest` from the newest run under the
selected robot experiment directory. You can also use:

```text
--run RUN_FOLDER_OR_PATH
--model 9999
--checkpoint PATH_TO_MODEL_PT
```

Recordings are 30 seconds by default (1,500 environment steps at 50 Hz).
Pass `--video-length` to override the length in environment steps.

Play and record save the following rollout diagnostics under the selected
checkpoint run's `plots/play/` directory by default:

```text
sim_data.npz
figure1_linear_velocities_and_position.png
figure2_E_C_frc_and_contact_forces.png
figure3_E_C_spd_and_foot_velocities.png
figure4_agg_E_C_frc_vs_contact.png
figure5_policy_actions_and_joint_limits.png
figure6_straight_line_reward_diagnostics.png
figure7_foot_clearance.png
figure8_leg_motor_torques.png
figure9_leg_motor_powers.png
figure10_leg_ground_reaction_forces.png
```

These reproduce the IsaacGym rollout plots for measured versus desired base
velocity/position, `E_C_frc` versus foot contact force, and `E_C_spd` versus
foot speed, with additional policy-action, joint-limit, and straight-line reward
diagnostics. The leg-usage figures are ordered front-left, front-right,
rear-left, rear-right. They plot the absolute value of each motor torque, each
motor's absolute mechanical power (`abs(torque * joint velocity)`), and each
absolute world-frame ground-reaction-force component. The black aggregate trace
is the L1 sum for that leg: `sum(abs(component))`, not `abs(sum(component))` or
the Euclidean force norm. Standard Go2/X1 playback filters contact to the flat
ground and adds the tangential friction force to the ground-normal force.

Each reported magnitude has a thin raw curve and a thicker dashed centered
1-second moving arithmetic mean. At the standard 50 Hz control rate this uses
51 samples spanning `t - 0.5 s` through `t + 0.5 s`. Plot edges use the available
partial window with the correct sample count, and smoothing never crosses an
episode reset. This centered, edge-corrected mean avoids the phase delay and
zero-padding bias of a causal or convolution-padded moving average.

`sim_data.npz` retains the signed source arrays for regeneration or signed power
analysis, as well as the derived absolute and smoothed arrays, actions, targets,
positions, soft limits, limit utilization, and measured/target swing-foot
heights. Use `--no-plots` to disable plots, `--plots_dir PATH` to override the
output directory, or `--plot_env_index INDEX` to select another environment.
Plot collection is limited to 30 seconds by default; use
`--plot_duration SECONDS` to change the window.

The main leg-usage arrays in `sim_data.npz` are:

- `joint_torques`, `joint_velocities`, `joint_powers`: `(T, 12)` in the saved `joint_names` order.
- `leg_joint_torques`, `leg_joint_powers`: `(T, 4, 3)` in FL, FR, RL, RR order.
- `leg_joint_torque_magnitudes`, `leg_joint_power_magnitudes`: absolute per-motor values;
  `leg_torque_magnitude_sums`, `leg_power_magnitude_sums`: their per-leg L1 sums.
- `leg_torque_sums`, `leg_power_sums`: signed sums retained for compatibility and analysis.
- `foot_ground_reaction_forces_w`: `(T, 4, 3)` world-frame force vectors; the
  boolean `ground_reaction_force_includes_friction` records whether each sample contains friction.
- `foot_ground_reaction_force_abs_components`, `foot_ground_reaction_force_abs_sums`:
  absolute force components and their L1 sums.
- Every plotted magnitude key also has a `_centered_moving_mean` array. The scalar
  `usage_plot_smoothing_window_s` and `usage_plot_smoothing_window_samples` fields
  record the configured duration and actual odd sample count; `episode_done`
  records the boundaries applied during smoothing.

Relative `--run` values are resolved under the selected robot's routine log
directory, such as `logs/rsl_rl/unitree_go2_symm_flat/`. For curated
`logs/rsl_rl/good_runs/` checkpoints, pass the checkpoint path directly with
`--checkpoint`.

Extra Isaac Lab or Hydra overrides can be passed after `--`:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --no-trs -- \
  env.commands.base_velocity.ranges.lin_vel_x='(-1.0, 2.0)'
```

## Matched TRS Study Analysis

`analyze_matched_trs_study.py` is the maintained entry point for the Phase
Mapping V2 four-run studies. Robot-specific run folders, coefficients, plotting
bounds, and command windows live in a `study.json` manifest beside each report;
the metric, validation, table, and SVG implementation is shared.

```powershell
python .\scripts\symm_locomotion\analyze_matched_trs_study.py `
  .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\phase_mapping_v2_go2_trs_run_analysis\study.json

python .\scripts\symm_locomotion\analyze_matched_trs_study.py `
  .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\phase_mapping_v2_x1_trs_run_analysis\study.json
```

The small `reproduce.py` file in either analysis directory invokes the same
shared engine. Before producing outputs, it verifies matched rollout members,
terminal checkpoints, resolved configurations, and archived training-source
provenance. Initial checkpoints are also compared when all are available; the
latest-only curated archive omits them, and a partially present initial set is
rejected. Generated `summary.json` files record the validation state, analysis
method, engine hash, and manifest hash. Treat this manifest-driven utility as
the source of truth for future matched TRS studies; the older hard-coded grid
and TensorBoard scripts remain available for compatibility. Analysis method
`phase_mapping_v2_matched_trs_v1` is intentionally scoped to four-condition,
20,000-iteration studies; use a new versioned method before changing that
horizon or the early-AUC definition.

## Logs

The script maps each robot to its task and experiment directory:

```text
go2 train: Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
go2 play:  Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
go2 logs:  logs/rsl_rl/unitree_go2_symm_flat/

x1 train:  Isaac-Velocity-Flat-Dobot-X1-Symm-v0
x1 play:   Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0
x1 logs:   logs/rsl_rl/dobot_x1_symm_flat/
```

`compare` prints recent run folders and latest checkpoints across robots.
`tensorboard` starts TensorBoard through `python -m tensorboard.main`. On
Windows, multiple selected robots use the shared `logs/rsl_rl/` root because
TensorBoard's named logdir grammar conflicts with drive letters. On Linux and
macOS, the launcher uses named log directories for the selected robots.

## Background Runs

On Linux, `train.sh` and `ablation.sh` support `--nohup`:

```bash
bash scripts/symm_locomotion/train.sh --nohup --robot go2 --iterations 20000 --no-trs
bash scripts/symm_locomotion/ablation.sh --nohup --robot x1 --iterations 20000 --seeds 1 2 3
```

Logs are written under:

```text
logs/symm_locomotion/
```

## Adding Another Robot

Add a new robot in four places:

1. Register the Isaac Lab tasks under
   `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/<robot>_symm/__init__.py`.
2. Put robot-specific assets, joint names, body names, contact sensors, default
   joint positions, height ranges, and actuator gains in
   `config/<robot>_symm/flat_env_cfg.py`.
3. Add robot morphology constants or a small adapter in
   `mdp/<robot>_symm.py` only when the shared defaults do not match.
4. Add one `RobotSpec` entry and any aliases in `symm_cli.py`.

Keep shared reward math, gait command logic, time-reversal transforms, wrapper
behavior, and PPO defaults in the `symm_quadruped` modules unless the behavior
is genuinely robot-specific.
