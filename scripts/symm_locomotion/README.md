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
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --num-envs 256 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --num-envs 256 --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1
```

Ubuntu/bash:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --iterations 30000 --num-envs 256 --no-trs
bash scripts/symm_locomotion/train.sh --robot x1 --iterations 30000 --num-envs 256 --no-trs
bash scripts/symm_locomotion/play.sh --robot go2 --checkpoint latest
bash scripts/symm_locomotion/play.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/record.sh --robot x1 --checkpoint latest --gif
bash scripts/symm_locomotion/compare.sh --robots go2 x1
bash scripts/symm_locomotion/tensorboard.sh --robots go2 x1
```

Direct Python style still works from an activated environment:

```bash
python scripts/symm_locomotion/train.py --robot go2 --iterations 30000 --no-trs
python scripts/symm_locomotion/play.py --robot x1 --checkpoint latest
```

## Parallel fixed-command recordings

From the repository root, run:

```bash
bash test_diff_cmds.sh --checkpoint logs/rsl_rl/dobot_x1_symm_flat/RUN/model_9999.pt
```

This evaluates trot, bound, both half-bounds, rotary gallop, and transverse
gallop in six successive simulation launches. Each launch creates **600
environments simultaneously**, with 100 assigned to each fixed command:

| Direction | Forward velocity [m/s] | Lateral velocity [m/s] | Yaw rate [rad/s] |
|---|---:|---:|---:|
| forward | 1.0 | 0.0 | 0.0 |
| backward | -1.0 | 0.0 | 0.0 |
| left | 0.0 | 0.5 | 0.0 |
| right | 0.0 | -0.5 | 0.0 |
| yaw_left | 0.0 | 0.0 | 0.6 |
| yaw_right | 0.0 | 0.0 | -0.6 |

Every environment runs for **20 simulated seconds** (1,000 control steps at
50 Hz). Commands remain assigned across command resampling and episode resets.
The fixed gait phase template is the same as in `test.sh`, with phase noise
disabled and observation history length 20. Recording runs headlessly and
saves data without rendering videos or generating per-environment figures.

One `sim_data.npz` per direction contains all 100 environments, producing 36
archives across the six gaits. Default paths are:

```text
<run>/eval/model_9999_diff_cmds_x1_y0.5_yaw0.6/
  trot/forward/sim_data.npz
  trot/backward/sim_data.npz
  ...
  transverse-gallop/yaw_right/sim_data.npz
```

The existing raw and derived signal names are preserved, with an additional
environment axis: `desired_lin_vel` and `true_lin_vel` have shape `(1000, 100, 3)`,
`joint_torques` and `joint_powers` have shape `(1000, 100, 12)`, foot force
vectors have shape `(1000, 100, 4, 3)`, and `episode_done` has shape `(1000, 100)`.
`time_steps` remains a shared `(1000,)` time vector. `env_ids` identifies the
100 environment columns; `command_name` and `assigned_command` identify the
fixed command. **No signals are averaged across environments.**

Terminal samples retain the state before the reset, and `episode_done` marks
each reset independently. Centered moving means are computed per environment
without crossing resets. If interrupted, the archives contain the samples
collected so far and `recording_complete=False`; `requested_steps` and
`recorded_steps` distinguish partial recordings. Twenty seconds is the total
evaluation duration per environment, including any resets during that period.

Existing analysis routines expecting single-environment arrays must select one
environment from these batched signals first. For example,
`data["joint_torques"][:, 0]` selects the first environment in a direction archive.

Use `--dry-run` to inspect all six commands. `--envs_per_command`, `--duration`,
`--forward_speed`, `--lateral_speed`, and `--yaw_rate` override the defaults;
`--output_dir PATH` selects the parent directory for the gait folders. Additional
Hydra settings are forwarded as in `test.sh`. To record just one gait:

```bash
bash scripts/symm_locomotion/_run.sh diff_cmds.py --robot x1 --gait trot \
  --checkpoint logs/rsl_rl/dobot_x1_symm_flat/RUN/model_9999.pt \
  env.policy_observation_history.history_length=20
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

Play and record save videos, GIFs, tracking errors, and rollout diagnostics
together under the selected checkpoint run's `eval/<checkpoint>/` directory by
default, for example `eval/model_9999/`:

```text
sim_data.npz
tracking_errors.txt
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

`analyze_trs_grid.analyze_rollout()` reports front/hind allocation in `metrics`
and left/right allocation in `metrics_left_right`. Left combines FL + RL;
right combines FR + RR, regardless of the movement command. Both mappings
cover squared torque, normalized torque utilization, absolute/positive/negative
mechanical work, force-magnitude impulse, vertical-force impulse, and contact
time. Each includes group totals and signed/absolute imbalance percentages;
`metrics_left_right["contact_time"]` also includes `left_duty_factor` and
`right_duty_factor` as fractions between zero and one.

Left/right signed imbalance is `100 * (left - right) / (left + right)`:
positive means greater left-side contribution, and negative means greater
right-side contribution. A smaller absolute value means more equal allocation,
which is not necessarily better for every gait or maneuver. These are allocation
measurements, not tests of footfall synchronization. Analyze each gait and
constant-command window separately when comparing policies.
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
bash scripts/symm_locomotion/train.sh --nohup --robot go2 --iterations 30000 --no-trs
bash scripts/symm_locomotion/ablation.sh --nohup --robot x1 --iterations 10000 --seeds 1 2 3
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

## Policy evaluation

`evaluation.py`, `evaluation.ps1`, and `evaluation.sh` evaluate a checkpoint at
fixed commands. They report velocity and heading tracking, gait/contact
fidelity, transient response and progress, actuator/foot loading, leg-use
balance, and combined success. The default protocol is
the Cartesian grid of all ten time-reversal-closed v2 evaluation gait rows and
`vx = -1.5, -1.0, -0.5, +0.5, +1.0, +1.5 m/s`. Every cell runs in its own
episode with zero lateral/yaw command and the nominal evaluation profile. Its
first 5 seconds are nominal settling time and the following 10 seconds are the
nominal measurement window. Analysis trims the post-settle slice to complete
common gait cycles when at least one full cycle is available, so its actual
measurement start, stop, duration, and sample count can be shorter and are
recorded per cell.

Archived 56D MLP policies are detected automatically from the checkpoint's
`params/env.yaml` and `params/agent.yaml`. Keep both files beside the checkpoint:

```text
my_policy/
  model_9999.pt
  params/
    env.yaml
    agent.yaml
```

For example, evaluate a local X1 policy with the eight-cell screen:

```bash
conda activate symm_rl_isaaclab
bash scripts/symm_locomotion/evaluation.sh \
  --robot x1 --checkpoint /path/to/my_policy/model_9999.pt --protocol light
```

On Windows, use `evaluation.ps1` with the same arguments. The Python entry point
also accepts these arguments. Use `--protocol full` for the complete grid.

The compatibility profile restores the saved observation order, scaling, joint
order, network architecture, actuator settings, joint limits, and action
convention. A saved history length of 20 supplies `56 * 20 = 1120` inputs,
with complete frames ordered oldest first and zero padding after reset; no
manual history override is needed. The 56D frame contains projected gravity,
velocity commands, joint positions, joint velocities, previous actions, phase
sine, phase cosine, raw foot offsets, and phase ratios, in that order. The
profile is recorded in `study.json` and validated against both network input
sizes before simulation. The evaluator retains xchen's native signed gait-clock convention.
These policies use the selected evaluation protocol's commands and nominal
conditions. The `legacy` protocol flag selects an evaluation grid, not a policy
observation layout.

The publication methods are immutable and distinct: `full` uses
`leg_usage_grid_full_v3`, while `light` uses `leg_usage_grid_light_v2`. The
eight-cell light screen resolves rows through stable gait names and declared
time-reversal partners: trot at `+/-1 m/s`, bound at `+/-1 m/s`,
`half_bound_front_a` at `+1 m/s` with its partner at `-1 m/s`, and `gallop_a`
at `+1 m/s` with its partner at `-1 m/s`. It does not select rows by copied
phase literals. The two protocols use separate output roots and cannot share a
study identity.

New full-v3 artifacts use `evaluations/leg_usage_grid_full_v3/`. Historical
and newly requested legacy-v1 custom grids remain under
`evaluations/leg_usage_grid/`, so either can be analyzed or resumed without
deleting the other. Legacy resume verifies the existing v1 manifest and every
requested grid/runtime control, then uses that manifest as the plan of record;
this avoids rewriting its historical source identity.
Full-v3 manifests created before the directory split are detected by their
stored method and remain available to `--resume` and `--analyze_only` in the
old root; all newly initialized full-v3 studies use the new root.

For each foot, measured contact is a hysteretic state driven only by the
nonnegative world-vertical force filtered to the literal collision path
`/World/ground/terrain/mesh`. Contact state starts false at the beginning of
the analyzed, cycle-trimmed slice. It enters at `Fz >= F_on`, exits at
`Fz <= F_off`, and confirms a change only after:

```text
n_dwell = max(1, ceil(minimum_dwell_s / step_dt - 1e-12))
```

Once confirmed, the new state is backfilled to the first threshold-crossing
sample. Thresholds satisfy `F_on > F_off >= 0`; body-weight mode uses
`g=9.80665 m/s^2` and resolves:

```text
F_on  = max(minimum_on_n,  alpha_on  * mass * g / 4)
F_off = max(minimum_off_n, alpha_off * mass * g / 4)
```

The resolved force thresholds, effective sample dwell, robot mass, and ground
filter paths are recorded and checked against each cell's recording manifest.
`contact.ground_filtered_required` defaults to `true`; setting it to `false`
allows contact analysis from a declared but non-ground-filtered force trace.
With the archived commanded duty trace `beta_k`, swing ratio `s_k=1-beta_k`,
and wrapped foot phase `psi_ki`, desired stance is
`c_ki*=1{psi_ki >= s_k}`. Boundary-excluded scores retain a sample only when
its circular distances from liftoff (`psi=0`) and touchdown (`psi=s_k`) are
both strictly greater than `boundary_exclusion_cycles`; equality is excluded.
Contact metrics are:

```text
A_contact       = 1 - mean(|c_i - c_i*|)
false swing     = sum(1{c_i=1 and c_i*=0}) / sum(1{c_i*=0})
missed stance   = sum(1{c_i=0 and c_i*=1}) / sum(1{c_i*=1})
precision       = TP / (TP + FP)
recall          = TP / (TP + FN)
F1              = 2 * precision * recall / (precision + recall)
beta_i_measured = mean_k(c_ki)
beta_commanded  = mean_k(beta_k)
duty error_i    = beta_i_measured - beta_commanded
abs duty error_i= abs(duty error_i)
```

The separately reported sampled desired-stance fraction is `mean_k(c_ki*)`;
it is not substituted for the archived commanded `beta_commanded`. A ratio
whose denominator is nonpositive is `N/A` (`None`). Thus an empty scored mask
makes all six classification metrics `N/A`; precision or recall with no
positive denominator is `N/A`; and F1 is `N/A` if either input is unavailable
or their sum is zero.

Agreement is reported both with and without event-boundary samples. Event
errors match measured touchdown/liftoff to commanded events in the same
complete cycle and use
`d_S1(a,b)=abs(remainder(a-b+0.5,1)-0.5)`. Mean, median, p95, matched count,
and per-foot event coverage remain explicit. For `M` matches, `U_e` missing
expected events, and `U_a` extra actual events, the combined unmatched fraction
is `(U_e+U_a)/(M+U_e+U_a)`; expected-only and actual-only fractions use
`U_e/(M+U_e)` and `U_a/(M+U_a)`. A zero denominator is `N/A`; with no match,
event error summaries are `N/A` and matched count is zero.

Same-phase pairs are configured when their declared circular offset difference
is at most `phase_sync_tolerance_cycles` (default 0.02, distinct from the 0.04
simultaneous-order tolerance). A pair-cycle is matched only when each foot has
exactly one event in that complete cycle. Aggregate coverage is
`matched_pair_cycles/(configured_pairs*complete_cycles)` and is `N/A` when the
denominator is zero. Touchdown and liftoff disagreement are reported in seconds
and, when period is available, cycles.

Cyclic order uses the first touchdown of each foot in each cycle and scores
only cycles containing all four feet. Events within the simultaneous tolerance
form an equivalence class, including a merge across the circular cycle
boundary. Modal ties are resolved lexicographically. For classification, each
foot's touchdown samples first form a circular mean `phi_bar_i`. Only the six
pairwise phase differences enter a canonical-row score:

```text
score_r = (1/6) sum_{i<j} d_S1(phi_bar_j-phi_bar_i,
                               phi^r_j-phi^r_i)
```

Here `phi^r_i=(-offset^r_i) mod 1` is the row's canonical touchdown phase.

Classification runs only after the configured complete-cycle and per-foot
touchdown/liftoff coverage minima are met. It returns `unclassified` if the
best score exceeds `classifier_max_error_cycles` or the second-best-minus-best
margin is below `classifier_min_margin_cycles`. Row/family predictions,
accuracy, confusion, coverage, complete-cycle count, and failure reasons are
preserved per cell; insufficient domains are never silently averaged.

Velocity and gait fidelity remain separate domains. For measurement samples,
with `e_x=v_x-v_x_cmd` and configured numerical `epsilon`, the velocity fields
use:

```text
vx_RMSE       = sqrt(mean(e_x^2))
vx_MAE        = mean(abs(e_x))
vx_bias       = mean(e_x)
gain          = mean(v_x) / v_x_cmd
relative_RMSE = vx_RMSE / (abs(v_x_cmd) + epsilon)
pair_bias(v)  = abs(mean(v_x | +v) + mean(v_x | -v))
pair_bias_norm(v) = pair_bias(v) / (2*abs(v) + epsilon)
```

All means above use the cycle-trimmed measurement slice. The command-direction
sign-error fraction counts only products `v_x_cmd*v_x<0`; an exact zero is not
a sign error. The 5/10/20-percent bands test
`abs(e_x)<=p(abs(v_x_cmd)+epsilon)`, and gain is `N/A` when
`abs(v_x_cmd)<=epsilon`. Lateral-velocity RMSE/MAE, yaw-rate RMSE/MAE, wrapped
heading-error RMSE/p95, lateral-position RMSE/p95, directed progress, progress
per commanded distance, and termination/loss-of-progress status are retained.
The configured `tracking.yaw_rmse_limit_radps` controls yaw tracking
qualification. `tracking.vx_relative_error_limit` controls the separately
reported `vx_relative_tracking_success` diagnostic. The established planar
qualification remains `tracking_rmse_mps <= 0.05 + 0.25*abs(vx_command)` so
default online report selection is unchanged.

Rise and settling use the complete recorded transient, not the trimmed slice.
Rise time is the first sample with commanded-direction velocity at least the
configured fraction of `abs(mean(full command))`. Settling is the first index
whose entire remaining suffix stays within the configured relative band. If
`T` is the full sample count, `k_m` the measurement start, and `k_s` the
settling index, the post-settle measurement fraction is
`(T-max(k_s,k_m))/(T-k_m)`; settling time and this fraction are `N/A` if no
such suffix exists. Heading metrics are independently `N/A` when authentic
heading state is unavailable, without invalidating velocity metrics.

Cell success thresholds live in `study.json`; they are protocol inputs rather
than scientific constants. Under the default rules, velocity-only success
requires no termination, positive command-direction progress, bounded relative
x-velocity RMSE, and bounded yaw-rate RMSE. Gait-only success requires no
termination, sufficient contact agreement, the correct gait family, and the
configured cycle/event coverage; it does not require positive progress. The
resolved `require_no_termination`, `require_positive_progress`, and
`require_correct_family` flags can disable their respective checks. Joint
success is the conjunction. When the gait domain is invalid, gait-only and
joint success are `N/A`, not failures, and their domain coverage remains
explicit.

Configured actuator limits are resolved from each articulation actuator in the
exact action-joint order. The archive records joint names, source per joint,
source-file hash, resolved vector, and fallback status. Publication normalized
load analysis rejects a missing, non-finite, non-positive, fallback, or solver-
sentinel limit. For leg `i`, its three joints `J_i`, analyzed samples `k`, and
control step `dt`, the integrated load definitions are:

```text
u_tau2_i = dt sum_k sum_{j in J_i} tau_kj^2
u_norm_i = dt sum_k sum_{j in J_i} (tau_kj / limit_j)^2
u_work_i = dt sum_k sum_{j in J_i} |power_kj|
         = dt sum_k sum_{j in J_i} |tau_kj qdot_kj|
u_grf_i  = dt sum_k max(Fz_ki, 0)
```

Vertical impulse integrates nonnegative ground-filtered `Fz` over every sample;
the contact state is required to validate the GRF domain and to compute contact
and impact summaries, but it does not gate the impulse sum. Totals per second
divide by `T*dt`; per-cycle totals are `N/A` for zero/unknown complete cycles;
per-directed-metre totals are `N/A` for nonpositive directed progress.

For any nonnegative per-leg exposure `u`, concentration fields are:

```text
CV(u)          = std(u) / (mean(u) + epsilon)
maximum share  = max_i(u_i) / (sum_i(u_i) + epsilon)
max:min        = max_i(u_i) / (min_i(u_i) + epsilon)
front/hind     = ((FL+FR) - (RL+RR)) / (sum_i(u_i) + epsilon)
left/right     = ((FL+RL) - (FR+RR)) / (sum_i(u_i) + epsilon)
```

Signed and absolute imbalances, worst-leg identity/value, raw and normalized
torque-squared exposure, absolute work, and vertical impulse are reported per
second, complete gait cycle, and positive directed metre. Joint tables include
worst normalized torque-squared and absolute-work identities, p95/p99 torque
utilization, saturation fraction, processed-target soft-limit utilization,
and action-clamp fraction. Foot tables include contact-only mean/p95/p99/max
vertical force, total impulse, and touchdown impact peak/impulse summaries.
An impact window has
`max(1,ceil(impact_window_s/dt))` samples beginning at each measured touchdown;
no pre-impact baseline is subtracted. No contact samples makes force summaries
`N/A`; no touchdown makes impact event count zero and impact summaries `N/A`.

Concentration ratios remain epsilon-regularized even when all values or a
minimum leg value are zero. `zero_leg_count` and a max:min reason field expose
that condition; the deterministic worst-leg tie is FL. The older percent
front/hind field is instead `N/A` when total exposure is nonpositive. These
quantities are load-allocation and concentration proxies only; they do not
estimate fatigue life or failure probability.

```powershell
.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --run 2026-08-21_example --model 19999 `
  --expected_branch 72d-symm-v4-integration
```

Use the exact light and full commands below for a resolved checkpoint:

```powershell
.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --checkpoint C:\path\to\model_19999.pt --protocol light `
  --expected_branch 72d-symm-v4-integration

.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --checkpoint C:\path\to\model_19999.pt --protocol full `
  --expected_branch 72d-symm-v4-integration
```

The utility resolves `--run`, `--model`, and `--checkpoint latest` in the same
way as `play` and `record`. It compiles the selected gait and velocity sequence
into a shared `study.json` plan consumed by the playback process. This keeps
the model loaded while the runner executes one isolated gait/velocity cell at
a time and prints completed-cell counts and an ETA. Available plan controls are:

```text
--protocol full|light|legacy
--velocities -1.5 -1.0 -0.5 0.5 1.0 1.5
--gait_indices 0 1 2 3 4 5 6 7 8 9
--settle_s 5.0
--measure_s 10.0
--evaluation_seed 42
--evaluation_config PATH_TO_JSON
--render_cell_plots
--resume
--analyze_only
```

Custom `--velocities` and `--gait_indices` belong to the deprecated `legacy`
profile. Immutable `full` requires its exact ten-by-six inventory, while
immutable `light` rejects any nondefault list and always resolves its eight
named/partner cells. Timing, evaluation seed, evaluation-config JSON, plot
choice, and ordered runtime overrides become part of each publication
protocol's study identity. Legacy v1 does not accept `--evaluation_config`.
Settling and measurement durations must be exact positive multiples of the
robot control step (`0.02 s` for Go2 and X1). Forwarded runtime options cannot
override the plan, task, checkpoint, video mode, environment count, seed, or RL
library. Detailed per-cell plots are off by default; the compressed raw arrays
needed for analysis are always retained.

All artifacts are stored directly under the resolved training run, without a
checkpoint-named intermediate folder. Full-v3 uses
`<training-run>/evaluations/leg_usage_grid_full_v3/`; light uses
`<training-run>/evaluations/leg_usage_grid_light/`; and legacy-v1 uses
`<training-run>/evaluations/leg_usage_grid/`. Each has this layout:

```text
<evaluation-root>/
  study.json
  cells/gait_00_trot/vx_neg_0p5/seed_0042/
    sim_data.npz
    metadata.json
    status.json
    recording_manifest.json
  metrics/cell_metrics.csv
  metrics/family_metrics.csv
  metrics/stratified_fidelity.csv
  metrics/stratified_fidelity.json
  metrics/overall_metrics.json
  metrics/analysis_provenance.json
  metrics/coverage.csv
  metrics/joint_metrics.csv                    # when rows exist
  metrics/foot_metrics.csv                     # when rows exist
  metrics/gait_classifications.csv             # when rows exist
  metrics/gait_confusion_matrix.csv            # when rows exist
  metrics/same_phase_pair_metrics.csv          # when pairs exist
  metrics/directional_pair_metrics.csv         # when partner cells exist
  metrics/REPORT.md
  metrics/SCREENING_REPORT.md
  figures/trot.svg
  figures/bound.svg
  figures/half_bound.svg
  figures/gallop.svg
  figures/overall.svg
  figures/coverage.svg
  figures/screening_report.svg
```

The manifest records the resolved checkpoint path, iteration and SHA-256,
source/configuration hashes, exact grid, timing, and seed. A folder can contain
only one checkpoint/protocol: a mismatch is rejected instead of overwriting or
mixing data. Use `--resume` to continue an interrupted identical grid, or
`--analyze_only` (`--analyze-only` remains an alias) to regenerate summaries
from its existing cell files. Any
Isaac Lab/Hydra arguments forwarded after `--` are recorded in order as part of
the immutable protocol; repeat them for `--resume` or `--analyze_only`.
Analyze-only verifies the recorded manifest and checkpoint without requiring
the current source hash to equal the recording source. It writes the current
analyzer, metrics-module, and Git identity to `metrics/analysis_provenance.json`.

Scientific domains remain independent. Velocity is valid from finite required
kinematic arrays; heading has its own authentic-state mask. Gait requires the
per-cell recording manifest, the configured force-source policy, valid
thresholds, and cycle/event coverage. Raw torque/work does not depend on contact or effort
limits. GRF load requires an available validated ground-force and contact-
classification path, but not successful gait coverage/classification.
Normalized load additionally requires the exact non-fallback effort-limit
provenance. A
failure in one domain is `N/A` there and does not erase other valid domains.
`coverage.csv`, both reports, and `overall_metrics.json` expose these counts;
`stratified_fidelity.*` repeats domain denominators by gait row, family,
velocity, direction, and checkpoint/policy hash.

Aggregation skips `None` and non-finite values and returns `None` for an empty
set. Metric-specific headline values are emitted only when every planned cell
is valid in that metric's domain; explicitly named `observed_*` fields retain
partial-domain summaries. The overall legacy-compatible `complete` flag uses
the velocity, gait, and normalized-load intersection, while raw-load, GRF,
heading, classification, and the three success domains retain separate
coverage. Reports render unavailable scalars as `N/A`, never as zero.

Front/hind imbalance is `100 * (front - hind) / (front + hind)`. Tables retain
that signed value for diagnosis, while primary summaries average its absolute
value so forward/backward or gait-row signs cannot cancel. Rows and velocities
are equally weighted within each family; the overall value is an equal mean of
the trot, bound, half-bound, and gallop family means. Missing, short, terminated,
invalid, and nonpositive-progress cells remain explicit in coverage outputs.
Per-distance metrics are reported only for positive commanded-direction
progress. Tracking quality is retained, not filtered: each cell must pass both
`tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and
the configured yaw limit (`yaw_tracking_rmse_radps <= 0.05` by default).
Category and overall figures use
tracking-qualified cells; tables retain both views. Normalized
torque uses the 12 action-ordered effort limits recorded from the loaded robot.
Directed progress uses commanded-sign world-x displacement across the same
intervals as effort integration, with integrated body-x velocity retained as a
consistency diagnostic.

Extra Isaac Lab or Hydra overrides can be passed after `--`. Launcher arguments
before that delimiter are parsed strictly, and the delimiter itself is not
forwarded. For compatibility, delimiter-free Hydra `key=value` overrides are
also accepted, while unknown `--options` are rejected by the launcher:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --no-trs -- \
  env.commands.base_velocity.ranges.lin_vel_x='(-1.0, 2.0)'
```
