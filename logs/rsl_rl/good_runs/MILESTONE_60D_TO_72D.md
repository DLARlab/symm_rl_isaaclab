# Symmetric quadruped 60D-to-72D milestone

Date: 2026-07-21

This document records the transition from the first two curated 60D policies
for each robot to the updated 72D sagittal-motion controller. It also records
the no/low/high time-reversal-symmetry (TRS) comparison and the procedure for
returning to the legacy behavior without destructively rewriting the current
branch.

The short result is that the 72D observation and straight-line reward produced
better path retention and survival in the reference runs, but the available
rollouts do **not** show that increasing TRS makes front/hind leg usage more
even. The strongest motor-heating proxy usually became less balanced as the TRS
coefficients increased.

## Archived runs

The "first two" runs below mean the two chronologically oldest directories
currently curated for each robot.

### Unitree Go2

| Structure | TRS level | Mirror/value | Run | Checkpoint |
| --- | --- | --- | --- | --- |
| 60D | none | `0 / 0` | [`2026-07-07_00-11-14_no_trs`](unitree_go2_symm_flat/2026-07-07_00-11-14_no_trs/) | `model_9999.pt` |
| 60D | more | `0.10 / 0.05` | [`2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral`](unitree_go2_symm_flat/2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral/) | `model_9999.pt` |
| 72D | none | `0 / 0` | [`2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20`](unitree_go2_symm_flat/2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20/) | `model_19999.pt` |
| 72D | low | `0.10 / 0.05` | [`2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20`](unitree_go2_symm_flat/2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20/) | `model_19999.pt` |
| 72D | high | `0.20 / 0.10` | [`2026-07-20_16-23-32_go2_trs_m0p20_v0p10_w500`](unitree_go2_symm_flat/2026-07-20_16-23-32_go2_trs_m0p20_v0p10_w500/) | `model_19999.pt` |

### Dobot X1

| Structure | TRS level | Mirror/value | Run | Checkpoint |
| --- | --- | --- | --- | --- |
| 60D | none | `0 / 0` | [`2026-07-11_02-59-13_no_trs`](dobot_x1_symm_flat/2026-07-11_02-59-13_no_trs/) | `model_9999.pt` |
| 60D | more | `0.10 / 0.05` | [`2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral`](dobot_x1_symm_flat/2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral/) | `model_9999.pt` |
| 72D | none | `0 / 0` | [`2026-07-19_10-33-04_x1_no_trs_pitch0p35`](dobot_x1_symm_flat/2026-07-19_10-33-04_x1_no_trs_pitch0p35/) | `model_19999.pt` |
| 72D | low | `0.10 / 0.05` | [`2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70`](dobot_x1_symm_flat/2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70/) | `model_19999.pt` |
| 72D | high | `0.20 / 0.10` | [`2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500`](dobot_x1_symm_flat/2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500/) | `model_19999.pt` |

## Observation change

The legacy and updated policies have the following exact concatenation order.
The joint and leg order is front-left, front-right, rear-left, rear-right.

| 60D offset | Legacy term | Size | 72D offset | Updated term | Size |
| ---: | --- | ---: | ---: | --- | ---: |
| 0 | projected gravity | 3 | 0 | measured base linear velocity | 3 |
| 3 | generated planar command | 3 | 3 | measured base angular velocity | 3 |
| 6 | relative joint position | 12 | 6 | projected gravity | 3 |
| 18 | relative joint velocity | 12 | 9 | desired base twist | 6 |
| 30 | previous action | 12 | 15 | relative joint position | 12 |
| 42 | foot phase sine | 4 | 27 | relative joint velocity | 12 |
| 46 | foot phase cosine | 4 | 39 | previous action | 12 |
| 50 | foot theta sine | 4 | 51 | foot phase sine | 4 |
| 54 | foot theta cosine | 4 | 55 | foot phase cosine | 4 |
| 58 | phase ratios | 2 | 59 | foot theta sine | 4 |
|  |  |  | 63 | foot theta cosine | 4 |
|  |  |  | 67 | phase ratios | 2 |
|  |  |  | 69 | sagittal-plane state | 3 |

The 72D change is a net increase of 12 dimensions:

- six measured base velocities were added;
- the three-component planar command became a six-component desired twist;
- the three-component sagittal state was appended: lateral displacement scaled
  by 0.5 m and sine/cosine of heading error.

The TRS command index consequently moved from 3 to 9. The 72D transform negates
measured and desired velocities plus the odd gait-phase channels. Projected
gravity, joint positions, action targets, even gait channels, phase ratios, and
the sagittal state are retained according to their defined parity. The action
transform is the identity because the policy output is a joint-position offset.

## Reward and controller change

| Area | Legacy 60D | Updated 72D | Reason |
| --- | --- | --- | --- |
| Alive reward | weight `1.0` | weight `0.20` | Reduce reward available merely for remaining alive. |
| Failure | no explicit terminal reward | termination penalty `-200` | Make falling and invalid contact directly costly. |
| Motion objective | `command_tracking_penalty`, weight `0.40` | `straight_line_motion_reward`, weight `1.0` | Optimize forward tracking together with lateral position, heading, lateral/yaw motion, roll, pitch, height, and stance support. |
| Joint safety | no target-limit term | target-limit penalty `0.05`; targets clamped | Penalize requests near soft joint limits and prevent unsafe target application. |
| Periodicity | weight `0.30`, contact-force scale `0.001` | weight `0.30`, contact-force scale `0.005` | Preserve commanded gait timing while retuning stance-force shaping. |
| Base height | weight `0.30` | weight `0.30` | Preserve the robot-specific target band. |
| Hip action | weight `0.15` | weight `0.15` | Preserve hip-action regularization. |
| Leg symmetry | `morphological_symmetry`, weight `0.30` | `leg_permutation_symmetry`, weight `0.30` | Rename the reward to describe what it actually compares. The former API remains as a deprecated alias and the reward semantics did not change. |
| Smoothness | weight `0.10` | weight `0.10` | Preserve torque-difference smoothness shaping. |
| Go2 clearance | phase-only penalty, weight `0.10` | contact-aware tracking reward, weight `0.15`, 0.08 m target | Penalize dragging and excessive lift while tracking the swing trajectory. |
| X1 clearance | phase penalty, weight `0.10` | retuned ground-relative phase penalty, 0.04 m minimum and 0.025 m scale | Retain the stable X1 mechanism with a geometry-appropriate target. |

The updated controller deliberately uses the six configured gait phase patterns
without terrain or gait curriculum learning.

### Robot-specific changes

Go2 retained its nominal `0.0 / 0.8 / -1.5 rad` leg posture and 0.35-0.45 m
height target. Its pitch termination limit was relaxed from 1.0 to 1.2 rad while
the pitch reward scale remained 0.50 rad, allowing more torso motion without
removing pitch shaping.

X1 changed its nominal leg posture to zero hip abduction, front/rear thigh
angles `+/-0.6983 rad`, and corresponding calf angles `-/+1.2842 rad`. Its root
height remains 0.50 m, while its target height changed from 0.35-0.55 m to
0.45-0.60 m. The updated safety bounds use a 0.25 m minimum base height,
0.70 rad roll/pitch limits, and an additional front-body clearance check.

The first no-TRS runs randomized reset lateral velocity/yaw in `[-0.5, 0.5]`
and lateral pushes in `[-0.25, 0.25]`. The legacy more-TRS and all updated runs
set those unsupported lateral/yaw components to zero for the sagittal task.
Playback uses deterministic nominal joint initialization. These playback
settings improve measurement repeatability and should not be mistaken for
extra TRS training data.

The command generator's `min_xy_command_norm` remains 0.2 m/s. The updated
`min_abs_command_velocity=0.0` setting is the PPO gate for applying TRS losses;
it does not by itself change the command sampler or create standing commands.

The rollout tooling was also expanded to record actions, targets, joint limits,
torque, velocity, power, per-foot normal/friction forces, reward diagnostics,
and episode boundaries. Contact-sensor filter handling was fixed for the
per-foot GRF views. These recording changes do not alter PPO training unless
the diagnostic plotter is explicitly enabled during playback.

## PPO and TRS configurations

| Generation | Variant | PPO | LR/schedule | Iterations | Mirror/value | Warm-up | Minimum command | Augmentation |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| 60D | none | `LegacyTimeReversalPPO` | `1e-3`, adaptive | 10,000 | `0 / 0` | 500 | 0.2 m/s | off |
| 60D | more | `TimeReversalPPO` | `1e-4`, fixed | 10,000 | `0.10 / 0.05` | 500 | 0.0 m/s | off |
| 72D | none | `TimeReversalPPO` | `1e-3`, adaptive | 20,000 | `0 / 0` | 500 | 0.0 m/s | off |
| 72D | low | `TimeReversalPPO` | `1e-3`, adaptive | 20,000 | `0.10 / 0.05` | 500 | 0.0 m/s | off |
| 72D | high | `TimeReversalPPO` | `1e-3`, adaptive | 20,000 | `0.20 / 0.10` | 500 | 0.0 m/s | off |

All listed runs use seed 42, 512 environments, and 24 rollout steps per
environment. The 72D no-TRS archive still resolves
`use_time_reversal_regularization: true`, but mirror loss is disabled and both
coefficients are zero, so TRS is effectively inactive.

The legacy none/more comparison is not a controlled TRS ablation: PPO class,
learning rate, schedule, minimum command threshold, and lateral/yaw reset
perturbations changed together. The 72D comparison holds these fields constant,
but still has only one training seed per coefficient level. The no/low archives
use the deprecated reward field name and the high archives use
`leg_permutation_symmetry`; the alias executes the same reward and is not a
behavioral difference.

## Leg-usage measurement

All ten policies were replayed with the current recorder. The 60D checkpoints
were evaluated with their archived 60D observation order and, for X1, their
archived nominal action offsets. Every run contains `plots/play/sim_data.npz`
and the following leg-use figures:

- `figure8_leg_motor_torques.png`;
- `figure9_leg_motor_powers.png`;
- `figure10_leg_ground_reaction_forces.png`.

The plots show absolute values and an episode-aware centered one-second moving
arithmetic mean. Quantitative comparisons use the raw 50 Hz arrays, not the
smoothed curves.

For a front-pair total `F` and hind-pair total `H`, signed imbalance is

```text
100 * (F - H) / (F + H)
```

Positive values mean the front pair is heavier, negative values mean the hind
pair is heavier, and zero is perfectly even. A magnitude of 10% corresponds to
a 55/45 split.

The reported measures are:

- normalized torque-squared exposure: sum over joints of
  `(torque / effort_limit)^2`, a motor current/copper-heating proxy;
- absolute mechanical work rate: sum over joints of
  `abs(torque * joint_velocity)`;
- GRF load: Euclidean norm of each foot's world-frame ground reaction force,
  including normal and friction components;
- total normalized torque-squared exposure, to detect apparent balance caused
  by increasing the previously underused pair's load.

The primary comparison window is `0.5 <= t < 9.5 s`: 450 samples under the
common steady `-0.5666 m/s` command. One-second non-overlapping block bootstrap
intervals describe temporal variation inside a rollout only. They are not
confidence intervals across independently trained policies.

## Updated 72D TRS results

| Robot | TRS | Heating imbalance | Work imbalance | GRF imbalance | Total heating vs none |
| --- | --- | ---: | ---: | ---: | ---: |
| Go2 | none | +7.1% `[1.6, 12.4]` | -3.8% | +17.5% | reference |
| Go2 | low | +14.4% `[12.4, 16.1]` | -47.1% | +23.7% | +14.1% |
| Go2 | high | +42.8% `[38.3, 46.7]` | +13.3% | +28.1% | +0.3% |
| X1 | none | +17.2% `[14.8, 19.6]` | +16.1% | -2.5% | reference |
| X1 | low | +34.6% `[32.7, 36.4]` | +8.5% | +11.7% | +18.1% |
| X1 | high | +31.3% `[28.3, 34.7]` | +15.5% | +10.7% | +37.4% |

For Go2, normalized heating imbalance worsened monotonically as TRS increased.
For X1, low/high TRS approximately doubled the backward-window imbalance. In
the additional X1 `+1.6819 m/s` interval, absolute heating imbalance also
worsened from 3.0% to 6.7% to 8.0%. Work and GRF had isolated improvements but
no monotonic cross-robot trend.

The updated Go2 no-TRS policy terminated just after the fast-forward command
transition at 10.26 s, while the low/high policies survived. This is one
possible robustness signal, not evidence of balanced wear.

## Legacy 60D results

| Robot | Variant | Heating imbalance | Work imbalance | GRF imbalance | Total heating vs none |
| --- | --- | ---: | ---: | ---: | ---: |
| Go2 | none | +6.7% | -40.5% | +5.2% | reference |
| Go2 | more TRS | +33.0% | -39.1% | +18.8% | +23.8% |
| X1 | none | +47.2% | -2.9% | -14.6% | reference |
| X1 | more TRS | +23.4% | +5.6% | -1.7% | -27.1% |

Legacy X1 partially supports the hypothesis: normalized heating and GRF became
more balanced and total heating fell. Legacy Go2 shows the opposite. Both
legacy more-TRS policies terminated after the fast-forward transition, while
both legacy no-TRS policies survived. The observation therefore does not
replicate across robots and is confounded by the simultaneous PPO/LR changes.

## Interpretation

The present evidence does not prove that TRS equalizes front/hind lifetime. The
updated data more often contradicts that hypothesis.

This is consistent with the implemented mechanism. Time reversal reverses
velocities and odd phase quantities; it does not permute front and hind legs.
The PPO losses enforce policy and value consistency between a state and its
time reverse, not equality of front/hind torque, power, or GRF. The separate leg
permutation reward compares phase-aligned joint positions, but it also does not
directly constrain normalized motor or impact load.

For a causal result, train at least eight paired seeds at each no/low/high
coefficient level with every other parameter frozen. Evaluate each checkpoint
on the same symmetric command manifest (`-2`, `-1`, `-0.5`, `0`, `+0.5`, `+1`,
`+2 m/s`) and all six gait patterns. Treat training seed, not rollout sample,
as the statistical unit. Report pair imbalance, total exposure, maximum pair
exposure, p99 torque utilization, GRF impulse, tracking error, distance, and
failure rate.

## Returning to the 60D milestone

### Preferred non-destructive rollback

The parent of this milestone work is the complete pre-update repository commit
`b61d36a9b8f38bf464d8a767732f7041c5fcacaf`. Create an isolated worktree instead
of resetting the active checkout:

```powershell
git worktree add ..\symm_rl_isaaclab_60d b61d36a9b8f38bf464d8a767732f7041c5fcacaf
```

Use the absolute checkpoint path from this curated directory when playing an
old policy in that worktree. The archived `params/env.yaml` and
`params/agent.yaml` are the authority for restoring configuration values. They
are resolved snapshots, not files that should be copied over the current
Python configuration modules.

To undo this entire milestone later while retaining Git history, create a new
feature branch and revert the milestone commit:

```powershell
git switch -c jding/revert-symm-72d
git revert <this-milestone-commit>
```

Do not use `git reset --hard`; it can destroy unrelated local work and does not
leave an auditable restoration commit.

### Per-run historical source references

The archives record these training-time base commits:

| Run | Base commit |
| --- | --- |
| Go2 60D none | `0954ba7e2445f5b0902e3f4b948ebcd5d48b8ee3` |
| X1 60D none | `f9fa92f25a43e317ec26fec0ed73baf73faa6a65` |
| Go2/X1 60D more TRS | `6932e59290ec55e9f62f3bd5da87736de8916a32` |
| Updated 72D runs | `b61d36a9b8f38bf464d8a767732f7041c5fcacaf` plus the captured working diff |

For historical investigation, create a separate worktree at the relevant hash.
Each run's `git/symm_rl_isaaclab.diff` contains metadata followed by a
`--- git diff ---` marker. Only the text after that marker is a patch. Check it
before applying it:

```powershell
$archive = 'logs\rsl_rl\good_runs\<robot>\<run>\git\symm_rl_isaaclab.diff'
$lines = Get-Content -LiteralPath $archive
$marker = [Array]::IndexOf($lines, '--- git diff ---')
$lines[($marker + 1)..($lines.Length - 1)] | Set-Content -LiteralPath '.\historical-run.patch'
git -C ..\historical-worktree apply --check "$PWD\historical-run.patch"
git -C ..\historical-worktree apply "$PWD\historical-run.patch"
```

The captured patch contains tracked diffs only. Training-time status sections
show that some source files were untracked, so the patch alone cannot promise a
byte-exact reconstruction. Keep the original checkpoint and resolved YAML
snapshots as the definitive run artifacts. The pre-milestone worktree is the
most complete practical 60D replay environment available in this repository.

### Configuration-level rollback checklist

If manually recreating legacy training, restore all of these together:

1. the 60D observation order and the corresponding 60D TRS index/parity map;
2. `command_tracking_penalty` at 0.40 and alive reward at 1.0;
3. removal of the straight-line reward, terminal penalty, and target-limit term;
4. legacy foot-clearance functions and robot-specific height/termination bands;
5. X1's archived nominal joint offsets;
6. the archived PPO class, LR/schedule, entropy, actor standard deviation, TRS
   coefficients, warm-up, and command threshold;
7. 10,000 iterations, 512 environments, 24 rollout steps, and seed 42.

Changing only the observation dimension is not a valid rollback: the old
checkpoint also depends on its exact feature order, action offsets, reward
semantics, and robot configuration.
