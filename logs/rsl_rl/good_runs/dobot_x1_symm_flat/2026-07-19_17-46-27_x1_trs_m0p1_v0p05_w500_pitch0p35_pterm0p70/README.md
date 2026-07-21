# Dobot X1 sagittal-motion good run

Run: `2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70`

See the shared [60D-to-72D milestone](../../MILESTONE_60D_TO_72D.md) for the
no/low/high TRS leg-usage comparison and restoration procedure.

This is the current reference X1 policy. It was trained for 20,000 iterations with
the 72-dimensional observation, sagittal-plane tracking objective, revised nominal
posture and height target, and TRS regularization enabled.

The comparison baseline is
[`2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral`](../2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral/).

## Configuration and artifacts

| Setting | Value |
| --- | --- |
| Task | `Isaac-Velocity-Flat-Dobot-X1-Symm-v0` |
| Environments | 512 |
| Seed | 42 |
| Iterations | 20,000 |
| Observation size | 72 |
| Initial actor standard deviation | 0.5 |
| Learning rate / schedule | `1e-3` / adaptive |
| Entropy coefficient | 0.005 |
| Pitch reward scale | 0.35 rad |
| Pitch/roll termination limits | 0.70 / 0.70 rad |
| Target base-height range | 0.45-0.60 m |
| TRS mirror / value coefficients | 0.10 / 0.05 |
| TRS warm-up / minimum command speed | 500 iterations / 0.0 m/s |

The archived files are the authority for reproducing this run:

- [Environment configuration](params/env.yaml)
- [Agent configuration](params/agent.yaml)
- [Captured implementation diff](git/symm_rl_isaaclab.diff)
- [Final checkpoint](model_19999.pt)
- [Exported policy](exported/policy.pt)
- [Thirty-second rollout data](plots/play/sim_data.npz)
- [Rollout video](videos/play/rl-video-step-0.mp4)

## Changes from the previous good run

The common controller changes match the Go2 update, with additional X1-specific
posture and safety work:

- Expanded the policy observation from 60D to 72D (a net increase of 12): added six
  measured base velocities, expanded the 3D planar command to a 6D desired twist
  (+3), and appended the 3D sagittal state (lateral displacement plus sine/cosine of
  heading error).
- Replaced the former command-tracking term with the composite
  `straight_line_motion` reward. It jointly scores forward velocity, lateral position,
  heading, lateral velocity, yaw, roll, pitch, height, and stance support.
- Reduced the alive reward from 1.0 to 0.2, added a `-200` termination penalty, and
  added a joint-target-limit penalty. Joint targets are now clamped before they are
  sent to the robot.
- Changed the nominal X1 leg posture. Hip-abduction joints moved from +/-0.1 rad to
  0; the front/rear thigh targets are now +0.6983/-0.6983 rad and the corresponding
  calf targets are -1.2842/+1.2842 rad. This places each foot approximately below its
  thigh motor while preserving the intended natural body height.
- Raised the X1 target base-height range from 0.35-0.55 m to 0.45-0.60 m. Root
  initialization remains 0.50 m and the archived reset event has no position
  randomization (`pose_range: {}`).
- Set pitch and roll termination limits to 0.70 rad, raised the minimum base-height
  termination threshold from 0.15 to 0.25 m, and added a front-body clearance check.
  Pitch shaping uses a 0.35-rad scale.
- Retuned X1 foot clearance to a 0.04 m minimum and 0.025 m scale, and disabled
  unused air-time tracking.
- Trained for 20,000 instead of 10,000 iterations. PPO changed from a fixed `1e-4`
  learning rate to adaptive `1e-3`; initial actor noise changed from 1.0 to 0.5 and
  entropy from 0.01 to 0.005.

TRS was enabled in both runs. Its mirror coefficient (0.10), value coefficient
(0.05), minimum command speed (0.0 m/s), and 500-iteration warm-up did not change.
The gain therefore should not be attributed to stronger TRS regularization.

## Measured improvement

### Matched rollout segment

Both archived rollouts use seed 42 and contain exactly the same desired command
sequence. The table compares the first 10.72 s, ending immediately before the old
policy reset. The new policy completed the full 30 s recording; the old policy reset
at 10.72 s.

| Metric | Previous | This run | Change |
| --- | ---: | ---: | ---: |
| Forward-velocity MAE [m/s] | 0.1797 | 0.0963 | -46.4% |
| Lateral-velocity MAE [m/s] | 0.0247 | 0.0216 | -12.4% |
| Yaw-rate MAE [rad/s] | 0.0771 | 0.0404 | -47.6% |
| X-position MAE [m] | 0.7775 | 0.1420 | -81.7% |
| Y-position MAE [m] | 0.1048 | 0.0622 | -40.7% |
| Mean XY path error [m] | 0.7888 | 0.1562 | -80.2% |
| Final XY path error [m] | 0.6595 | 0.6315 | -4.2% |

The reduction is consistent across forward velocity, lateral motion, yaw, and
accumulated path error. Across each run's complete valid rollout, forward-velocity
MAE fell from 0.1210 to 0.0522 m/s, mean XY path error fell from 2.7012 to 0.5941 m,
and stance-weight/vertical-force correlation increased from 0.569 to 0.593.

### Training diagnostics

Values below are means over the last 100 logged iterations. “This run at 10k” makes
an iteration-matched comparison; “this run final” is near iteration 20,000.

| Logged metric | Previous final (10k) | This run at 10k | This run final (20k) |
| --- | ---: | ---: | ---: |
| XY velocity error | 0.3082 | 0.1499 | — |
| Yaw error | 0.1720 | 0.1719 | 0.1240 |
| Timeout fraction | 0.6417 | 0.9091 | 0.9589 |
| Mean episode length [steps] | 1,150 | 1,415 | 1,460 |
| Base-height termination fraction | 0.00044 | 0.00113 | 0.00201 |
| Orientation termination fraction | 0.05639 | 0.04113 | 0.01764 |
| Calf-contact termination fraction | 0.30143 | 0.04708 | 0.02105 |

At the same 10,000-iteration budget, logged XY velocity error was 51.4% lower,
timeouts were 26.7 percentage points higher, and calf-contact terminations were 84.4%
lower. Continued training reduced yaw error and raised the timeout fraction to 95.9%.

The latest rollout has a forward score of 0.969, straightness score of 0.862,
posture score of 0.892, and support loss of 0.0053. Only 0.13% of samples had their
negative reward clipped. No actual or target joint-limit proximity or violation was
recorded during this rollout. The foot-clearance penalty magnitude in training fell
from 0.0130 to 0.00161, an 87.7% reduction.

## Interpretation and limitations

The data supports the intended mechanism: providing heading/lateral-position error
to the policy and rewarding accumulated straight-line motion improved both local
tracking and long-horizon path retention. The centered nominal leg posture and
joint-target clamping plausibly explain the sharp reduction in calf contact while
keeping X1 away from joint limits.

The higher base target remains unfinished work. The base-height reward penalty grew
from 0.00339 to 0.00916 in magnitude and the height-termination fraction increased
slightly, so this policy has not fully adapted to the stricter 0.45-0.60 m band. The
archived rollout does not log base Z, so it cannot provide a defensible measured mean
height or height range.

This is not a single-variable ablation. Nominal posture, PPO exploration settings,
learning-rate schedule, reward semantics, safety limits, and total training duration
also changed. Reward totals cannot be compared directly across the two runs. The
matched 10.72 s rollout and 10,000-iteration training comparisons reduce those
confounds but do not eliminate them. The logged success-rate metric remains zero
under its current strict threshold and is not useful for ranking these policies.

## Reproduction command

Run from the repository root in PowerShell. Current source should match the archived
configuration and captured diff before treating this as an exact reproduction.

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 512 --max_iterations 20000 --seed 42 --run_name x1_pitch0p35_pterm0p70_trs_m0p1_v0p05_w500 agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1 agent.algorithm.symmetry_cfg.value_loss_coeff=0.05 agent.algorithm.symmetry_cfg.warmup_iterations=500 agent.algorithm.symmetry_cfg.min_abs_command_velocity=0.0
```
