# Go2 sagittal-motion good run

Run: `2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20`

See the shared [60D-to-72D milestone](../../MILESTONE_60D_TO_72D.md) for the
no/low/high TRS leg-usage comparison and restoration procedure.

This is the current reference Go2 policy. It was trained for 20,000 iterations with
the 72-dimensional observation, sagittal-plane tracking objective, relaxed pitch
termination, and TRS regularization enabled.

The comparison baseline is
[`2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral`](../2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral/).

## Configuration and artifacts

| Setting | Value |
| --- | --- |
| Task | `Isaac-Velocity-Flat-Unitree-Go2-Symm-v0` |
| Environments | 512 |
| Seed | 42 |
| Iterations | 20,000 |
| Observation size | 72 |
| Initial actor standard deviation | 0.5 |
| Learning rate / schedule | `1e-3` / adaptive |
| Entropy coefficient | 0.005 |
| Pitch reward scale | 0.50 rad |
| Pitch termination limit | 1.20 rad |
| Target base-height range | 0.35-0.45 m |
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

The main change was a reformulation from velocity-only locomotion to explicit
sagittal-plane motion control:

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
- Relaxed the Go2 pitch termination limit from 1.0 to 1.2 rad while retaining a
  0.50-rad pitch reward scale. This allows more torso motion without removing posture
  shaping.
- Changed foot clearance from the old phase-only penalty to contact-aware clearance
  tracking with an 0.08 m target, 0.03 m scale, 0.03 m margin, and 0.15 reward weight.
- Disabled unused air-time tracking and expanded diagnostics for posture, support,
  action saturation, target clipping, and joint-limit proximity.
- Trained for 20,000 instead of 10,000 iterations. PPO changed from a fixed `1e-4`
  learning rate to adaptive `1e-3`; initial actor noise changed from 1.0 to 0.5 and
  entropy from 0.01 to 0.005.

TRS was enabled in both runs. Its mirror coefficient (0.10), value coefficient
(0.05), minimum command speed (0.0 m/s), and 500-iteration warm-up did not change.
The gain therefore should not be attributed to stronger TRS regularization.

## Measured improvement

### Matched rollout segment

Both archived rollouts use seed 42 and contain exactly the same desired command
sequence. The table compares the first 10.24 s, ending immediately before the old
policy reset. The new policy completed the full 30 s recording; the old policy reset
at 10.24 s.

| Metric | Previous | This run | Change |
| --- | ---: | ---: | ---: |
| Forward-velocity MAE [m/s] | 0.1311 | 0.0803 | -38.8% |
| Lateral-velocity MAE [m/s] | 0.0188 | 0.0240 | +28.0% |
| Yaw-rate MAE [rad/s] | 0.1110 | 0.1415 | +27.5% |
| X-position MAE [m] | 0.6066 | 0.1189 | -80.4% |
| Y-position MAE [m] | 1.2557 | 0.0535 | -95.7% |
| Mean XY path error [m] | 1.4300 | 0.1315 | -90.8% |
| Final XY path error [m] | 3.5598 | 0.4279 | -88.0% |

The important result is the large reduction in accumulated path drift. Instantaneous
lateral-velocity and yaw-rate errors were slightly worse in this single matched
rollout, but the heading and lateral-position feedback kept those errors from
integrating into a large trajectory error.

Across each run's complete valid rollout, forward-velocity MAE fell from 0.1039 to
0.0742 m/s and mean XY path error fell from 3.6699 to 0.9375 m. The new rollout's
stance-weight/vertical-force correlation also increased slightly, from 0.588 to
0.592.

### Training diagnostics

Values below are means over the last 100 logged iterations. “This run at 10k” makes
an iteration-matched comparison; “this run final” is near iteration 20,000.

| Logged metric | Previous final (10k) | This run at 10k | This run final (20k) |
| --- | ---: | ---: | ---: |
| XY velocity error | 0.3521 | 0.1524 | — |
| Yaw error | 0.2346 | 0.2674 | 0.2205 |
| Timeout fraction | 0.6128 | 0.8190 | 0.8642 |
| Mean episode length [steps] | 1,146 | 1,344 | 1,386 |
| Base-height termination fraction | 0.02634 | 0.00031 | 0.00022 |
| Orientation termination fraction | 0.09722 | 0.02113 | 0.01277 |
| Calf-contact termination fraction | 0.23968 | 0.14551 | 0.11069 |

At the same 10,000-iteration budget, the logged XY velocity error was 56.7% lower,
timeouts were 20.6 percentage points higher, and the main early-termination modes
were substantially lower. Continued training improved yaw tracking and survival.

The latest rollout has a forward score of 0.948, straightness score of 0.755,
posture score of 0.968, and support loss of 0.0011. Only 0.13% of samples had their
negative reward clipped. Actual joints were near a limit for 2.47% of samples and
violated a limit for 0.57%; commanded targets were near a limit for 22.68% of samples
but were successfully clamped, with no target-limit violation.

## Interpretation and limitations

The measurements support the intended mechanism: explicit lateral-position and
heading state, paired with a reward on accumulated straight-line motion, removed the
large drift seen in the baseline. The termination penalty and relaxed pitch limit
also made falling costly without suppressing all torso dynamics.

This is not a single-variable ablation. Network exploration settings, learning-rate
schedule, reward semantics, and total training duration also changed. Reward totals
cannot be compared directly across the two runs. The matched 10.24 s rollout and
10,000-iteration training comparisons reduce those confounds but do not eliminate
them. The remaining Go2 issues are yaw-rate accuracy, frequent near-limit joint
targets, and occasional actual joint-limit violations. The logged success-rate
metric remains zero under its current strict threshold and is not useful for ranking
these policies.

## Reproduction command

Run from the repository root in PowerShell. Current source should match the archived
configuration and captured diff before treating this as an exact reproduction.

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 512 --max_iterations 20000 --seed 42 --run_name go2_h0p35to0p45_pitch0p50_pterm1p20_trs_m0p1_v0p05_w500_minv0_env512_it20000_seed42 agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1 agent.algorithm.symmetry_cfg.value_loss_coeff=0.05 agent.algorithm.symmetry_cfg.warmup_iterations=500 agent.algorithm.symmetry_cfg.min_abs_command_velocity=0.0
```
