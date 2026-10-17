<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# X1 October 14 playback comparison: no TRS versus actor TRS 0.1

Run date: **2026-10-14**. Saved: **2026-10-15**.

**Rotary gallop showed the clearest TRS benefit. Other gaits showed smaller benefits or tradeoffs in tracking, foot execution, mechanical effort, and load distribution.** Abrupt-motion behavior and target-jump metrics were excluded from the conclusions, as requested.

## Compared checkpoints and method

- **No TRS:** `2026-10-14_09-33-57_x1_no_trs/model_9999.pt`.
- **Actor TRS 0.1:** `2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/model_9999.pt`.

The TRS directory now includes `_valuecoef0_actorcoef0.1`; it was previously named `2026-10-14_05-19-36_x1_with_trs_mirror0p1`.

This report saves the previously discussed playback results from each run's `eval/` directory. Each checkpoint has six commanded gait templates and six sequential velocity commands per gait, giving **36 gait/command scenarios per checkpoint**. These are playback-derived metrics.

- Commands: $v_x=\pm1$ m/s, $v_y=\pm0.5$ m/s, and $\omega_z=\pm0.6$ rad/s; the other two components were zero in each segment.
- Each command lasted 5 s. The first second after each change was excluded, then the window was trimmed to complete gait cycles: 3.6 s/six cycles for positive commands and 3.0 s/five cycles for negative commands.
- Per-gait results equally average the six command scenarios. Overall results equally average all 36 scenarios.
- Both training runs used eight gait templates, seed 42, and matching environment settings. Actor TRS was 0 versus 0.1; critic TRS was zero for both.
- Both used cycle duration 0.60 s, stance duty factor 0.56, and base-height target 0.35 m.
- Foot clearance used the squared-sine trajectory, a 0.0275 m resting offset, 0.1075 m peak foot-link height, outer weight 0.1, and 0.025 m absolute-error scale.

**All paired entries below are No TRS → TRS.** These are descriptive comparisons from one training seed per configuration; the sequential command segments are not independent repeated trials.

Metric labels: **↓ lower is better**, **↑ higher is better**.
**↓\*** means lower effort or less operation near actuator limits is preferable when tracking and gait quality are comparable.
**↓†** means lower imbalance gives more even front/hind loading; this is a balance objective, not a universal measure of gait quality.

Within each pair, **bold** marks the preferred value according to the metric label, including its effort/balance qualifications. Ties at the displayed precision are left unbolded.

## Per-gait conclusions

- **Trot — mixed.** Contact agreement and stance-foot motion improved slightly. Velocity tracking, clearance accuracy, and normalized motor-load balance worsened slightly.
- **Bound — better tracking with tradeoffs.** Overall command tracking and swing-force suppression improved, while duty tracking, stance-foot motion, and power consumption worsened.
- **Half-bound left — better gait execution, worse tracking.** Contact agreement, duty tracking, and swing-force suppression improved. Overall command tracking and normalized motor-load balance worsened.
- **Half-bound right — modest foot-level improvement.** Clearance accuracy, contact agreement, and swing-force suppression improved. Overall tracking remained approximately unchanged; power consumption and normalized motor-load imbalance increased.
- **Rotary gallop — strongest overall benefit.** Contact timing, clearance, swing-force suppression, and overall command tracking improved, while mechanical power decreased. Lateral tracking remained weak, and normalized torque and vertical-force balance worsened.
- **Transverse gallop — small benefits with limited overall advantage.** Foot trajectory and periodicity improved slightly, while overall tracking worsened slightly and power consumption increased.

## Tracking and gait timing

| Gait | Active-command relative RMSE [%] ↓ | Contact agreement [%] ↑ | Requested template classified ↑ |
| --- | ---: | ---: | ---: |
| Trot | **11.80** → 13.80 | 98.13 → **98.93** | 6/6 → 6/6 |
| Bound | 16.61 → **13.91** | **96.13** → 95.76 | 5/6 → 5/6 |
| Half-bound left | **11.68** → 16.70 | 97.21 → **98.42** | 5/6 → **6/6** |
| Half-bound right | **15.09** → 15.33 | 97.62 → **98.32** | 5/6 → 5/6 |
| Rotary gallop | 21.87 → **19.12** | 94.10 → **97.15** | 4/6 → **5/6** |
| Transverse gallop | **11.28** → 12.88 | 97.82 → **98.14** | 6/6 → 6/6 |

Active-command relative RMSE is the RMSE of the nonzero commanded component divided by its command magnitude, expressed as a percentage. Each per-gait entry averages all six commands, including yaw. It does not include off-axis errors.

Contact agreement is the fraction of measured contact states matching the recorded desired stance/swing states. The calculation used the actual signed per-foot phases, contact hysteresis of 10 N on / 5 N off, a 20 ms dwell, and exclusion of samples within 0.02 cycles of phase boundaries. Higher agreement is better.

The template classifier uses pairwise circular touchdown-phase differences, a maximum mean error of 0.12 cycles, and a minimum winning margin of 0.01 cycles. A matching label does not establish perfect trajectory or contact tracking.

Averaged over all gaits, active-component RMSE for each command was:

| Command | Active-component RMSE ↓, No TRS → TRS |
| --- | ---: |
| $v_x=+1$ m/s | 0.119 → **0.092** m/s |
| $v_x=-1$ m/s | **0.092** → 0.102 m/s |
| $v_y=+0.5$ m/s | **0.097** → 0.104 m/s |
| $v_y=-0.5$ m/s | **0.082** → 0.107 m/s |
| $\omega_z=+0.6$ rad/s | **0.107** → 0.111 rad/s |
| $\omega_z=-0.6$ rad/s | 0.082 → **0.069** rad/s |

## Foot clearance and periodicity

| Gait | Clearance error [cm] ↓ | Duty error [percentage points] ↓ | Swing force [N] ↓ | Stance foot speed [m/s] ↓ |
| --- | ---: | ---: | ---: | ---: |
| Trot | **0.48** → 0.58 | **2.20** → 2.60 | **0.79** → 1.90 | 0.081 → **0.058** |
| Bound | **0.61** → 0.64 | **1.72** → 3.67 | 6.91 → **5.84** | **0.119** → 0.143 |
| Half-bound left | 0.55 → 0.55 | 4.12 → **2.98** | 14.67 → **10.29** | **0.067** → 0.080 |
| Half-bound right | 0.54 → **0.42** | 4.33 → **4.00** | 13.17 → **10.75** | 0.060 → **0.057** |
| Rotary gallop | 0.77 → **0.60** | 6.38 → **3.91** | 19.75 → **10.61** | 0.104 → **0.076** |
| Transverse gallop | 0.49 → **0.47** | **1.66** → 1.67 | 5.47 → **4.37** | 0.068 → **0.060** |

- **Clearance error:** swing-gate-weighted mean absolute difference between measured foot-link height and the commanded trajectory.
- **Duty error:** mean absolute difference between each foot's measured contact fraction and the 56% commanded stance duty factor.
- **Swing force:** mean full 3D force magnitude during commanded swing, using the reward's maximum over three stored sensor samples.
- **Stance foot speed:** mean full 3D world-frame foot speed during commanded stance. Swing-force and stance-speed means exclude the same 0.02-cycle phase-boundary bands used for contact agreement.

Lower values indicate smaller errors in these four metrics. Swing-force magnitude and the vertical-force impulse used below are different force measurements.

## Mechanical effort

| Gait | Absolute mechanical power [W] ↓* | Joint torque RMS [N·m] ↓* | Near torque-limit samples [%] ↓* |
| --- | ---: | ---: | ---: |
| Trot | 43.0 → **42.9** | **5.81** → 5.98 | 0.02 → **0.00** |
| Bound | **59.1** → 64.2 | **5.96** → 6.08 | **1.00** → 2.23 |
| Half-bound left | 62.0 → **60.3** | **5.72** → 5.73 | **0.67** → 0.85 |
| Half-bound right | **59.5** → 62.2 | 5.82 → **5.67** | **0.60** → 1.27 |
| Rotary gallop | 60.9 → **56.4** | **5.56** → 5.71 | **0.70** → 1.35 |
| Transverse gallop | **53.7** → 57.3 | **5.85** → 5.97 | 0.16 → **0.10** |

Absolute mechanical power is the time mean of $\sum_j|\tau_j\dot q_j|$, summed over all 12 joints. Joint torque RMS is computed across all joint/time samples. Near-limit samples have torque magnitude at least 95% of the saved configured actuator effort limit.

The configured limits were 17 N·m for the hip/abad and front-thigh joints, 20 N·m for rear-thigh joints, and 37 N·m for calf joints. These power results describe mechanical actuation effort; velocity tracking differs between the checkpoints, so lower power alone does not establish higher efficiency.

## Front/hind load distribution

| Gait | Mechanical-work imbalance [%] ↓† | Normalized-torque imbalance [%] ↓† | Vertical-force impulse imbalance [%] ↓† |
| --- | ---: | ---: | ---: |
| Trot | 4.86 → **4.09** | **3.88** → 7.58 | **1.66** → 2.41 |
| Bound | **4.77** → 5.09 | **5.48** → 12.27 | 5.37 → **2.08** |
| Half-bound left | 9.12 → **4.56** | **20.33** → 31.34 | **4.61** → 10.02 |
| Half-bound right | 8.44 → **5.18** | **15.14** → 29.39 | **2.63** → 7.87 |
| Rotary gallop | 9.40 → **7.49** | **23.89** → 32.92 | **4.10** → 7.83 |
| Transverse gallop | **6.60** → 9.50 | 13.85 → **12.77** | **2.98** → 3.72 |

Each imbalance was computed separately for each command scenario before averaging:

$$
I=100\frac{|L_{\mathrm{front}}-L_{\mathrm{hind}}|}{L_{\mathrm{front}}+L_{\mathrm{hind}}}.
$$

- **Mechanical work:** $L=\int\sum_j|\tau_j\dot q_j|\,dt$ for the relevant legs.
- **Normalized torque:** $L=\int\sum_j(\tau_j/\tau_j^{\mathrm{limit}})^2\,dt$.
- **Vertical-force impulse:** $L=\int\sum_\ell\max(F^w_{z,\ell},0)\,dt$.

Lower imbalance means more even loading between front and hind legs. It does not establish better gait fidelity. Mechanical work depends on both torque and joint motion; normalized torque also captures demand while a joint moves slowly. This explains why their trends can differ.

TRS reduced work imbalance in both half-bounds and rotary gallop, while normalized torque and vertical-force imbalance increased. Improved foot timing therefore did not consistently produce more even actuator or support loading.

## Overall observations

| Metric | No TRS | TRS 0.1 |
| --- | ---: | ---: |
| Active-command relative RMSE [%] ↓ | **14.72** | 15.29 |
| Contact agreement [%] ↑ | 96.83 | **97.79** |
| Clearance error [cm] ↓ | 0.575 | **0.544** |
| Absolute mechanical power [W] ↓* | **56.4** | 57.2 |
| Requested template classified ↑ | 31/36 | **33/36** |

Measured joint positions had zero **soft-limit violations (↓)** in the analyzed windows. Every recording had an `episode_done` flag only at its final frame, consistent with the configured 30 s episode limit; the playback archive did not separately record the termination reason.

## Source recordings and configurations

The following links point to the source NPZ recordings used for the numerical comparison:

| Gait | No TRS recording | TRS recording |
| --- | ---: | ---: |
| Trot | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_trot_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_trot_x1_y0.5_yaw0.6/sim_data.npz) |
| Bound | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_bound_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_bound_x1_y0.5_yaw0.6/sim_data.npz) |
| Half-bound left | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_half-bound-left_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_half-bound-left_x1_y0.5_yaw0.6/sim_data.npz) |
| Half-bound right | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_half-bound-right_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_half-bound-right_x1_y0.5_yaw0.6/sim_data.npz) |
| Rotary gallop | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_rotary-gallop_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_rotary-gallop_x1_y0.5_yaw0.6/sim_data.npz) |
| Transverse gallop | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/eval/model_9999_transverse-gallop_x1_y0.5_yaw0.6/sim_data.npz) | [NPZ](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/eval/model_9999_transverse-gallop_x1_y0.5_yaw0.6/sim_data.npz) |

- [No TRS env configuration](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/params/env.yaml)
- [No TRS agent configuration](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_09-33-57_x1_no_trs/params/agent.yaml)
- [TRS env configuration](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/params/env.yaml)
- [TRS agent configuration](logs/rsl_rl/dobot_x1_symm_flat/2026-10-14_05-19-36_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/params/agent.yaml)
