<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# TRS velocity-sweep comparison

**Date:** 2026-10-22 · **Robots:** X1, X1_go2bend, Go2 · **Checkpoint:** `model_9999.pt` for all six policies.

## Main findings

**Time-reversal symmetry (TRS) improves moderate/high-speed tracking and high-speed rollout completion in these policies. Contact agreement improves at every tested speed for all three robots.** The tradeoffs depend on the robot, speed, and requested gait: gait fidelity, heading control, and actuator effort do not improve consistently.

The tables below highlight three outcomes for each robot. Every comparison reads **No TRS → TRS**; bold red marks the preferred displayed value. A command pair such as **±1.5 m/s** includes both forward and backward commands.

### X1

| Outcome | No TRS → TRS |
| --- | ---: |
| [Speed RMSE](#metric-velocity-rmse) at ±1.5 m/s [m/s] ↓ | 0.108 → <span style="color: #c00000;">**0.080**</span> |
| [Completion](#metric-completion) at ±2 m/s [%] ↑ | 94.31 → <span style="color: #c00000;">**99.94**</span> |
| [Heading RMSE](#metric-heading-rmse) at ±1.5 m/s [degrees] ↓ | <span style="color: #c00000;">**30.9**</span> → 33.1 |

Speed RMSE improves from ±1 m/s onward. Action clipping and normalized torque cost decrease at all four speeds. Touchdown timing and exact gait-template fidelity do not consistently improve; heading and world progress worsen at the two higher speeds.

### X1_go2bend

| Outcome | No TRS → TRS |
| --- | ---: |
| [Speed RMSE](#metric-velocity-rmse) at ±1.5 m/s [m/s] ↓ | 0.100 → <span style="color: #c00000;">**0.086**</span> |
| [Completion](#metric-completion) at ±2 m/s [%] ↑ | 100.00 → 100.00 |
| [Heading RMSE](#metric-heading-rmse) at ±1.5 m/s [degrees] ↓ | 46.8 → <span style="color: #c00000;">**20.9**</span> |

Both methods complete every trial. Contact agreement, heading retention, and directed world progress improve at all four speeds. Speed tracking is mixed at lower speeds; mechanical power usually increases, and exact template matching falls at the two higher speeds.

### Go2

| Outcome | No TRS → TRS |
| --- | ---: |
| [Speed RMSE](#metric-velocity-rmse) at ±1.5 m/s [m/s] ↓ | 0.219 → <span style="color: #c00000;">**0.171**</span> |
| [Completion](#metric-completion) at ±2 m/s [%] ↑ | 81.56 → <span style="color: #c00000;">**99.94**</span> |
| [Heading RMSE](#metric-heading-rmse) at ±1.5 m/s [degrees] ↓ | <span style="color: #c00000;">**47.5**</span> → 58.8 |

Speed RMSE improves from ±1 m/s onward, and touchdown timing improves at every speed. At ±1.5 and ±2 m/s, yaw-rate error, heading drift, and mechanical power increase while world progress decreases.

> **Scope:** one training seed per method and robot, evaluated with nominal replicas. These results do not establish statistical significance across training seeds or quantify training-reward and convergence-speed advantages.

This report combines the six-speed sweep from `evaluation_batch.sh` with the existing ±1 m/s evaluations, covering **38,400 planned trials**. It supersedes the earlier ±1-only comparison and includes every trained gait in both directions. Lateral and yaw command tests are outside its scope.

## Contents

- [How to read the tables](#how-to-read-the-tables)
- [Tracking, gait fidelity, and completion](#tracking-gait-fidelity-and-completion)
- [Additional paired metrics](#additional-paired-metrics)
- [Forward/backward speed consistency](#forwardbackward-speed-consistency)
- [Early terminations and matched-completion checks](#early-terminations-and-matched-completion-checks)
- [Separate positive and negative commands](#separate-positive-and-negative-commands)
- [Runs and evaluation protocol](#runs-and-evaluation-protocol)
- [Metric definitions](#metric-definitions)
- [Corrections and limitations](#corrections-and-limitations)
- [Appendix A: each gait, pooled over both directions](#appendix-a-each-gait-pooled-over-both-directions) — expandable tables by robot
- [Appendix B: all metrics for every gait and direction](#appendix-b-all-metrics-for-every-gait-and-direction) — expandable tables by robot and speed
- [Source manifests and audit provenance](#source-manifests-and-audit-provenance) — manifests and checkpoint hashes

## How to read the tables

Every paired entry is **No TRS → TRS**. **Bold red** marks the preferred displayed value; ties at the displayed precision are unbolded. RMSE means root mean square error. Speeds (`vx`, `vy`) are measured in the robot's body frame; heading and directed progress describe its world-frame motion.

Click a metric name in any result table to jump to its mathematical definition.

| Symbol | How to interpret it |
| --- | --- |
| ↓ / ↑ | Lower / higher is preferred for that diagnostic. |
| ↓\* | Lower effort is preferable when tracking, gait execution, completion, and displacement are comparable. |
| ↓† | Lower front/rear imbalance means more even loading; the best balance depends on the gait. |
| ≈100% / ≈1 | Closer to 100% directed progress / one touchdown per foot per cycle is preferred. |
| N/A | No full rollout was available for that method in that condition; its continuous metric cannot be compared. |

**Which trials count:**

- **Continuous metrics:** arithmetic means of per-trial metrics from **full rollouts only**. For example, each trial's RMSE is computed first, then those RMSE values are averaged. Failed and partial rollouts are excluded.
- **Classifier, completion, and success rates:** percentages of **all planned trials**. Early terminations count as unsuccessful.
- **Pooled ±speed tables:** combine eight templates and both directions. They have equal weight when every trial completes. At ±2 m/s, the completed-trial populations differ; use the [matched-completion check](#comparing-the-same-completed-trials) alongside the pooled means.

## Tracking, gait fidelity, and completion

Forward and backward vx errors are shown separately; yaw, contact agreement, classification, and completion pool both directions. Yaw-rate RMSE has units rad/s, and contact/classifier/completion values are percentages.

### X1

| Command pair [m/s] | [+vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [−vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) ↓ | [Contact agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**0.050**</span> → 0.052 | 0.047 → <span style="color: #c00000;">**0.046**</span> | 0.072 → <span style="color: #c00000;">**0.056**</span> | 94.6 → <span style="color: #c00000;">**95.7**</span> | <span style="color: #c00000;">**81.1**</span> → 75.0 | <span style="color: #c00000;">**81.1**</span> → 75.0 | 100.00 → 100.00 |
| ±1 | <span style="color: #c00000;">**0.070**</span> → 0.085 | 0.101 → <span style="color: #c00000;">**0.064**</span> | 0.079 → <span style="color: #c00000;">**0.064**</span> | 92.4 → <span style="color: #c00000;">**93.2**</span> | 75.0 → 75.0 | 75.0 → 75.0 | 100.00 → 100.00 |
| ±1.5 | 0.094 → <span style="color: #c00000;">**0.073**</span> | 0.123 → <span style="color: #c00000;">**0.087**</span> | 0.100 → <span style="color: #c00000;">**0.092**</span> | 87.5 → <span style="color: #c00000;">**87.6**</span> | 50.0 → 50.0 | 52.7 → <span style="color: #c00000;">**62.2**</span> | 100.00 → 100.00 |
| ±2 | 0.180 → <span style="color: #c00000;">**0.163**</span> | 0.181 → <span style="color: #c00000;">**0.158**</span> | 0.157 → <span style="color: #c00000;">**0.144**</span> | 80.2 → <span style="color: #c00000;">**81.2**</span> | <span style="color: #c00000;">**40.4**</span> → 30.9 | 45.0 → <span style="color: #c00000;">**55.6**</span> | 94.31 → <span style="color: #c00000;">**99.94**</span> |

### X1_go2bend

| Command pair [m/s] | [+vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [−vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) ↓ | [Contact agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | 0.060 → <span style="color: #c00000;">**0.052**</span> | <span style="color: #c00000;">**0.057**</span> → 0.083 | 0.089 → <span style="color: #c00000;">**0.066**</span> | 91.9 → <span style="color: #c00000;">**94.3**</span> | 50.0 → <span style="color: #c00000;">**68.8**</span> | 50.0 → <span style="color: #c00000;">**68.8**</span> | 100.00 → 100.00 |
| ±1 | <span style="color: #c00000;">**0.076**</span> → 0.090 | <span style="color: #c00000;">**0.074**</span> → 0.089 | 0.094 → <span style="color: #c00000;">**0.090**</span> | 89.5 → <span style="color: #c00000;">**91.6**</span> | 43.8 → <span style="color: #c00000;">**50.0**</span> | <span style="color: #c00000;">**56.2**</span> → 50.0 | 100.00 → 100.00 |
| ±1.5 | <span style="color: #c00000;">**0.074**</span> → 0.077 | 0.126 → <span style="color: #c00000;">**0.095**</span> | 0.110 → <span style="color: #c00000;">**0.084**</span> | 84.9 → <span style="color: #c00000;">**86.2**</span> | <span style="color: #c00000;">**37.5**</span> → 31.2 | <span style="color: #c00000;">**50.0**</span> → 37.5 | 100.00 → 100.00 |
| ±2 | 0.100 → <span style="color: #c00000;">**0.094**</span> | 0.287 → <span style="color: #c00000;">**0.216**</span> | 0.134 → <span style="color: #c00000;">**0.105**</span> | 78.5 → <span style="color: #c00000;">**79.4**</span> | <span style="color: #c00000;">**43.6**</span> → 37.6 | <span style="color: #c00000;">**56.1**</span> → 50.1 | 100.00 → 100.00 |

### Go2

| Command pair [m/s] | [+vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [−vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) ↓ | [Contact agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**0.043**</span> → 0.049 | <span style="color: #c00000;">**0.045**</span> → 0.047 | <span style="color: #c00000;">**0.064**</span> → 0.065 | 95.8 → <span style="color: #c00000;">**96.5**</span> | 93.8 → 93.8 | 93.8 → 93.8 | 100.00 → 100.00 |
| ±1 | 0.071 → <span style="color: #c00000;">**0.066**</span> | 0.095 → <span style="color: #c00000;">**0.061**</span> | <span style="color: #c00000;">**0.073**</span> → 0.083 | 93.4 → <span style="color: #c00000;">**94.1**</span> | 68.8 → <span style="color: #c00000;">**75.0**</span> | 68.8 → <span style="color: #c00000;">**75.0**</span> | 100.00 → 100.00 |
| ±1.5 | <span style="color: #c00000;">**0.144**</span> → 0.156 | 0.294 → <span style="color: #c00000;">**0.187**</span> | <span style="color: #c00000;">**0.102**</span> → 0.123 | 90.1 → <span style="color: #c00000;">**90.9**</span> | 53.5 → <span style="color: #c00000;">**68.8**</span> | 53.5 → <span style="color: #c00000;">**68.8**</span> | 100.00 → 100.00 |
| ±2 | 0.472 → <span style="color: #c00000;">**0.459**</span> | 0.714 → <span style="color: #c00000;">**0.528**</span> | <span style="color: #c00000;">**0.132**</span> → 0.153 | 86.6 → <span style="color: #c00000;">**88.5**</span> | 31.1 → <span style="color: #c00000;">**64.4**</span> | 31.1 → <span style="color: #c00000;">**64.4**</span> | 81.56 → <span style="color: #c00000;">**99.94**</span> |

## Additional paired metrics

### Pooled velocity errors

**Takeaway:** all three robots have lower pooled speed RMSE with TRS at ±1.5 and ±2 m/s. At the two lower speeds, X1_go2bend has higher pooled speed RMSE.

#### X1

| Command pair [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | 0.049 → 0.049 | 0.041 → <span style="color: #c00000;">**0.035**</span> | <span style="color: #c00000;">**9.73**</span> → 9.75 |
| ±1 | 0.085 → <span style="color: #c00000;">**0.075**</span> | 0.041 → <span style="color: #c00000;">**0.040**</span> | 8.54 → <span style="color: #c00000;">**7.48**</span> |
| ±1.5 | 0.108 → <span style="color: #c00000;">**0.080**</span> | 0.057 → 0.057 | 7.23 → <span style="color: #c00000;">**5.31**</span> |
| ±2 | 0.181 → <span style="color: #c00000;">**0.160**</span> | <span style="color: #c00000;">**0.086**</span> → 0.087 | 9.03 → <span style="color: #c00000;">**8.00**</span> |

#### X1_go2bend

| Command pair [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**0.059**</span> → 0.067 | <span style="color: #c00000;">**0.044**</span> → 0.045 | <span style="color: #c00000;">**11.76**</span> → 13.43 |
| ±1 | <span style="color: #c00000;">**0.075**</span> → 0.089 | <span style="color: #c00000;">**0.045**</span> → 0.056 | <span style="color: #c00000;">**7.48**</span> → 8.91 |
| ±1.5 | 0.100 → <span style="color: #c00000;">**0.086**</span> | 0.069 → <span style="color: #c00000;">**0.067**</span> | 6.67 → <span style="color: #c00000;">**5.73**</span> |
| ±2 | 0.194 → <span style="color: #c00000;">**0.155**</span> | 0.101 → <span style="color: #c00000;">**0.079**</span> | 9.68 → <span style="color: #c00000;">**7.74**</span> |

#### Go2

| Command pair [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**0.044**</span> → 0.048 | <span style="color: #c00000;">**0.064**</span> → 0.065 | <span style="color: #c00000;">**8.84**</span> → 9.62 |
| ±1 | 0.083 → <span style="color: #c00000;">**0.064**</span> | 0.072 → <span style="color: #c00000;">**0.070**</span> | 8.30 → <span style="color: #c00000;">**6.37**</span> |
| ±1.5 | 0.219 → <span style="color: #c00000;">**0.171**</span> | 0.103 → <span style="color: #c00000;">**0.093**</span> | 14.60 → <span style="color: #c00000;">**11.41**</span> |
| ±2 | 0.567 → <span style="color: #c00000;">**0.494**</span> | <span style="color: #c00000;">**0.110**</span> → 0.138 | 28.33 → <span style="color: #c00000;">**24.68**</span> |

### Contact timing and duty tracking

**Takeaway:** Go2 touchdown timing improves at every speed. The two X1 configurations have mixed timing results, even though their contact agreement improves throughout.

Touchdown timing measures alignment with the commanded schedule, not the timing difference between two legs. Duty error is measured in percentage points (pp); measured duty is shown as a percentage with target 56%.

#### X1

| Command pair [m/s] | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Duty error](#metric-duty-error) [pp] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**16.8**</span> → 18.6 | 2.91 → <span style="color: #c00000;">**2.65**</span> | 53.77 → <span style="color: #c00000;">**54.55**</span> |
| ±1 | <span style="color: #c00000;">**21.2**</span> → 23.2 | 4.98 → <span style="color: #c00000;">**4.49**</span> | 51.54 → <span style="color: #c00000;">**51.65**</span> |
| ±1.5 | 29.7 → <span style="color: #c00000;">**29.6**</span> | <span style="color: #c00000;">**7.99**</span> → 8.40 | <span style="color: #c00000;">**48.03**</span> → 47.67 |
| ±2 | <span style="color: #c00000;">**38.3**</span> → 41.0 | 12.58 → <span style="color: #c00000;">**11.73**</span> | 43.43 → <span style="color: #c00000;">**44.29**</span> |

#### X1_go2bend

| Command pair [m/s] | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Duty error](#metric-duty-error) [pp] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 |
| ---: | ---: | ---: | ---: |
| ±0.5 | 24.5 → <span style="color: #c00000;">**23.5**</span> | 4.30 → <span style="color: #c00000;">**3.36**</span> | 54.07 → <span style="color: #c00000;">**55.14**</span> |
| ±1 | 32.8 → <span style="color: #c00000;">**28.8**</span> | 5.47 → <span style="color: #c00000;">**4.50**</span> | 51.70 → <span style="color: #c00000;">**52.44**</span> |
| ±1.5 | <span style="color: #c00000;">**38.8**</span> → 41.7 | <span style="color: #c00000;">**7.81**</span> → 7.86 | <span style="color: #c00000;">**48.37**</span> → 48.24 |
| ±2 | <span style="color: #c00000;">**43.1**</span> → 46.8 | 12.53 → <span style="color: #c00000;">**12.47**</span> | 43.49 → <span style="color: #c00000;">**43.53**</span> |

#### Go2

| Command pair [m/s] | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Duty error](#metric-duty-error) [pp] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 |
| ---: | ---: | ---: | ---: |
| ±0.5 | 13.8 → <span style="color: #c00000;">**13.0**</span> | 4.90 → <span style="color: #c00000;">**4.49**</span> | <span style="color: #c00000;">**57.46**</span> → 57.96 |
| ±1 | 16.8 → <span style="color: #c00000;">**15.7**</span> | 5.48 → <span style="color: #c00000;">**4.49**</span> | 55.50 → <span style="color: #c00000;">**55.90**</span> |
| ±1.5 | 24.3 → <span style="color: #c00000;">**17.3**</span> | 5.96 → <span style="color: #c00000;">**5.50**</span> | 53.15 → <span style="color: #c00000;">**53.33**</span> |
| ±2 | 29.6 → <span style="color: #c00000;">**22.8**</span> | <span style="color: #c00000;">**5.70**</span> → 6.30 | <span style="color: #c00000;">**52.92**</span> → 51.52 |

### Contact errors and repeated contacts

**Takeaway:** better contact agreement does not guarantee fewer repeated contacts. At ±1.5 m/s, touchdowns/cycle change from **1.24 → 1.04** for X1, **1.29 → 1.41** for X1_go2bend, and **1.32 → 1.51** for Go2.

The last column is the average, across trials, of the largest per-foot touchdown count divided by complete cycles within each trial. Approximately one is desired; it is not the maximum over the entire study.

#### X1

| Command pair [m/s] | [Contact during commanded swing](#metric-swing-contact) [%] ↓ | [Missed commanded stance](#metric-missed-stance) [%] ↓ | [Max-foot touchdowns/cycle](#metric-touchdowns-per-cycle) ≈1 |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**3.72**</span> → 3.78 | 6.71 → <span style="color: #c00000;">**4.62**</span> | 1.00 → 1.00 |
| ±1 | 3.76 → <span style="color: #c00000;">**3.67**</span> | 10.42 → <span style="color: #c00000;">**9.20**</span> | 1.00 → 1.00 |
| ±1.5 | 5.63 → <span style="color: #c00000;">**5.23**</span> | <span style="color: #c00000;">**17.57**</span> → 17.71 | 1.24 → <span style="color: #c00000;">**1.04**</span> |
| ±2 | 8.37 → <span style="color: #c00000;">**8.10**</span> | 28.45 → <span style="color: #c00000;">**26.79**</span> | 1.71 → <span style="color: #c00000;">**1.30**</span> |

#### X1_go2bend

| Command pair [m/s] | [Contact during commanded swing](#metric-swing-contact) [%] ↓ | [Missed commanded stance](#metric-missed-stance) [%] ↓ | [Max-foot touchdowns/cycle](#metric-touchdowns-per-cycle) ≈1 |
| ---: | ---: | ---: | ---: |
| ±0.5 | 6.98 → <span style="color: #c00000;">**5.21**</span> | 8.88 → <span style="color: #c00000;">**6.02**</span> | 1.00 → 1.00 |
| ±1 | 7.12 → <span style="color: #c00000;">**5.89**</span> | 12.97 → <span style="color: #c00000;">**10.27**</span> | <span style="color: #c00000;">**1.06**</span> → 1.09 |
| ±1.5 | 8.73 → <span style="color: #c00000;">**7.51**</span> | 19.87 → <span style="color: #c00000;">**18.48**</span> | <span style="color: #c00000;">**1.29**</span> → 1.41 |
| ±2 | 10.02 → <span style="color: #c00000;">**9.50**</span> | 30.15 → <span style="color: #c00000;">**28.87**</span> | 1.71 → <span style="color: #c00000;">**1.57**</span> |

#### Go2

| Command pair [m/s] | [Contact during commanded swing](#metric-swing-contact) [%] ↓ | [Missed commanded stance](#metric-missed-stance) [%] ↓ | [Max-foot touchdowns/cycle](#metric-touchdowns-per-cycle) ≈1 |
| ---: | ---: | ---: | ---: |
| ±0.5 | 5.21 → <span style="color: #c00000;">**5.07**</span> | 3.37 → <span style="color: #c00000;">**2.25**</span> | 1.00 → 1.00 |
| ±1 | 6.37 → <span style="color: #c00000;">**6.22**</span> | 6.78 → <span style="color: #c00000;">**5.59**</span> | <span style="color: #c00000;">**1.28**</span> → 1.31 |
| ±1.5 | 8.05 → <span style="color: #c00000;">**7.38**</span> | 11.32 → <span style="color: #c00000;">**10.42**</span> | <span style="color: #c00000;">**1.32**</span> → 1.51 |
| ±2 | 11.86 → <span style="color: #c00000;">**8.49**</span> | 14.47 → <span style="color: #c00000;">**13.74**</span> | 1.64 → <span style="color: #c00000;">**1.61**</span> |

### Actuator effort and action clipping

**Takeaway:** action clipping decreases at every speed for both X1 configurations, and X1 normalized torque cost also decreases throughout. Go2 mechanical power increases at the two higher speeds.

Mechanical power is summed absolute joint mechanical power, not electrical power. Normalized torque cost is dimensionless. Lower clipping indicates fewer target-constraint interventions; it does not establish smooth motion.

#### X1

| Command pair [m/s] | [Mechanical power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Action clipping](#metric-action-clipping) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**80.6**</span> → 81.4 | 0.598 → <span style="color: #c00000;">**0.580**</span> | 0.07 → <span style="color: #c00000;">**0.00**</span> |
| ±1 | 134.3 → <span style="color: #c00000;">**124.2**</span> | 0.819 → <span style="color: #c00000;">**0.758**</span> | 0.63 → <span style="color: #c00000;">**0.06**</span> |
| ±1.5 | 231.4 → <span style="color: #c00000;">**217.0**</span> | 1.232 → <span style="color: #c00000;">**1.138**</span> | 2.09 → <span style="color: #c00000;">**0.73**</span> |
| ±2 | 312.4 → <span style="color: #c00000;">**300.0**</span> | 1.516 → <span style="color: #c00000;">**1.471**</span> | 3.84 → <span style="color: #c00000;">**2.22**</span> |

#### X1_go2bend

| Command pair [m/s] | [Mechanical power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Action clipping](#metric-action-clipping) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**90.7**</span> → 97.2 | 0.722 → <span style="color: #c00000;">**0.659**</span> | 2.85 → <span style="color: #c00000;">**1.86**</span> |
| ±1 | <span style="color: #c00000;">**142.8**</span> → 150.8 | 0.900 → <span style="color: #c00000;">**0.852**</span> | 3.39 → <span style="color: #c00000;">**2.19**</span> |
| ±1.5 | <span style="color: #c00000;">**211.8**</span> → 215.9 | <span style="color: #c00000;">**1.090**</span> → 1.096 | 4.17 → <span style="color: #c00000;">**2.72**</span> |
| ±2 | 293.6 → <span style="color: #c00000;">**288.9**</span> | 1.281 → <span style="color: #c00000;">**1.264**</span> | 4.60 → <span style="color: #c00000;">**2.91**</span> |

#### Go2

| Command pair [m/s] | [Mechanical power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Action clipping](#metric-action-clipping) [%] ↓ |
| ---: | ---: | ---: | ---: |
| ±0.5 | 75.5 → <span style="color: #c00000;">**66.2**</span> | <span style="color: #c00000;">**0.277**</span> → 0.279 | 1.22 → <span style="color: #c00000;">**0.00**</span> |
| ±1 | 103.0 → <span style="color: #c00000;">**100.1**</span> | 0.382 → <span style="color: #c00000;">**0.372**</span> | 1.35 → <span style="color: #c00000;">**0.30**</span> |
| ±1.5 | <span style="color: #c00000;">**145.9**</span> → 170.1 | <span style="color: #c00000;">**0.522**</span> → 0.532 | 1.86 → <span style="color: #c00000;">**1.39**</span> |
| ±2 | <span style="color: #c00000;">**187.7**</span> → 231.3 | <span style="color: #c00000;">**0.611**</span> → 0.648 | <span style="color: #c00000;">**2.75**</span> → 3.16 |

### Front/rear load balance

Each imbalance is the absolute front/rear difference divided by their sum, expressed as a percentage. Values are averaged after taking each trial's absolute imbalance; signed differences are not allowed to cancel across directions.

#### X1

| Command pair [m/s] | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Vertical-force impulse imbalance](#metric-front-rear-imbalance) [%] ↓† | [Normalized torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| ---: | ---: | ---: | ---: |
| ±0.5 | 5.43 → <span style="color: #c00000;">**4.08**</span> | <span style="color: #c00000;">**3.13**</span> → 3.62 | 11.83 → <span style="color: #c00000;">**7.53**</span> |
| ±1 | 7.48 → <span style="color: #c00000;">**3.80**</span> | 5.43 → <span style="color: #c00000;">**2.91**</span> | 14.24 → <span style="color: #c00000;">**8.12**</span> |
| ±1.5 | <span style="color: #c00000;">**7.90**</span> → 10.76 | <span style="color: #c00000;">**7.52**</span> → 7.63 | <span style="color: #c00000;">**8.46**</span> → 10.21 |
| ±2 | <span style="color: #c00000;">**6.40**</span> → 9.80 | <span style="color: #c00000;">**6.61**</span> → 7.13 | <span style="color: #c00000;">**6.92**</span> → 12.47 |

#### X1_go2bend

| Command pair [m/s] | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Vertical-force impulse imbalance](#metric-front-rear-imbalance) [%] ↓† | [Normalized torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| ---: | ---: | ---: | ---: |
| ±0.5 | 24.63 → <span style="color: #c00000;">**21.45**</span> | <span style="color: #c00000;">**8.79**</span> → 12.57 | <span style="color: #c00000;">**7.02**</span> → 9.33 |
| ±1 | 16.28 → <span style="color: #c00000;">**16.26**</span> | <span style="color: #c00000;">**7.01**</span> → 9.03 | <span style="color: #c00000;">**7.11**</span> → 8.05 |
| ±1.5 | <span style="color: #c00000;">**9.53**</span> → 9.79 | <span style="color: #c00000;">**5.40**</span> → 7.41 | 9.13 → <span style="color: #c00000;">**6.00**</span> |
| ±2 | <span style="color: #c00000;">**6.93**</span> → 8.30 | 4.29 → <span style="color: #c00000;">**4.26**</span> | 7.88 → <span style="color: #c00000;">**6.40**</span> |

#### Go2

| Command pair [m/s] | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Vertical-force impulse imbalance](#metric-front-rear-imbalance) [%] ↓† | [Normalized torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| ---: | ---: | ---: | ---: |
| ±0.5 | 20.05 → <span style="color: #c00000;">**11.72**</span> | 29.92 → <span style="color: #c00000;">**28.13**</span> | 39.88 → <span style="color: #c00000;">**35.05**</span> |
| ±1 | 15.38 → <span style="color: #c00000;">**12.64**</span> | 27.27 → <span style="color: #c00000;">**26.46**</span> | 39.57 → <span style="color: #c00000;">**35.40**</span> |
| ±1.5 | <span style="color: #c00000;">**13.56**</span> → 15.42 | <span style="color: #c00000;">**26.68**</span> → 28.74 | 38.13 → <span style="color: #c00000;">**35.89**</span> |
| ±2 | <span style="color: #c00000;">**11.94**</span> → 14.89 | <span style="color: #c00000;">**25.39**</span> → 30.42 | 33.75 → <span style="color: #c00000;">**33.62**</span> |

### Heading retention and directed world progress

**Takeaway:** X1_go2bend improves on both measures at all four speeds. X1 and Go2 improve at the two lower speeds and worsen at the two higher speeds.

Good body-frame velocity tracking can coexist with poor world progress if the robot turns away from its initial heading. Progress is displacement along the requested initial world direction, divided by requested distance.

#### X1

| Command pair [m/s] | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Directed progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: |
| ±0.5 | 51.9 → <span style="color: #c00000;">**23.1**</span> | 53.5 → <span style="color: #c00000;">**82.1**</span> |
| ±1 | 43.6 → <span style="color: #c00000;">**39.7**</span> | 61.4 → <span style="color: #c00000;">**66.2**</span> |
| ±1.5 | <span style="color: #c00000;">**30.9**</span> → 33.1 | <span style="color: #c00000;">**77.4**</span> → 76.7 |
| ±2 | <span style="color: #c00000;">**43.8**</span> → 46.5 | <span style="color: #c00000;">**60.2**</span> → 58.2 |

#### X1_go2bend

| Command pair [m/s] | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Directed progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: |
| ±0.5 | 47.6 → <span style="color: #c00000;">**43.7**</span> | 63.4 → <span style="color: #c00000;">**64.7**</span> |
| ±1 | 43.2 → <span style="color: #c00000;">**42.1**</span> | 64.3 → <span style="color: #c00000;">**66.4**</span> |
| ±1.5 | 46.8 → <span style="color: #c00000;">**20.9**</span> | 62.2 → <span style="color: #c00000;">**88.5**</span> |
| ±2 | 41.9 → <span style="color: #c00000;">**33.2**</span> | 63.0 → <span style="color: #c00000;">**72.7**</span> |

#### Go2

| Command pair [m/s] | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Directed progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: |
| ±0.5 | 43.3 → <span style="color: #c00000;">**22.6**</span> | 64.7 → <span style="color: #c00000;">**84.1**</span> |
| ±1 | 40.1 → <span style="color: #c00000;">**38.6**</span> | 68.3 → <span style="color: #c00000;">**72.0**</span> |
| ±1.5 | <span style="color: #c00000;">**47.5**</span> → 58.8 | <span style="color: #c00000;">**54.4**</span> → 43.4 |
| ±2 | <span style="color: #c00000;">**55.8**</span> → 69.6 | <span style="color: #c00000;">**37.6**</span> → 27.7 |

### Success rates

**Takeaway:** velocity and combined success are **0% for every robot and method at ±1.5 and ±2 m/s** under these thresholds. Higher completion and lower speed error therefore do not guarantee success on the full set of criteria.

- **Velocity success:** relative vx RMSE ≤0.20, yaw-rate RMSE ≤0.05 rad/s, positive directed progress, and no termination.
- **Gait success:** no termination, contact agreement ≥0.80, correct gait family, at least two complete cycles, and at least two touchdown and liftoff events per foot.
- **Combined success:** satisfies both velocity and gait criteria.
- **Positive-progress trials:** a separate diagnostic; requires completion and positive net displacement along the requested world direction.

#### X1

| Command pair [m/s] | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive-progress trials](#metric-positive-progress) [%] ↑ |
| ---: | ---: | ---: | ---: | ---: |
| ±0.5 | 25.00 → <span style="color: #c00000;">**50.00**</span> | <span style="color: #c00000;">**81.12**</span> → 75.00 | 25.00 → <span style="color: #c00000;">**50.00**</span> | 86.94 → <span style="color: #c00000;">**100.00**</span> |
| ±1 | 12.50 → 12.50 | 75.00 → 75.00 | 12.50 → 12.50 | 81.25 → <span style="color: #c00000;">**93.75**</span> |
| ±1.5 | 0.00 → 0.00 | 50.00 → 50.00 | 0.00 → 0.00 | <span style="color: #c00000;">**100.00**</span> → 93.75 |
| ±2 | 0.00 → 0.00 | <span style="color: #c00000;">**38.88**</span> → 30.88 | 0.00 → 0.00 | 87.19 → <span style="color: #c00000;">**94.25**</span> |

#### X1_go2bend

| Command pair [m/s] | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive-progress trials](#metric-positive-progress) [%] ↑ |
| ---: | ---: | ---: | ---: | ---: |
| ±0.5 | 12.50 → <span style="color: #c00000;">**25.00**</span> | 50.00 → <span style="color: #c00000;">**68.75**</span> | 12.50 → <span style="color: #c00000;">**18.75**</span> | 81.25 → <span style="color: #c00000;">**93.75**</span> |
| ±1 | <span style="color: #c00000;">**6.25**</span> → 0.00 | 43.75 → <span style="color: #c00000;">**50.00**</span> | 0.00 → 0.00 | 87.50 → 87.50 |
| ±1.5 | 0.00 → 0.00 | <span style="color: #c00000;">**37.50**</span> → 31.25 | 0.00 → 0.00 | 93.75 → <span style="color: #c00000;">**100.00**</span> |
| ±2 | 0.00 → 0.00 | <span style="color: #c00000;">**37.31**</span> → 31.31 | 0.00 → 0.00 | 93.75 → <span style="color: #c00000;">**96.81**</span> |

#### Go2

| Command pair [m/s] | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive-progress trials](#metric-positive-progress) [%] ↑ |
| ---: | ---: | ---: | ---: | ---: |
| ±0.5 | <span style="color: #c00000;">**37.50**</span> → 6.25 | 93.75 → 93.75 | <span style="color: #c00000;">**31.25**</span> → 6.25 | 93.75 → <span style="color: #c00000;">**100.00**</span> |
| ±1 | <span style="color: #c00000;">**12.50**</span> → 6.25 | 68.75 → <span style="color: #c00000;">**75.00**</span> | 6.25 → 6.25 | <span style="color: #c00000;">**100.00**</span> → 93.75 |
| ±1.5 | 0.00 → 0.00 | 53.50 → <span style="color: #c00000;">**68.75**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**84.56**</span> → 81.25 |
| ±2 | 0.00 → 0.00 | 31.12 → <span style="color: #c00000;">**64.44**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**75.31**</span> → 68.19 |

## Forward/backward speed consistency

**Takeaway:** X1 and Go2 improve from ±1 m/s onward. X1_go2bend improves only at ±2 m/s under this metric.

This comparison pairs the forward rollout of template g with the backward rollout of its **time-reversed template TR(g)**, rather than necessarily the same named template. The statistic is the mean of `abs(mean_vx(g,+v) + mean_vx(TR(g),−v))` in m/s.

Only pairs for which all four relevant rollouts complete in both methods are included, so each row compares the same cohort. Lower bias indicates more consistent speed under reversal; it does not establish full action or trajectory symmetry.

### X1

| Command pair [m/s] | [Matched completed directional pairs](#metric-matched-directional-pairs) | [TR-partner speed bias](#metric-tr-partner-speed-bias) [m/s] ↓ |
| ---: | ---: | ---: |
| ±0.5 | 800 | <span style="color: #c00000;">**0.028**</span> → 0.032 |
| ±1 | 800 | 0.083 → <span style="color: #c00000;">**0.050**</span> |
| ±1.5 | 800 | 0.119 → <span style="color: #c00000;">**0.047**</span> |
| ±2 | 708 | 0.108 → <span style="color: #c00000;">**0.089**</span> |

### X1_go2bend

| Command pair [m/s] | [Matched completed directional pairs](#metric-matched-directional-pairs) | [TR-partner speed bias](#metric-tr-partner-speed-bias) [m/s] ↓ |
| ---: | ---: | ---: |
| ±0.5 | 800 | <span style="color: #c00000;">**0.045**</span> → 0.072 |
| ±1 | 800 | <span style="color: #c00000;">**0.044**</span> → 0.100 |
| ±1.5 | 800 | <span style="color: #c00000;">**0.089**</span> → 0.091 |
| ±2 | 800 | 0.289 → <span style="color: #c00000;">**0.221**</span> |

### Go2

| Command pair [m/s] | [Matched completed directional pairs](#metric-matched-directional-pairs) | [TR-partner speed bias](#metric-tr-partner-speed-bias) [m/s] ↓ |
| ---: | ---: | ---: |
| ±0.5 | 800 | <span style="color: #c00000;">**0.032**</span> → 0.046 |
| ±1 | 800 | 0.087 → <span style="color: #c00000;">**0.041**</span> |
| ±1.5 | 800 | 0.185 → <span style="color: #c00000;">**0.049**</span> |
| ±2 | 509 | 0.222 → <span style="color: #c00000;">**0.087**</span> |

## Early terminations and matched-completion checks

All trials at ±0.5, ±1, and ±1.5 complete. There are **388 early terminations**, all at ±2: 91 X1 no-TRS, 1 X1 TRS, 295 Go2 no-TRS, and 1 Go2 TRS. Both X1_go2bend methods complete every trial.

### Where rollouts ended early

Termination labels below identify the environment condition that caused reset; `calf_height` does not by itself establish that the robot fell. Counts include failures before the measured interval, which can be missing from the original metrics CSV's `terminated` field.

#### X1

| Method | Requested gait | vx [m/s] | [Terminated / planned](#metric-termination-counts) | Termination condition |
| ---: | ---: | ---: | ---: | ---: |
| TRS | Half-bound A | +2 | 1/100 | base_height: 1 |
| No TRS | Half-bound A | -2 | 91/100 | base_height: 91 |

#### X1_go2bend

| Method | Requested gait | vx [m/s] | [Terminated / planned](#metric-termination-counts) | Termination condition |
| ---: | ---: | ---: | ---: | ---: |
| No TRS | All eight templates | ±2 | 0/1,600 | None |
| TRS | All eight templates | ±2 | 0/1,600 | None |

#### Go2

| Method | Requested gait | vx [m/s] | [Terminated / planned](#metric-termination-counts) | Termination condition |
| ---: | ---: | ---: | ---: | ---: |
| No TRS | Trot | -2 | 100/100 | calf_height: 100 |
| No TRS | Half-bound B | -2 | 100/100 | calf_height: 100 |
| No TRS | Transverse A | +2 | 5/100 | calf_height: 5 |
| TRS | Transverse B | +2 | 1/100 | calf_height: 1 |
| No TRS | Transverse B | -2 | 90/100 | calf_height: 90 |

### Comparing the same completed trials

At ±2 m/s, the tracking improvements persist when comparing the same completed trials: vx RMSE decreases for all three robots, while high-speed heading and progress remain worse for X1 and Go2.

The tables below restrict continuous metrics to identical trial identities that complete in both methods. This restriction reduces differences in cohort composition, but still excludes conditions that fail under either method and therefore is not an unconditional performance estimate.

#### X1

| [Matched full trials](#metric-matched-completion) / 1,600 | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1508 | 0.181 → <span style="color: #c00000;">**0.161**</span> | 0.157 → <span style="color: #c00000;">**0.147**</span> | 80.1 → <span style="color: #c00000;">**81.3**</span> | <span style="color: #c00000;">**43.8**</span> → 48.1 | <span style="color: #c00000;">**60.1**</span> → 56.4 |

#### X1_go2bend

| [Matched full trials](#metric-matched-completion) / 1,600 | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1600 | 0.194 → <span style="color: #c00000;">**0.155**</span> | 0.134 → <span style="color: #c00000;">**0.105**</span> | 78.5 → <span style="color: #c00000;">**79.4**</span> | 41.9 → <span style="color: #c00000;">**33.2**</span> | 63.0 → <span style="color: #c00000;">**72.7**</span> |

#### Go2

| [Matched full trials](#metric-matched-completion) / 1,600 | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1304 | 0.567 → <span style="color: #c00000;">**0.490**</span> | <span style="color: #c00000;">**0.132**</span> → 0.147 | 86.6 → <span style="color: #c00000;">**88.3**</span> | <span style="color: #c00000;">**55.8**</span> → 68.8 | <span style="color: #c00000;">**37.7**</span> → 28.7 |

## Separate positive and negative commands

Each row pools eight templates at one signed velocity; 800 trials are planned per method. Full per-gait metrics are in Appendix B.

### X1: direction-level results

| vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | <span style="color: #c00000;">**0.050**</span> → 0.052 | 0.078 → <span style="color: #c00000;">**0.054**</span> | 94.7 → <span style="color: #c00000;">**95.7**</span> | 59.4 → <span style="color: #c00000;">**21.3**</span> | 46.2 → <span style="color: #c00000;">**81.3**</span> | 100.00 → 100.00 |
| -0.5 | 0.047 → <span style="color: #c00000;">**0.046**</span> | 0.067 → <span style="color: #c00000;">**0.057**</span> | 94.4 → <span style="color: #c00000;">**95.8**</span> | 44.4 → <span style="color: #c00000;">**24.8**</span> | 60.8 → <span style="color: #c00000;">**82.9**</span> | 100.00 → 100.00 |
| +1 | <span style="color: #c00000;">**0.070**</span> → 0.085 | 0.088 → <span style="color: #c00000;">**0.066**</span> | 91.9 → <span style="color: #c00000;">**93.2**</span> | 49.7 → <span style="color: #c00000;">**29.9**</span> | 58.6 → <span style="color: #c00000;">**74.1**</span> | 100.00 → 100.00 |
| -1 | 0.101 → <span style="color: #c00000;">**0.064**</span> | 0.071 → <span style="color: #c00000;">**0.062**</span> | 93.0 → <span style="color: #c00000;">**93.1**</span> | <span style="color: #c00000;">**37.5**</span> → 49.4 | <span style="color: #c00000;">**64.2**</span> → 58.3 | 100.00 → 100.00 |
| +1.5 | 0.094 → <span style="color: #c00000;">**0.073**</span> | 0.109 → <span style="color: #c00000;">**0.089**</span> | <span style="color: #c00000;">**87.6**</span> → 87.3 | 34.3 → <span style="color: #c00000;">**33.6**</span> | 76.5 → <span style="color: #c00000;">**77.8**</span> | 100.00 → 100.00 |
| -1.5 | 0.123 → <span style="color: #c00000;">**0.087**</span> | <span style="color: #c00000;">**0.091**</span> → 0.096 | 87.5 → <span style="color: #c00000;">**88.0**</span> | <span style="color: #c00000;">**27.5**</span> → 32.5 | <span style="color: #c00000;">**78.2**</span> → 75.6 | 100.00 → 100.00 |
| +2 | 0.180 → <span style="color: #c00000;">**0.163**</span> | <span style="color: #c00000;">**0.158**</span> → 0.159 | <span style="color: #c00000;">**81.6**</span> → 81.5 | <span style="color: #c00000;">**31.8**</span> → 59.9 | <span style="color: #c00000;">**72.0**</span> → 43.2 | <span style="color: #c00000;">**100.00**</span> → 99.88 |
| -2 | 0.181 → <span style="color: #c00000;">**0.158**</span> | 0.155 → <span style="color: #c00000;">**0.129**</span> | 78.5 → <span style="color: #c00000;">**81.0**</span> | 57.3 → <span style="color: #c00000;">**33.1**</span> | 46.8 → <span style="color: #c00000;">**73.2**</span> | 88.62 → <span style="color: #c00000;">**100.00**</span> |

### X1_go2bend: direction-level results

| vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | 0.060 → <span style="color: #c00000;">**0.052**</span> | 0.089 → <span style="color: #c00000;">**0.072**</span> | 92.2 → <span style="color: #c00000;">**94.9**</span> | <span style="color: #c00000;">**36.9**</span> → 55.6 | <span style="color: #c00000;">**78.2**</span> → 56.2 | 100.00 → 100.00 |
| -0.5 | <span style="color: #c00000;">**0.057**</span> → 0.083 | 0.089 → <span style="color: #c00000;">**0.061**</span> | 91.7 → <span style="color: #c00000;">**93.8**</span> | 58.2 → <span style="color: #c00000;">**31.8**</span> | 48.7 → <span style="color: #c00000;">**73.2**</span> | 100.00 → 100.00 |
| +1 | <span style="color: #c00000;">**0.076**</span> → 0.090 | 0.096 → 0.096 | 89.7 → <span style="color: #c00000;">**92.2**</span> | <span style="color: #c00000;">**32.0**</span> → 50.7 | <span style="color: #c00000;">**75.0**</span> → 58.4 | 100.00 → 100.00 |
| -1 | <span style="color: #c00000;">**0.074**</span> → 0.089 | 0.092 → <span style="color: #c00000;">**0.084**</span> | 89.4 → <span style="color: #c00000;">**91.1**</span> | 54.4 → <span style="color: #c00000;">**33.6**</span> | 53.6 → <span style="color: #c00000;">**74.4**</span> | 100.00 → 100.00 |
| +1.5 | <span style="color: #c00000;">**0.074**</span> → 0.077 | 0.100 → <span style="color: #c00000;">**0.083**</span> | 84.3 → <span style="color: #c00000;">**87.5**</span> | 31.3 → <span style="color: #c00000;">**18.1**</span> | 78.3 → <span style="color: #c00000;">**92.4**</span> | 100.00 → 100.00 |
| -1.5 | 0.126 → <span style="color: #c00000;">**0.095**</span> | 0.121 → <span style="color: #c00000;">**0.085**</span> | <span style="color: #c00000;">**85.5**</span> → 84.9 | 62.3 → <span style="color: #c00000;">**23.8**</span> | 46.0 → <span style="color: #c00000;">**84.5**</span> | 100.00 → 100.00 |
| +2 | 0.100 → <span style="color: #c00000;">**0.094**</span> | 0.123 → <span style="color: #c00000;">**0.105**</span> | 78.2 → <span style="color: #c00000;">**79.8**</span> | 33.0 → <span style="color: #c00000;">**29.0**</span> | 75.7 → <span style="color: #c00000;">**81.1**</span> | 100.00 → 100.00 |
| -2 | 0.287 → <span style="color: #c00000;">**0.216**</span> | 0.145 → <span style="color: #c00000;">**0.106**</span> | 78.8 → <span style="color: #c00000;">**79.1**</span> | 50.8 → <span style="color: #c00000;">**37.3**</span> | 50.2 → <span style="color: #c00000;">**64.4**</span> | 100.00 → 100.00 |

### Go2: direction-level results

| vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | <span style="color: #c00000;">**0.043**</span> → 0.049 | <span style="color: #c00000;">**0.060**</span> → 0.070 | 95.6 → <span style="color: #c00000;">**96.4**</span> | 33.0 → <span style="color: #c00000;">**26.4**</span> | 77.5 → <span style="color: #c00000;">**81.0**</span> | 100.00 → 100.00 |
| -0.5 | <span style="color: #c00000;">**0.045**</span> → 0.047 | 0.067 → <span style="color: #c00000;">**0.061**</span> | 96.0 → <span style="color: #c00000;">**96.7**</span> | 53.6 → <span style="color: #c00000;">**18.8**</span> | 51.9 → <span style="color: #c00000;">**87.2**</span> | 100.00 → 100.00 |
| +1 | 0.071 → <span style="color: #c00000;">**0.066**</span> | <span style="color: #c00000;">**0.076**</span> → 0.085 | 92.9 → <span style="color: #c00000;">**94.5**</span> | <span style="color: #c00000;">**33.6**</span> → 36.6 | <span style="color: #c00000;">**78.4**</span> → 77.1 | 100.00 → 100.00 |
| -1 | 0.095 → <span style="color: #c00000;">**0.061**</span> | <span style="color: #c00000;">**0.070**</span> → 0.082 | <span style="color: #c00000;">**93.9**</span> → 93.7 | 46.5 → <span style="color: #c00000;">**40.6**</span> | 58.2 → <span style="color: #c00000;">**66.9**</span> | 100.00 → 100.00 |
| +1.5 | <span style="color: #c00000;">**0.144**</span> → 0.156 | <span style="color: #c00000;">**0.099**</span> → 0.106 | 88.8 → <span style="color: #c00000;">**91.3**</span> | <span style="color: #c00000;">**28.2**</span> → 42.8 | <span style="color: #c00000;">**79.4**</span> → 63.2 | 100.00 → 100.00 |
| -1.5 | 0.294 → <span style="color: #c00000;">**0.187**</span> | <span style="color: #c00000;">**0.106**</span> → 0.140 | <span style="color: #c00000;">**91.3**</span> → 90.5 | <span style="color: #c00000;">**66.8**</span> → 74.9 | <span style="color: #c00000;">**29.4**</span> → 23.6 | 100.00 → 100.00 |
| +2 | 0.472 → <span style="color: #c00000;">**0.459**</span> | 0.133 → <span style="color: #c00000;">**0.124**</span> | 86.0 → <span style="color: #c00000;">**88.9**</span> | 59.4 → <span style="color: #c00000;">**40.6**</span> | 38.5 → <span style="color: #c00000;">**54.7**</span> | 99.38 → <span style="color: #c00000;">**99.88**</span> |
| -2 | 0.714 → <span style="color: #c00000;">**0.528**</span> | <span style="color: #c00000;">**0.129**</span> → 0.182 | 87.6 → <span style="color: #c00000;">**88.1**</span> | <span style="color: #c00000;">**50.2**</span> → 98.7 | <span style="color: #c00000;">**36.4**</span> → 0.7 | 63.75 → <span style="color: #c00000;">**100.00**</span> |

## Runs and evaluation protocol

<details>
<summary>Show training run identifiers</summary>

### X1

| Method | Training run |
| ---: | ---: |
| No TRS | [2026-10-19_05-22-27_x1_no_trs](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs) |
| Actor TRS 0.1 | [2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |

### X1_go2bend

| Method | Training run |
| ---: | ---: |
| No TRS | [2026-10-21_08-45-56_x1_go2bend_no_trs](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs) |
| Actor TRS 0.1 | [2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |

### Go2

| Method | Training run |
| ---: | ---: |
| No TRS | [2026-10-19_09-33-24_go2_no_trs](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs) |
| Actor TRS 0.1 | [2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |

</details>

- Commands: **vx = −2, −1.5, −1, −0.5, +0.5, +1, +1.5, +2 m/s**. All have **vy = 0** and **yaw-rate command = 0**.
- Eight trained templates: trot, bound, half-bound A/B, rotary A/B, transverse A/B. A/B identify command templates; they do not assert that the observed motion has that gait identity.
- **100 parallel nominal replicas per gait/command**, giving 6,400 trials per policy, 12,800 per robot, and **38,400 total**. There are 1,600 planned trials per method per ±speed pair and 100 per individual gait/direction.
- Control step: **0.02 s**. Settling: **250 steps / 5 s**. Measured interval: **1,000 steps / 20 s**. Full rollout: **1,250 steps / 25 s**.
- Complete-cycle metric windows: **960 samples / 19.2 s / 32 cycles** for both X1 configurations; **988 samples / 19.76 s / 38 cycles** for Go2.
- Fixed cycle duration: **0.60 s** for X1 and X1_go2bend; **0.52 s** for Go2. Commanded stance duty factor: **0.56** for all robots.
- TRS uses actor coefficient **0.1**, value coefficient **0**, and warmup **0**. The run-name field `mirror0p1` refers to this actor TRS coefficient in this implementation.
- The earlier ±1 study and the new six-speed study use the same checkpoint, saved configuration provenance, runtime overrides, gait library, nominal profile, and timing within each policy. Recorded source changes between those studies are confined to evaluation folder-layout handling.
- One training seed per method; evaluation replicas are nominal initializations, not independent training seeds or randomized robustness trials.

## Metric definitions

The equations below define the reported quantities. Unless a formula explicitly sums over trials, it describes **one trial**; the trial index is omitted for readability.

Jump to [notation](#notation-and-averaging), [velocity and heading](#velocity-and-heading), [contact metrics](#contact-timing-and-execution), [effort and balance](#effort-and-load-balance), [classification and success](#classification-completion-and-success), or [forward/backward consistency](#forwardbackward-consistency).

### Notation and averaging

| Symbol | Meaning |
| --- | --- |
| $t=1,\ldots,T$ | Samples in the complete-cycle measurement window; $\Delta t=0.02$ s. |
| $\langle z\rangle_t=\frac{1}{T}\sum_{t=1}^{T}z_t$ | Arithmetic time average within one trial. |
| $\ell=1,\ldots,L$; $j=1,\ldots,J$ | Feet and joints; $L=4$, $J=12$. |
| $u_x,u_y,u_\omega$ | Constant commanded body-frame velocities [m/s, m/s, rad/s]; $u_y=u_\omega=0$ here. |
| $v^b_{x,t},v^b_{y,t},\omega_{z,t}$ | Measured body-frame linear velocities [m/s] and yaw rate [rad/s]. |
| $\psi_t,x_t^w$ | Heading [rad] and world-x position [m]. |
| $P,K,\delta$ | Commanded cycle duration [s], complete-cycle count, and commanded stance fraction $\delta=0.56$. |
| $\mathbf{1}[\cdot]$ | Indicator: one when the condition holds, zero otherwise. |
| $\mathcal P,N$ | Planned trials in the displayed condition and their count, $N=\#\mathcal P$. |
| $C_r$ | Completion indicator for planned trial $r$, defined below. |

**Averaging across trials.** For a continuous metric $z$, let $\mathcal F_z$ contain the trials that complete and have a defined value of $z$. The displayed value is

$$
\overline z=\frac{1}{\#\mathcal F_z}\sum_{r\in\mathcal F_z}z_r.
$$

Thus, the reported speed error is the **mean of trial RMSEs**. Each trial receives equal weight; its samples are not pooled with other trials before computing RMSE. Percent-valued contact, duty, clipping, and imbalance diagnostics also use this averaging rule.

<a id="metric-matched-completion"></a>

Completion, classifier, and success rates instead retain **all $N$ planned trials** in their denominators. The matched-completion table restricts continuous metrics to trial identities that complete in both methods. An empty eligible set yields N/A.

### Velocity and heading

<a id="metric-velocity-rmse"></a>

**vx and vy RMSE [m/s] ↓.** These measure body-frame speed-tracking error:

$$
E_x=\sqrt{\left\langle(v^b_{x,t}-u_x)^2\right\rangle_t},
\qquad
E_y=\sqrt{\left\langle(v^b_{y,t}-u_y)^2\right\rangle_t}.
$$

The +vx and −vx columns apply the same definition to the forward and backward trial groups separately.

<a id="metric-relative-velocity-rmse"></a>

**Relative vx RMSE [%] ↓.** This expresses speed error as a percentage of commanded speed:

$$
R_x=100\,\frac{E_x}{\lvert u_x\rvert}.
$$

All tested commands are nonzero. The displayed relative error uses this denominator; the velocity-success check uses the guarded ratio defined under success below.

<a id="metric-yaw-rate-rmse"></a>

**Yaw-rate RMSE [rad/s] ↓.** This measures rotational velocity error, rather than accumulated heading drift:

$$
E_\omega=\sqrt{\left\langle(\omega_{z,t}-u_\omega)^2\right\rangle_t}.
$$

<a id="metric-heading-rmse"></a>

**Heading RMSE [degrees] ↓.** Define the wrapped heading error relative to nominal reset heading $\psi_{\mathrm{ref}}=0$:

$$
e_{\psi,t}=\operatorname{atan2}
\bigl(\sin(\psi_t-\psi_{\mathrm{ref}}),\cos(\psi_t-\psi_{\mathrm{ref}})\bigr),
\qquad
E_\psi=\frac{180}{\pi}\sqrt{\left\langle e_{\psi,t}^{\,2}\right\rangle_t}.
$$

Wrapping keeps angular differences within $[-\pi,\pi]$ before computing RMSE.

<a id="metric-directed-progress"></a>

**Directed progress [%] ≈100.** Let $x_0^w$ be world-x position at the start boundary of the measurement window, and $x_T^w$ its final position:

$$
\Delta x^\rightarrow=\operatorname{sgn}(u_x)(x_T^w-x_0^w),
\qquad
d_{\mathrm{cmd}}=T\Delta t\,\lvert u_x\rvert,
\qquad
G=100\,\frac{\Delta x^\rightarrow}{d_{\mathrm{cmd}}}.
$$

The displacement and commanded distance are in metres. The start boundary uses the position immediately before the first measured step. A value of 100% means the requested world-direction distance was covered; negative values mean net motion in the opposite direction. Good body-frame speed tracking can coexist with poor progress if the robot turns.

### Contact timing and execution

**Contact state, commanded stance, and eligible samples.** Let $c_{t\ell}\in\{0,1\}$ denote force-detected contact and $p_{t\ell}\in[0,1)$ the archived signed foot phase. Desired stance is

$$
d_{t\ell}=\mathbf{1}[p_{t\ell}\ge 1-\delta].
$$

The archived phases follow $p_{t\ell}=\operatorname{frac}\bigl(s(\phi_t+o_\ell)\bigr)$, where $s=\operatorname{sgn}(u_x)$, $\phi_t$ is the increasing common gait clock [cycles], $o_\ell$ is the saved foot offset [cycles], and $\operatorname{frac}(a)=a-\lfloor a\rfloor$.

Define circular difference and distance in cycles:

$$
\operatorname{wrap}_1(a)=\operatorname{frac}(a+\tfrac12)-\tfrac12,
\qquad
\rho_1(a,b)=\bigl\lvert\operatorname{wrap}_1(a-b)\bigr\rvert.
$$

Samples within 0.02 cycles of either stance boundary are excluded from agreement and swing/stance errors:

$$
m_{t\ell}=
\mathbf{1}[\rho_1(p_{t\ell},0)>0.02]\,
\mathbf{1}[\rho_1(p_{t\ell},1-\delta)>0.02].
$$

<details>
<summary>Contact-force thresholds and hysteresis</summary>

The detector uses positive, ground-filtered vertical foot force $F^+_{z,t\ell}=\max(F_{z,t\ell},0)$ [N]. For robot mass $M$ [kg] and $g_0=9.80665$ m/s², the saved thresholds are

$$
F_{\mathrm{on}}=\max(10\ \mathrm{N},\,0.12Mg_0/4),
\qquad
F_{\mathrm{off}}=\max(5\ \mathrm{N},\,0.06Mg_0/4).
$$

With the saved dwell time of 0.02 s, equal to one control sample, contact is computed from an initially false state:

$$
c_{t\ell}=
\begin{cases}
1, & F^+_{z,t\ell}\ge F_{\mathrm{on}},\\
0, & F^+_{z,t\ell}\le F_{\mathrm{off}},\\
c_{t-1,\ell}, & \text{otherwise}.
\end{cases}
$$

</details>

<a id="metric-contact-agreement"></a>

**Contact agreement [%] ↑.** This counts agreement with commanded stance over eligible foot-time samples:

$$
A=100\,\frac{\sum_{t,\ell}m_{t\ell}\mathbf{1}[c_{t\ell}=d_{t\ell}]}
{\sum_{t,\ell}m_{t\ell}}.
$$

<a id="metric-swing-contact"></a>

**Contact during commanded swing [%] ↓.** Its denominator contains eligible swing samples only:

$$
E_{\mathrm{swing}}=
100\,\frac{\sum_{t,\ell}m_{t\ell}(1-d_{t\ell})c_{t\ell}}
{\sum_{t,\ell}m_{t\ell}(1-d_{t\ell})}.
$$

<a id="metric-missed-stance"></a>

**Missed commanded stance [%] ↓.** Its denominator contains eligible stance samples only:

$$
E_{\mathrm{stance}}=
100\,\frac{\sum_{t,\ell}m_{t\ell}d_{t\ell}(1-c_{t\ell})}
{\sum_{t,\ell}m_{t\ell}d_{t\ell}}.
$$

<a id="metric-touchdown-error"></a>

**Touchdown timing error [ms] ↓.** Let $\mathcal M$ contain matched observed touchdown events across all feet. For a matched event at sample $t$ on foot $\ell$, the increasing-clock foot phase is $\theta_{t\ell}=\operatorname{frac}(\phi_t+\widetilde o_\ell)$, with

$$
\widetilde o_\ell=
\begin{cases}
o_\ell, & u_x>0,\\
o_\ell+(1-\delta), & u_x<0.
\end{cases}
\qquad
E_{\mathrm{TD}}=
\frac{1000P}{\#\mathcal M}
\sum_{(t,\ell)\in\mathcal M}\rho_1(\theta_{t\ell},1-\delta).
$$

Each expected touchdown is matched to the nearest unused observed touchdown on the same foot and in the same common-clock cycle, with circular phase error at most 0.20 cycles. Errors are averaged over matched events, then converted from cycles to milliseconds using $P$. The backward offset above applies the audit correction.

Unmatched events do not enter this timing mean; repeated contacts are assessed separately. This metric measures schedule alignment, rather than synchronization between feet.

<a id="metric-measured-duty"></a>

**Measured duty [%], target 56.** First compute each foot's contact fraction using all samples, without the boundary-exclusion mask:

$$
D_\ell=\frac{1}{T}\sum_{t=1}^{T}c_{t\ell},
\qquad
D_{\mathrm{meas}}=\frac{100}{L}\sum_{\ell=1}^{L}D_\ell.
$$

<a id="metric-duty-error"></a>

**Duty error [percentage points] ↓.** Take the absolute error before averaging across feet:

$$
E_D=\frac{100}{L}\sum_{\ell=1}^{L}\lvert D_\ell-\delta\rvert.
$$

Unlike mean measured duty, this error does not allow one foot's excess stance to cancel another foot's deficit.

<a id="metric-touchdowns-per-cycle"></a>

**Max-foot touchdowns/cycle ≈1.** Count rising contact transitions within the measurement window:

$$
n_\ell^{\mathrm{TD}}=
\sum_{t=2}^{T}\mathbf{1}[c_{t-1,\ell}=0,\ c_{t\ell}=1],
\qquad
R_{\mathrm{TD}}=\frac{\max_\ell n_\ell^{\mathrm{TD}}}{K}.
$$

The table reports the mean of this per-trial ratio, rather than the maximum over all trials. Small deviations below one can reflect event-window boundaries. Liftoff counts $n_\ell^{\mathrm{LO}}$, used in the success rules, count the opposite transition, $1\rightarrow0$.

### Effort and load balance

Let $\tau_{tj}$ be applied joint torque [N·m], $\dot q_{tj}$ joint velocity [rad/s], and $\tau_j^{\mathrm{lim}}$ the configured effort limit [N·m].

<a id="metric-mechanical-power"></a>

**Mechanical power [W] ↓\*.** Sum absolute mechanical power over joints, then average over time:

$$
P_{\mathrm{mech}}=
\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{J}
\lvert\tau_{tj}\dot q_{tj}\rvert.
$$

Positive and negative work both contribute. This measures mechanical effort, rather than electrical consumption.

<a id="metric-normalized-torque-cost"></a>

**Normalized torque cost [dimensionless] ↓\*.** This sums squared torque utilization across joints:

$$
Q_\tau=
\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{J}
\left(\frac{\tau_{tj}}{\tau_j^{\mathrm{lim}}}\right)^2.
$$

Both effort metrics should be interpreted alongside tracking, gait execution, completion, and displacement.

<a id="metric-action-clipping"></a>

**Action clipping [%] ↓.** Let $\chi_{tj}=1$ when the target-constraint logic clips joint $j$'s action at sample $t$:

$$
A_{\mathrm{clip}}=
\frac{100}{TJ}\sum_{t=1}^{T}\sum_{j=1}^{J}\chi_{tj}.
$$

The denominator counts joint-time commands. Lower clipping indicates fewer interventions; it does not establish smooth motion.

<a id="metric-front-rear-imbalance"></a>

**Front/rear imbalance [%] ↓†.** Let $\mathcal J_\ell$ contain the joints belonging to foot $\ell$. The accumulated quantities per leg are

$$
\begin{aligned}
W_\ell &=
\Delta t\sum_{t=1}^{T}\sum_{j\in\mathcal J_\ell}
\lvert\tau_{tj}\dot q_{tj}\rvert
&&[\mathrm{J}],\\
I_\ell &=
\Delta t\sum_{t=1}^{T}F^+_{z,t\ell}
&&[\mathrm{N\,s}],\\
Q_\ell &=
\Delta t\sum_{t=1}^{T}\sum_{j\in\mathcal J_\ell}
\left(\frac{\tau_{tj}}{\tau_j^{\mathrm{lim}}}\right)^2
&&[\mathrm{s}].
\end{aligned}
$$

For any one of these quantities $Z$, sum the front and rear legs separately:

$$
Z_F=Z_{\mathrm{FL}}+Z_{\mathrm{FR}},
\qquad
Z_R=Z_{\mathrm{RL}}+Z_{\mathrm{RR}},
\qquad
\mathcal I(Z)=100\,\frac{\lvert Z_F-Z_R\rvert}{Z_F+Z_R+\epsilon_Z}.
$$

This defines the three reported imbalance metrics:

$$
I_{\mathrm{work}}=\mathcal I(W),
\qquad
I_{\mathrm{force}}=\mathcal I(I),
\qquad
I_{\mathrm{torque}}=\mathcal I(Q).
$$

The saved numerical guard $\epsilon_Z$ is $10^{-12}$ in the native units of the accumulated quantity. Absolute imbalance is computed within each trial before averaging, so front-heavy and rear-heavy trials cannot cancel. More even loading is not universally optimal for every gait.

### Classification, completion, and success

<a id="metric-completion"></a>

**Completion [%] ↑.** A completed trial records all 250 settling steps plus 1,000 measurement steps without episode termination:

$$
C_r=\mathbf{1}[\text{all 1,250 steps recorded without episode termination}],
\qquad
S_{\mathrm{complete}}=\frac{100}{N}\sum_{r\in\mathcal P}C_r.
$$

All rates below use the same all-planned denominator. A failed trial contributes zero to the numerator.

<a id="metric-termination-counts"></a>

Termination counts sum $\mathbf{1}[\text{episode terminated}]$ over all planned trials, including terminations during settling. In this finished study, every trial either completes or terminates early, so the total is $N-\sum_r C_r$.

<a id="metric-gait-classification"></a>

**Exact template and correct family [%] ↑.** Let $g_r,f_r$ be the requested template and family, and $\widehat g_r,\widehat f_r$ their classifier estimates:

$$
S_{\mathrm{exact}}=
\frac{100}{N}\sum_{r\in\mathcal P}C_r\,\mathbf{1}[\widehat g_r=g_r],
\qquad
S_{\mathrm{family}}=
\frac{100}{N}\sum_{r\in\mathcal P}C_r\,\mathbf{1}[\widehat f_r=f_r].
$$

The classifier compares all six pairwise touchdown-phase differences. It accepts a template only when the best mean circular error is at most 0.12 cycles and the next-best score is at least 0.01 cycles worse. Insufficient cycle/event coverage or an ambiguous match yields an unclassified result.

<details>
<summary>Mathematical definition of the gait classifier</summary>

For each foot, let $\mathcal T_\ell$ contain all observed touchdown sample indices. Its circular mean phase in the common gait clock is

$$
\mu_\ell=
\operatorname{frac}\left[
\frac{1}{2\pi}\arg\left(
\sum_{t\in\mathcal T_\ell}e^{\,\mathrm{i}2\pi\phi_t}
\right)\right].
$$

For candidate template $g$ with offsets $o_{g\ell}$, define $\beta_{g\ell}=\operatorname{frac}(-o_{g\ell})$. A shared touchdown-phase shift cancels in pairwise differences. The template score is

$$
s_g=\frac{1}{6}\sum_{\ell<k}
\left\lvert
\operatorname{wrap}_1\left[
(\mu_k-\mu_\ell)-(\beta_{gk}-\beta_{g\ell})
\right]\right\rvert.
$$

Let $s_{(1)}$ and $s_{(2)}$ be the smallest and second-smallest candidate scores. The best template is accepted when

$$
s_{(1)}\le0.12,
\qquad
s_{(2)}-s_{(1)}\ge0.01,
$$

with scores measured in cycles. Classification additionally requires at least two complete cycles and at least two touchdown and two liftoff events on every foot. The estimated family is the accepted template's family.

</details>

<a id="metric-velocity-success"></a>

**Velocity success [%] ↑.** The success predicate uses the dimensionless, guarded relative speed error

$$
\widetilde R_x=\frac{E_x}{\lvert u_x\rvert+\epsilon_v},
\qquad
\epsilon_v=10^{-6}\ \mathrm{m/s}.
$$

For a completed trial, define

$$
V_r=
\mathbf{1}[\widetilde R_{x,r}\le0.20]\,
\mathbf{1}[E_{\omega,r}\le0.05\ \mathrm{rad/s}]\,
\mathbf{1}[\Delta x^\rightarrow_r>0].
$$

<a id="metric-gait-success"></a>

**Gait success [%] ↑.** For a completed trial, define

$$
H_r=
\mathbf{1}[A_r/100\ge0.80]\,
\mathbf{1}[\widehat f_r=f_r]\,
\mathbf{1}[K_r\ge2]\,
\prod_{\ell=1}^{L}
\left(
\mathbf{1}[n_{r\ell}^{\mathrm{TD}}\ge2]\,
\mathbf{1}[n_{r\ell}^{\mathrm{LO}}\ge2]
\right).
$$

Here $A_r$ is on the displayed percent scale, so $A_r/100$ is the agreement fraction. For incomplete trials, set $V_r=H_r=0$ without evaluating missing measurements.

<a id="metric-combined-success"></a>

**Velocity, gait, and combined rates** then follow as

$$
S_{\mathrm{velocity}}=\frac{100}{N}\sum_{r\in\mathcal P}C_rV_r,
\qquad
S_{\mathrm{gait}}=\frac{100}{N}\sum_{r\in\mathcal P}C_rH_r,
\qquad
S_{\mathrm{combined}}=\frac{100}{N}\sum_{r\in\mathcal P}C_rV_rH_r.
$$

<a id="metric-positive-progress"></a>

**Positive-progress trials [%] ↑.** This requires completion and positive directed displacement, without the other velocity or gait checks:

$$
S_{\mathrm{positive}}=
\frac{100}{N}\sum_{r\in\mathcal P}
C_r\,\mathbf{1}[\Delta x^\rightarrow_r>0].
$$

### Forward/backward consistency

<a id="metric-tr-partner-speed-bias"></a>

**TR-partner speed bias [m/s] ↓.** Let $\mathrm{TR}(g)$ be the saved time-reversed partner of requested template $g$, and let $k$ identify a nominal replica. For speed magnitude $v>0$, let $\mathcal B_v$ contain the pairs $(g,k)$ whose forward and backward partner trials both complete under **both** methods.

For method $m$, the mean body-frame speed in a trial and the reversal bias are

$$
\overline v^{\,b}_{x,m,g,u,k}
=\frac{1}{T_{m,g,u,k}}\sum_t v^b_{x,m,g,u,k,t},
$$

$$
B_m(v)=
\frac{1}{\#\mathcal B_v}
\sum_{(g,k)\in\mathcal B_v}
\left\lvert
\overline v^{\,b}_{x,m,g,+v,k}
+
\overline v^{\,b}_{x,m,\mathrm{TR}(g),-v,k}
\right\rvert.
$$

<a id="metric-matched-directional-pairs"></a>

The matched-pair count is $\#\mathcal B_v$, shared by both methods. Ideal speed reversal gives equal magnitudes with opposite signs, hence zero bias. This metric does not establish full trajectory or action symmetry.

These definitions follow the [metric calculations](scripts/symm_locomotion/_leg_usage_metrics.py) and [evaluation-window and displacement handling](scripts/symm_locomotion/evaluation.py), with the signed-phase, backward-touchdown, and heading-reference corrections described below.

## Corrections and limitations

The audit corrected derived comparisons offline; it did not change simulator code, original archives, or original evaluation CSVs. Therefore backward contact agreement and heading values here can differ from the original reports.

1. **Backward phase scoring:** the common evaluation clock increases even when native per-foot phases reverse. Contact agreement and swing/stance errors here use archived signed foot phases. For fixed duty, backward touchdown timing uses the equivalent increasing-clock offset shifted by `1 − duty`. Recorded phases were checked against the signed analytic formula.
2. **Heading reference:** a stale lazy pose immediately after reset could retain the previous scenario's final heading. This report evaluates wrapped heading against the known nominal reset heading of zero. Constant position-reference offsets do not affect displacement differences within the measurement interval.
3. **Termination accounting:** runtime trial status and archived episode-done flags were checked, including failures before settling completed. An absent continuous metric does not mean that a trial succeeded.
4. **Cohort differences:** continuous means exclude partial/terminated trials, particularly affecting ±2 comparisons. Matched-completion results and all-trial completion/success rates are supplied separately. There is no universally valid single ranking from survivor-only means.
5. **Scope:** one trained policy per method per robot, one training seed, and nominal replicas. These results do not estimate variation across training seeds or establish statistically significant population-level benefits. No confidence interval treating 100 nominal replicas as 100 independent training runs is reported.
6. **Interpretation:** touchdown schedule error is not leg-pair synchronization; action clipping is not smoothness; even front/rear loading is not universally optimal; absolute mechanical power is not electrical efficiency. Gait classification and agreement measure different aspects of execution.
7. **Unmeasured conditions:** the report does not test nonzero lateral/yaw commands, random pushes, terrain robustness, or training convergence speed. Abrupt-motion claims are not inferred from these summary diagnostics.

## Appendix A: each gait, pooled over both directions

Expand a robot below to see all four speeds: [X1](#x1-gait-level-paired-results) · [X1_go2bend](#x1_go2bend-gait-level-paired-results) · [Go2](#go2-gait-level-paired-results).

Each row has 200 planned trials per method. Continuous metrics average completed trials from both signs; percentages retain all 200 trials in their denominator. A/B labels denote the saved command templates. The naming map is `gallop_a/c = Rotary A/B`, `gallop_b/d = Transverse A/B`, and `half_bound_front_a/b = Half-bound A/B`.

### X1: gait-level paired results

<details>
<summary>Show X1 gait tables at all four speeds</summary>

**vx = ±0.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | <span style="color: #c00000;">**0.048**</span> → 0.054 | <span style="color: #c00000;">**0.046**</span> → 0.064 | 96.4 → <span style="color: #c00000;">**98.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.058 → <span style="color: #c00000;">**0.037**</span> | 0.094 → <span style="color: #c00000;">**0.046**</span> | 96.7 → <span style="color: #c00000;">**99.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.037**</span> → 0.048 | 0.077 → <span style="color: #c00000;">**0.044**</span> | 96.0 → <span style="color: #c00000;">**96.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | 0.046 → 0.046 | 0.081 → <span style="color: #c00000;">**0.046**</span> | 95.1 → <span style="color: #c00000;">**97.1**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | <span style="color: #c00000;">**0.033**</span> → 0.046 | 0.094 → <span style="color: #c00000;">**0.080**</span> | 86.6 → <span style="color: #c00000;">**88.8**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | <span style="color: #c00000;">**0.043**</span> → 0.054 | 0.078 → <span style="color: #c00000;">**0.068**</span> | <span style="color: #c00000;">**89.9**</span> → 88.8 | <span style="color: #c00000;">**49.0**</span> → 0.0 | <span style="color: #c00000;">**49.0**</span> → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.066 → <span style="color: #c00000;">**0.053**</span> | 0.051 → <span style="color: #c00000;">**0.046**</span> | <span style="color: #c00000;">**99.1**</span> → 98.7 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.057 → <span style="color: #c00000;">**0.053**</span> | 0.058 → <span style="color: #c00000;">**0.051**</span> | 96.9 → <span style="color: #c00000;">**98.7**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.071 → <span style="color: #c00000;">**0.067**</span> | <span style="color: #c00000;">**0.051**</span> → 0.072 | 95.1 → <span style="color: #c00000;">**96.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.092 → <span style="color: #c00000;">**0.081**</span> | 0.094 → <span style="color: #c00000;">**0.050**</span> | 95.5 → <span style="color: #c00000;">**96.2**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.064**</span> → 0.069 | 0.076 → <span style="color: #c00000;">**0.063**</span> | <span style="color: #c00000;">**95.1**</span> → 93.2 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | 0.122 → <span style="color: #c00000;">**0.086**</span> | 0.096 → <span style="color: #c00000;">**0.056**</span> | 94.0 → <span style="color: #c00000;">**96.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | <span style="color: #c00000;">**0.077**</span> → 0.080 | 0.092 → <span style="color: #c00000;">**0.074**</span> | 81.0 → <span style="color: #c00000;">**86.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | 0.091 → <span style="color: #c00000;">**0.074**</span> | 0.104 → <span style="color: #c00000;">**0.069**</span> | <span style="color: #c00000;">**86.0**</span> → 84.4 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.079 → <span style="color: #c00000;">**0.071**</span> | <span style="color: #c00000;">**0.053**</span> → 0.070 | <span style="color: #c00000;">**97.7**</span> → 96.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.086 → <span style="color: #c00000;">**0.070**</span> | 0.069 → <span style="color: #c00000;">**0.062**</span> | 95.1 → <span style="color: #c00000;">**97.1**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.069 → <span style="color: #c00000;">**0.063**</span> | 0.079 → <span style="color: #c00000;">**0.070**</span> | 92.0 → <span style="color: #c00000;">**92.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.136 → <span style="color: #c00000;">**0.098**</span> | <span style="color: #c00000;">**0.101**</span> → 0.106 | 88.2 → <span style="color: #c00000;">**89.4**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.088**</span> → 0.102 | 0.105 → <span style="color: #c00000;">**0.095**</span> | <span style="color: #c00000;">**90.7**</span> → 87.8 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound B | 0.195 → <span style="color: #c00000;">**0.084**</span> | 0.096 → 0.096 | 87.6 → <span style="color: #c00000;">**91.5**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Rotary A | <span style="color: #c00000;">**0.069**</span> → 0.075 | 0.101 → <span style="color: #c00000;">**0.073**</span> | 77.1 → <span style="color: #c00000;">**80.9**</span> | 0.0 → 0.0 | <span style="color: #c00000;">**21.5**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B | 0.114 → <span style="color: #c00000;">**0.077**</span> | <span style="color: #c00000;">**0.120**</span> → 0.128 | <span style="color: #c00000;">**81.2**</span> → 74.2 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**98.0**</span> | 100.0 → 100.0 |
| Transverse A | <span style="color: #c00000;">**0.067**</span> → 0.070 | <span style="color: #c00000;">**0.092**</span> → 0.093 | <span style="color: #c00000;">**93.5**</span> → 90.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.130 → <span style="color: #c00000;">**0.068**</span> | 0.105 → <span style="color: #c00000;">**0.078**</span> | 90.2 → <span style="color: #c00000;">**94.1**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | <span style="color: #c00000;">**0.130**</span> → 0.170 | <span style="color: #c00000;">**0.129**</span> → 0.188 | 88.0 → <span style="color: #c00000;">**89.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.234 → <span style="color: #c00000;">**0.220**</span> | <span style="color: #c00000;">**0.140**</span> → 0.163 | <span style="color: #c00000;">**81.0**</span> → 80.6 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.201 → <span style="color: #c00000;">**0.133**</span> | 0.144 → <span style="color: #c00000;">**0.122**</span> | <span style="color: #c00000;">**87.2**</span> → 78.6 | <span style="color: #c00000;">**32.0**</span> → 0.0 | <span style="color: #c00000;">**32.0**</span> → 0.0 | 54.5 → <span style="color: #c00000;">**99.5**</span> |
| Half-bound B | 0.181 → <span style="color: #c00000;">**0.150**</span> | 0.264 → <span style="color: #c00000;">**0.123**</span> | 72.6 → <span style="color: #c00000;">**84.2**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary A | 0.219 → <span style="color: #c00000;">**0.146**</span> | 0.164 → <span style="color: #c00000;">**0.124**</span> | 70.6 → <span style="color: #c00000;">**73.6**</span> | 0.0 → 0.0 | 36.5 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary B | <span style="color: #c00000;">**0.132**</span> → 0.157 | <span style="color: #c00000;">**0.153**</span> → 0.213 | <span style="color: #c00000;">**75.0**</span> → 69.5 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**98.0**</span> | 100.0 → 100.0 |
| Transverse A | 0.219 → <span style="color: #c00000;">**0.104**</span> | 0.115 → <span style="color: #c00000;">**0.105**</span> | <span style="color: #c00000;">**89.2**</span> → 85.2 | <span style="color: #c00000;">**92.5**</span> → 50.0 | <span style="color: #c00000;">**92.5**</span> → 50.0 | 100.0 → 100.0 |
| Transverse B | <span style="color: #c00000;">**0.139**</span> → 0.201 | 0.139 → <span style="color: #c00000;">**0.112**</span> | 80.8 → <span style="color: #c00000;">**88.6**</span> | <span style="color: #c00000;">**99.0**</span> → 97.0 | <span style="color: #c00000;">**99.0**</span> → 97.0 | 100.0 → 100.0 |

</details>

### X1_go2bend: gait-level paired results

<details>
<summary>Show X1_go2bend gait tables at all four speeds</summary>

**vx = ±0.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | <span style="color: #c00000;">**0.027**</span> → 0.035 | <span style="color: #c00000;">**0.042**</span> → 0.062 | 96.9 → <span style="color: #c00000;">**99.1**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.100 → <span style="color: #c00000;">**0.081**</span> | 0.109 → <span style="color: #c00000;">**0.060**</span> | 91.4 → <span style="color: #c00000;">**91.7**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.071**</span> → 0.088 | 0.100 → <span style="color: #c00000;">**0.073**</span> | 95.8 → <span style="color: #c00000;">**97.3**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B | 0.061 → <span style="color: #c00000;">**0.048**</span> | 0.092 → <span style="color: #c00000;">**0.049**</span> | 91.5 → <span style="color: #c00000;">**96.9**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary A | 0.057 → <span style="color: #c00000;">**0.048**</span> | <span style="color: #c00000;">**0.083**</span> → 0.095 | 78.1 → <span style="color: #c00000;">**86.8**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | <span style="color: #c00000;">**0.052**</span> → 0.103 | 0.108 → <span style="color: #c00000;">**0.088**</span> | <span style="color: #c00000;">**87.5**</span> → 86.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | <span style="color: #c00000;">**0.052**</span> → 0.078 | 0.094 → <span style="color: #c00000;">**0.058**</span> | 96.9 → <span style="color: #c00000;">**98.7**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | <span style="color: #c00000;">**0.051**</span> → 0.056 | 0.085 → <span style="color: #c00000;">**0.045**</span> | 97.3 → <span style="color: #c00000;">**98.2**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.059 → <span style="color: #c00000;">**0.053**</span> | <span style="color: #c00000;">**0.061**</span> → 0.063 | 96.4 → <span style="color: #c00000;">**96.9**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.125 → <span style="color: #c00000;">**0.109**</span> | 0.146 → <span style="color: #c00000;">**0.090**</span> | 88.4 → <span style="color: #c00000;">**89.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.076**</span> → 0.093 | <span style="color: #c00000;">**0.071**</span> → 0.107 | 93.8 → <span style="color: #c00000;">**95.5**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Half-bound B | <span style="color: #c00000;">**0.074**</span> → 0.084 | 0.112 → <span style="color: #c00000;">**0.078**</span> | 89.9 → <span style="color: #c00000;">**94.2**</span> | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary A | <span style="color: #c00000;">**0.069**</span> → 0.079 | <span style="color: #c00000;">**0.100**</span> → 0.115 | 74.1 → <span style="color: #c00000;">**81.2**</span> | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B | <span style="color: #c00000;">**0.075**</span> → 0.131 | <span style="color: #c00000;">**0.087**</span> → 0.134 | 83.3 → <span style="color: #c00000;">**84.6**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | <span style="color: #c00000;">**0.057**</span> → 0.091 | 0.094 → <span style="color: #c00000;">**0.057**</span> | 95.5 → <span style="color: #c00000;">**96.9**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | <span style="color: #c00000;">**0.062**</span> → 0.073 | 0.079 → <span style="color: #c00000;">**0.077**</span> | <span style="color: #c00000;">**94.8**</span> → 94.6 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.116 → <span style="color: #c00000;">**0.079**</span> | 0.075 → <span style="color: #c00000;">**0.054**</span> | 93.3 → <span style="color: #c00000;">**95.1**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.118 → <span style="color: #c00000;">**0.072**</span> | 0.124 → <span style="color: #c00000;">**0.108**</span> | <span style="color: #c00000;">**82.9**</span> → 81.9 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.092 → <span style="color: #c00000;">**0.077**</span> | 0.129 → <span style="color: #c00000;">**0.090**</span> | <span style="color: #c00000;">**89.6**</span> → 88.2 | <span style="color: #c00000;">**50.0**</span> → 0.0 | <span style="color: #c00000;">**50.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound B | <span style="color: #c00000;">**0.083**</span> → 0.101 | 0.129 → <span style="color: #c00000;">**0.084**</span> | 83.7 → <span style="color: #c00000;">**89.1**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Rotary A | 0.090 → <span style="color: #c00000;">**0.079**</span> | 0.113 → <span style="color: #c00000;">**0.086**</span> | 69.8 → <span style="color: #c00000;">**75.0**</span> | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.108 → <span style="color: #c00000;">**0.096**</span> | 0.128 → <span style="color: #c00000;">**0.096**</span> | <span style="color: #c00000;">**78.8**</span> → 77.0 | 0.0 → 0.0 | <span style="color: #c00000;">**50.0**</span> → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.095 → <span style="color: #c00000;">**0.090**</span> | 0.084 → <span style="color: #c00000;">**0.061**</span> | 91.0 → <span style="color: #c00000;">**92.4**</span> | <span style="color: #c00000;">**100.0**</span> → 50.0 | <span style="color: #c00000;">**100.0**</span> → 50.0 | 100.0 → 100.0 |
| Transverse B | 0.098 → <span style="color: #c00000;">**0.092**</span> | 0.102 → <span style="color: #c00000;">**0.092**</span> | 90.2 → <span style="color: #c00000;">**91.1**</span> | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.223 → <span style="color: #c00000;">**0.161**</span> | 0.093 → <span style="color: #c00000;">**0.071**</span> | 89.5 → <span style="color: #c00000;">**91.8**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.226 → <span style="color: #c00000;">**0.155**</span> | 0.143 → <span style="color: #c00000;">**0.138**</span> | <span style="color: #c00000;">**75.7**</span> → 74.6 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.164**</span> → 0.169 | 0.124 → <span style="color: #c00000;">**0.104**</span> | <span style="color: #c00000;">**82.2**</span> → 80.7 | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound B | 0.209 → <span style="color: #c00000;">**0.171**</span> | 0.151 → <span style="color: #c00000;">**0.096**</span> | 75.2 → <span style="color: #c00000;">**80.3**</span> | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary A | 0.175 → <span style="color: #c00000;">**0.130**</span> | 0.148 → <span style="color: #c00000;">**0.090**</span> | 63.0 → <span style="color: #c00000;">**68.1**</span> | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.208 → <span style="color: #c00000;">**0.156**</span> | 0.140 → <span style="color: #c00000;">**0.124**</span> | <span style="color: #c00000;">**72.0**</span> → 68.4 | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Transverse A | <span style="color: #c00000;">**0.137**</span> → 0.149 | 0.137 → <span style="color: #c00000;">**0.118**</span> | 85.7 → <span style="color: #c00000;">**86.3**</span> | <span style="color: #c00000;">**89.0**</span> → 53.5 | <span style="color: #c00000;">**89.0**</span> → 53.5 | 100.0 → 100.0 |
| Transverse B | 0.206 → <span style="color: #c00000;">**0.146**</span> | 0.136 → <span style="color: #c00000;">**0.101**</span> | 84.6 → <span style="color: #c00000;">**85.3**</span> | <span style="color: #c00000;">**59.5**</span> → 47.0 | <span style="color: #c00000;">**59.5**</span> → 47.0 | 100.0 → 100.0 |

</details>

### Go2: gait-level paired results

<details>
<summary>Show Go2 gait tables at all four speeds</summary>

**vx = ±0.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.048 → <span style="color: #c00000;">**0.031**</span> | <span style="color: #c00000;">**0.060**</span> → 0.070 | 97.6 → <span style="color: #c00000;">**99.5**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | <span style="color: #c00000;">**0.041**</span> → 0.051 | <span style="color: #c00000;">**0.055**</span> → 0.073 | 93.2 → <span style="color: #c00000;">**94.6**</span> | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound A | <span style="color: #c00000;">**0.036**</span> → 0.058 | <span style="color: #c00000;">**0.058**</span> → 0.060 | 96.4 → <span style="color: #c00000;">**96.9**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | <span style="color: #c00000;">**0.040**</span> → 0.051 | 0.070 → <span style="color: #c00000;">**0.050**</span> | 94.3 → <span style="color: #c00000;">**95.8**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | <span style="color: #c00000;">**0.047**</span> → 0.057 | 0.067 → <span style="color: #c00000;">**0.063**</span> | 96.9 → <span style="color: #c00000;">**97.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B | 0.048 → 0.048 | 0.112 → <span style="color: #c00000;">**0.063**</span> | 96.4 → 96.4 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A | <span style="color: #c00000;">**0.051**</span> → 0.066 | <span style="color: #c00000;">**0.037**</span> → 0.075 | 96.9 → 96.9 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.041 → <span style="color: #c00000;">**0.023**</span> | <span style="color: #c00000;">**0.050**</span> → 0.068 | <span style="color: #c00000;">**95.1**</span> → 94.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.039 → <span style="color: #c00000;">**0.037**</span> | 0.082 → <span style="color: #c00000;">**0.066**</span> | 96.4 → <span style="color: #c00000;">**97.4**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.119 → <span style="color: #c00000;">**0.097**</span> | <span style="color: #c00000;">**0.065**</span> → 0.084 | <span style="color: #c00000;">**89.5**</span> → 88.0 | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound A | 0.096 → <span style="color: #c00000;">**0.065**</span> | 0.074 → <span style="color: #c00000;">**0.067**</span> | 94.7 → <span style="color: #c00000;">**96.3**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B | 0.081 → <span style="color: #c00000;">**0.076**</span> | 0.094 → <span style="color: #c00000;">**0.075**</span> | 91.9 → <span style="color: #c00000;">**94.3**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Rotary A | 0.091 → <span style="color: #c00000;">**0.078**</span> | <span style="color: #c00000;">**0.049**</span> → 0.104 | 92.2 → <span style="color: #c00000;">**94.3**</span> | <span style="color: #c00000;">**100.0**</span> → 50.0 | <span style="color: #c00000;">**100.0**</span> → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.106 → <span style="color: #c00000;">**0.055**</span> | <span style="color: #c00000;">**0.082**</span> → 0.096 | 93.4 → <span style="color: #c00000;">**94.3**</span> | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Transverse A | 0.067 → <span style="color: #c00000;">**0.059**</span> | <span style="color: #c00000;">**0.067**</span> → 0.100 | <span style="color: #c00000;">**96.3**</span> → 94.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.066 → <span style="color: #c00000;">**0.044**</span> | <span style="color: #c00000;">**0.072**</span> → 0.076 | 92.7 → <span style="color: #c00000;">**93.8**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | <span style="color: #c00000;">**0.130**</span> → 0.142 | <span style="color: #c00000;">**0.099**</span> → 0.107 | 93.0 → <span style="color: #c00000;">**94.5**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.281 → <span style="color: #c00000;">**0.136**</span> | <span style="color: #c00000;">**0.068**</span> → 0.125 | <span style="color: #c00000;">**86.4**</span> → 82.9 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.247 → <span style="color: #c00000;">**0.190**</span> | 0.096 → <span style="color: #c00000;">**0.085**</span> | <span style="color: #c00000;">**94.1**</span> → 93.2 | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B | 0.228 → <span style="color: #c00000;">**0.181**</span> | <span style="color: #c00000;">**0.119**</span> → 0.184 | 88.7 → <span style="color: #c00000;">**92.2**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary A | 0.230 → <span style="color: #c00000;">**0.162**</span> | <span style="color: #c00000;">**0.100**</span> → 0.158 | 87.8 → <span style="color: #c00000;">**88.8**</span> | <span style="color: #c00000;">**100.0**</span> → 50.0 | <span style="color: #c00000;">**100.0**</span> → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.220 → <span style="color: #c00000;">**0.176**</span> | 0.127 → <span style="color: #c00000;">**0.104**</span> | 85.6 → <span style="color: #c00000;">**92.2**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 0.0 → <span style="color: #c00000;">**50.0**</span> | 100.0 → 100.0 |
| Transverse A | 0.212 → <span style="color: #c00000;">**0.180**</span> | <span style="color: #c00000;">**0.098**</span> → 0.124 | <span style="color: #c00000;">**93.7**</span> → 91.7 | <span style="color: #c00000;">**78.0**</span> → 50.0 | <span style="color: #c00000;">**78.0**</span> → 50.0 | 100.0 → 100.0 |
| Transverse B | 0.204 → <span style="color: #c00000;">**0.202**</span> | 0.112 → <span style="color: #c00000;">**0.099**</span> | 91.3 → <span style="color: #c00000;">**91.6**</span> | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [Yaw-rate RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Agreement](#metric-contact-agreement) [%] ↑ | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | <span style="color: #c00000;">**0.439**</span> → 0.491 | 0.133 → <span style="color: #c00000;">**0.111**</span> | <span style="color: #c00000;">**93.6**</span> → 93.1 | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> |
| Bound | 0.609 → <span style="color: #c00000;">**0.437**</span> | <span style="color: #c00000;">**0.107**</span> → 0.171 | <span style="color: #c00000;">**83.4**</span> → 79.6 | 0.0 → <span style="color: #c00000;">**48.5**</span> | 0.0 → <span style="color: #c00000;">**48.5**</span> | 100.0 → 100.0 |
| Half-bound A | 0.585 → <span style="color: #c00000;">**0.532**</span> | <span style="color: #c00000;">**0.130**</span> → 0.159 | 89.3 → <span style="color: #c00000;">**90.5**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B | 0.551 → <span style="color: #c00000;">**0.441**</span> | <span style="color: #c00000;">**0.140**</span> → 0.217 | 83.1 → <span style="color: #c00000;">**88.4**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 50.0 → <span style="color: #c00000;">**100.0**</span> |
| Rotary A | 0.621 → <span style="color: #c00000;">**0.439**</span> | <span style="color: #c00000;">**0.126**</span> → 0.167 | <span style="color: #c00000;">**86.4**</span> → 85.4 | <span style="color: #c00000;">**100.0**</span> → 17.5 | <span style="color: #c00000;">**100.0**</span> → 17.5 | 100.0 → 100.0 |
| Rotary B | 0.566 → <span style="color: #c00000;">**0.520**</span> | <span style="color: #c00000;">**0.123**</span> → 0.149 | 80.4 → <span style="color: #c00000;">**89.1**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.556 → <span style="color: #c00000;">**0.501**</span> | 0.145 → <span style="color: #c00000;">**0.127**</span> | <span style="color: #c00000;">**91.9**</span> → 89.5 | 44.0 → <span style="color: #c00000;">**50.0**</span> | 44.0 → <span style="color: #c00000;">**50.0**</span> | 97.5 → <span style="color: #c00000;">**100.0**</span> |
| Transverse B | <span style="color: #c00000;">**0.508**</span> → 0.588 | 0.173 → <span style="color: #c00000;">**0.122**</span> | 87.1 → <span style="color: #c00000;">**92.5**</span> | 55.0 → <span style="color: #c00000;">**99.5**</span> | 55.0 → <span style="color: #c00000;">**99.5**</span> | 55.0 → <span style="color: #c00000;">**99.5**</span> |

</details>

## Appendix B: all metrics for every gait and direction

Choose a robot and speed, then expand its tables. Each section contains tracking, contact, effort, and success metrics.

### X1: appendix links

| Command pair | Full gait/direction details |
| --- | --- |
| ±0.5 m/s | [Show](#detail-x1-0p5) |
| ±1 m/s | [Show](#detail-x1-1) |
| ±1.5 m/s | [Show](#detail-x1-1p5) |
| ±2 m/s | [Show](#detail-x1-2) |

### X1_go2bend: appendix links

| Command pair | Full gait/direction details |
| --- | --- |
| ±0.5 m/s | [Show](#detail-x1-go2bend-0p5) |
| ±1 m/s | [Show](#detail-x1-go2bend-1) |
| ±1.5 m/s | [Show](#detail-x1-go2bend-1p5) |
| ±2 m/s | [Show](#detail-x1-go2bend-2) |

### Go2: appendix links

| Command pair | Full gait/direction details |
| --- | --- |
| ±0.5 m/s | [Show](#detail-go2-0p5) |
| ±1 m/s | [Show](#detail-go2-1) |
| ±1.5 m/s | [Show](#detail-go2-1p5) |
| ±2 m/s | [Show](#detail-go2-2) |

Every condition below contains **100 planned trials per method**. All entries remain No TRS → TRS. Tables separate metric groups to keep each comparison readable. `N/A` is retained for continuous metrics when no full trial completes; success/completion still use all 100 planned trials.

### X1: full gait/direction details

<a id="detail-x1-0p5"></a>

#### X1, vx = ±0.5 m/s

<details>
<summary>Show all metrics for X1 at ±0.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | <span style="color: #c00000;">**0.051**</span> → 0.068 | <span style="color: #c00000;">**0.025**</span> → 0.033 | <span style="color: #c00000;">**10.18**</span> → 13.60 | <span style="color: #c00000;">**0.035**</span> → 0.069 | <span style="color: #c00000;">**19.1**</span> → 42.2 | <span style="color: #c00000;">**87.3**</span> → 62.6 |
| Trot / -0.5 | 0.046 → <span style="color: #c00000;">**0.040**</span> | 0.021 → 0.021 | 9.22 → <span style="color: #c00000;">**8.02**</span> | 0.058 → 0.058 | 44.4 → <span style="color: #c00000;">**9.8**</span> | 66.8 → <span style="color: #c00000;">**93.0**</span> |
| Bound / +0.5 | 0.045 → <span style="color: #c00000;">**0.033**</span> | 0.039 → <span style="color: #c00000;">**0.009**</span> | 8.90 → <span style="color: #c00000;">**6.55**</span> | 0.139 → <span style="color: #c00000;">**0.043**</span> | 126.1 → <span style="color: #c00000;">**5.1**</span> | -33.1 → <span style="color: #c00000;">**91.2**</span> |
| Bound / -0.5 | 0.072 → <span style="color: #c00000;">**0.041**</span> | 0.036 → <span style="color: #c00000;">**0.012**</span> | 14.36 → <span style="color: #c00000;">**8.14**</span> | <span style="color: #c00000;">**0.049**</span> → 0.050 | 66.7 → <span style="color: #c00000;">**15.2**</span> | 41.2 → <span style="color: #c00000;">**85.3**</span> |
| Half-bound A / +0.5 | <span style="color: #c00000;">**0.047**</span> → 0.054 | 0.043 → <span style="color: #c00000;">**0.025**</span> | <span style="color: #c00000;">**9.47**</span> → 10.83 | 0.074 → <span style="color: #c00000;">**0.041**</span> | 46.7 → <span style="color: #c00000;">**43.5**</span> | 66.9 → <span style="color: #c00000;">**68.4**</span> |
| Half-bound A / -0.5 | <span style="color: #c00000;">**0.027**</span> → 0.041 | 0.035 → <span style="color: #c00000;">**0.032**</span> | <span style="color: #c00000;">**5.41**</span> → 8.22 | 0.079 → <span style="color: #c00000;">**0.048**</span> | 50.3 → <span style="color: #c00000;">**26.9**</span> | 64.8 → <span style="color: #c00000;">**89.6**</span> |
| Half-bound B / +0.5 | <span style="color: #c00000;">**0.028**</span> → 0.054 | <span style="color: #c00000;">**0.025**</span> → 0.031 | <span style="color: #c00000;">**5.63**</span> → 10.79 | 0.070 → <span style="color: #c00000;">**0.048**</span> | 68.4 → <span style="color: #c00000;">**7.5**</span> | 41.7 → <span style="color: #c00000;">**86.1**</span> |
| Half-bound B / -0.5 | 0.063 → <span style="color: #c00000;">**0.038**</span> | 0.046 → 0.046 | 12.60 → <span style="color: #c00000;">**7.52**</span> | 0.092 → <span style="color: #c00000;">**0.044**</span> | 73.2 → <span style="color: #c00000;">**18.4**</span> | 33.9 → <span style="color: #c00000;">**86.6**</span> |
| Rotary A / +0.5 | <span style="color: #c00000;">**0.024**</span> → 0.059 | <span style="color: #c00000;">**0.044**</span> → 0.051 | <span style="color: #c00000;">**4.76**</span> → 11.86 | 0.101 → <span style="color: #c00000;">**0.089**</span> | 104.6 → <span style="color: #c00000;">**21.1**</span> | -9.1 → <span style="color: #c00000;">**80.3**</span> |
| Rotary A / -0.5 | 0.042 → <span style="color: #c00000;">**0.033**</span> | <span style="color: #c00000;">**0.051**</span> → 0.059 | 8.50 → <span style="color: #c00000;">**6.65**</span> | 0.086 → <span style="color: #c00000;">**0.072**</span> | 97.1 → <span style="color: #c00000;">**6.1**</span> | 1.2 → <span style="color: #c00000;">**91.3**</span> |
| Rotary B / +0.5 | <span style="color: #c00000;">**0.044**</span> → 0.048 | 0.081 → <span style="color: #c00000;">**0.067**</span> | <span style="color: #c00000;">**8.71**</span> → 9.68 | 0.091 → <span style="color: #c00000;">**0.055**</span> | 12.8 → <span style="color: #c00000;">**3.4**</span> | <span style="color: #c00000;">**96.3**</span> → 92.2 |
| Rotary B / -0.5 | <span style="color: #c00000;">**0.043**</span> → 0.059 | <span style="color: #c00000;">**0.056**</span> → 0.060 | <span style="color: #c00000;">**8.54**</span> → 11.76 | <span style="color: #c00000;">**0.065**</span> → 0.081 | <span style="color: #c00000;">**4.7**</span> → 51.8 | <span style="color: #c00000;">**92.4**</span> → 66.2 |
| Transverse A / +0.5 | 0.080 → <span style="color: #c00000;">**0.064**</span> | <span style="color: #c00000;">**0.015**</span> → 0.016 | 15.99 → <span style="color: #c00000;">**12.82**</span> | 0.045 → <span style="color: #c00000;">**0.029**</span> | 34.1 → <span style="color: #c00000;">**10.2**</span> | 71.1 → <span style="color: #c00000;">**90.3**</span> |
| Transverse A / -0.5 | 0.053 → <span style="color: #c00000;">**0.042**</span> | <span style="color: #c00000;">**0.029**</span> → 0.034 | 10.60 → <span style="color: #c00000;">**8.44**</span> | <span style="color: #c00000;">**0.057**</span> → 0.063 | <span style="color: #c00000;">**17.3**</span> → 49.6 | <span style="color: #c00000;">**88.2**</span> → 71.1 |
| Transverse B / +0.5 | 0.082 → <span style="color: #c00000;">**0.034**</span> | 0.067 → <span style="color: #c00000;">**0.037**</span> | 16.41 → <span style="color: #c00000;">**6.80**</span> | 0.071 → <span style="color: #c00000;">**0.063**</span> | 63.4 → <span style="color: #c00000;">**37.3**</span> | 48.5 → <span style="color: #c00000;">**79.6**</span> |
| Transverse B / -0.5 | <span style="color: #c00000;">**0.032**</span> → 0.071 | 0.049 → <span style="color: #c00000;">**0.021**</span> | <span style="color: #c00000;">**6.37**</span> → 14.24 | 0.046 → <span style="color: #c00000;">**0.040**</span> | <span style="color: #c00000;">**1.4**</span> → 20.9 | <span style="color: #c00000;">**98.1**</span> → 79.6 |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 96.4 → <span style="color: #c00000;">**100.0**</span> | <span style="color: #c00000;">**4.0**</span> → 16.0 | 53.33 → 53.33 | 2.67 → 2.67 | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**0.00**</span> | 1.00 → 1.00 |
| Trot / -0.5 | 96.4 → <span style="color: #c00000;">**96.9**</span> | <span style="color: #c00000;">**5.1**</span> → 7.8 | 52.50 → 52.50 | 3.50 → 3.50 | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**5.47**</span> | 1.00 → 1.00 |
| Bound / +0.5 | 96.9 → <span style="color: #c00000;">**97.9**</span> | 16.0 → <span style="color: #c00000;">**14.0**</span> | 50.42 → <span style="color: #c00000;">**51.95**</span> | 5.58 → <span style="color: #c00000;">**4.05**</span> | 0.00 → 0.00 | 5.47 → <span style="color: #c00000;">**3.61**</span> | 1.00 → 1.00 |
| Bound / -0.5 | 96.4 → <span style="color: #c00000;">**100.0**</span> | <span style="color: #c00000;">**10.4**</span> → 20.0 | 51.67 → <span style="color: #c00000;">**53.33**</span> | 4.33 → <span style="color: #c00000;">**2.67**</span> | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**0.00**</span> | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.5 → <span style="color: #c00000;">**96.4**</span> | 15.5 → <span style="color: #c00000;">**7.9**</span> | 52.92 → <span style="color: #c00000;">**53.79**</span> | <span style="color: #c00000;">**3.42**</span> → 3.79 | 2.08 → <span style="color: #c00000;">**0.00**</span> | 6.25 → 6.25 | 1.00 → 1.00 |
| Half-bound A / -0.5 | 96.4 → 96.4 | <span style="color: #c00000;">**14.0**</span> → 19.0 | 54.17 → 54.17 | 2.50 → <span style="color: #c00000;">**2.17**</span> | 2.08 → 2.08 | 4.69 → 4.69 | 1.00 → 1.00 |
| Half-bound B / +0.5 | 95.5 → <span style="color: #c00000;">**96.9**</span> | <span style="color: #c00000;">**14.0**</span> → 20.8 | 53.33 → <span style="color: #c00000;">**54.58**</span> | 3.00 → <span style="color: #c00000;">**2.08**</span> | 2.08 → 2.08 | 6.25 → <span style="color: #c00000;">**3.91**</span> | 1.00 → 1.00 |
| Half-bound B / -0.5 | 94.6 → <span style="color: #c00000;">**97.3**</span> | 24.0 → <span style="color: #c00000;">**19.0**</span> | <span style="color: #c00000;">**55.83**</span> → 54.17 | 2.50 → 2.50 | 6.25 → <span style="color: #c00000;">**2.08**</span> | 4.69 → <span style="color: #c00000;">**3.12**</span> | 1.00 → 1.00 |
| Rotary A / +0.5 | 87.1 → <span style="color: #c00000;">**88.4**</span> | 42.9 → <span style="color: #c00000;">**38.0**</span> | <span style="color: #c00000;">**55.41**</span> → 57.50 | <span style="color: #c00000;">**1.25**</span> → 1.50 | <span style="color: #c00000;">**13.54**</span> → 14.58 | 12.50 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Rotary A / -0.5 | 86.1 → <span style="color: #c00000;">**89.3**</span> | 35.5 → <span style="color: #c00000;">**33.0**</span> | <span style="color: #c00000;">**56.18**</span> → 56.67 | <span style="color: #c00000;">**2.64**</span> → 3.67 | 14.58 → <span style="color: #c00000;">**12.50**</span> | 13.42 → <span style="color: #c00000;">**9.37**</span> | 1.00 → 1.00 |
| Rotary B / +0.5 | <span style="color: #c00000;">**90.1**</span> → 88.4 | <span style="color: #c00000;">**27.3**</span> → 37.8 | 54.23 → <span style="color: #c00000;">**57.50**</span> | <span style="color: #c00000;">**2.57**</span> → 2.83 | <span style="color: #c00000;">**8.49**</span> → 14.58 | 10.94 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Rotary B / -0.5 | <span style="color: #c00000;">**89.6**</span> → 89.3 | <span style="color: #c00000;">**31.5**</span> → 33.0 | 55.31 → <span style="color: #c00000;">**56.67**</span> | 2.31 → <span style="color: #c00000;">**2.00**</span> | <span style="color: #c00000;">**10.42**</span> → 12.50 | 10.35 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Transverse A / +0.5 | 99.1 → 99.1 | 8.0 → <span style="color: #c00000;">**6.0**</span> | 54.17 → <span style="color: #c00000;">**55.00**</span> | 2.17 → <span style="color: #c00000;">**1.67**</span> | 0.00 → 0.00 | 1.56 → 1.56 | 1.00 → 1.00 |
| Transverse A / -0.5 | <span style="color: #c00000;">**99.1**</span> → 98.2 | 11.0 → <span style="color: #c00000;">**10.0**</span> | <span style="color: #c00000;">**54.17**</span> → 53.33 | <span style="color: #c00000;">**2.50**</span> → 3.00 | 0.00 → 0.00 | <span style="color: #c00000;">**1.56**</span> → 3.12 | 1.00 → 1.00 |
| Transverse B / +0.5 | 97.3 → <span style="color: #c00000;">**98.2**</span> | 8.0 → <span style="color: #c00000;">**6.0**</span> | 53.33 → <span style="color: #c00000;">**54.17**</span> | 2.67 → <span style="color: #c00000;">**2.17**</span> | 0.00 → 0.00 | 4.69 → <span style="color: #c00000;">**3.12**</span> | 1.00 → 1.00 |
| Transverse B / -0.5 | 96.4 → <span style="color: #c00000;">**99.1**</span> | <span style="color: #c00000;">**2.0**</span> → 10.0 | 53.33 → <span style="color: #c00000;">**54.17**</span> | 3.00 → <span style="color: #c00000;">**2.17**</span> | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 38.6 → <span style="color: #c00000;">**30.7**</span> | 0.582 → <span style="color: #c00000;">**0.575**</span> | 0.00 → 0.00 | 7.15 → <span style="color: #c00000;">**6.79**</span> | 0.88 → <span style="color: #c00000;">**0.09**</span> | 10.71 → <span style="color: #c00000;">**3.93**</span> |
| Trot / -0.5 | 37.9 → <span style="color: #c00000;">**35.5**</span> | 0.625 → <span style="color: #c00000;">**0.614**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**3.23**</span> → 14.93 | <span style="color: #c00000;">**0.18**</span> → 6.25 | <span style="color: #c00000;">**0.23**</span> → 7.38 |
| Bound / +0.5 | <span style="color: #c00000;">**97.8**</span> → 103.4 | 0.594 → <span style="color: #c00000;">**0.534**</span> | 0.00 → 0.00 | 7.01 → <span style="color: #c00000;">**1.35**</span> | <span style="color: #c00000;">**0.08**</span> → 0.51 | <span style="color: #c00000;">**1.07**</span> → 8.54 |
| Bound / -0.5 | <span style="color: #c00000;">**99.7**</span> → 100.8 | 0.572 → <span style="color: #c00000;">**0.546**</span> | 0.00 → 0.00 | 7.27 → <span style="color: #c00000;">**0.30**</span> | <span style="color: #c00000;">**0.74**</span> → 4.11 | 21.50 → <span style="color: #c00000;">**13.25**</span> |
| Half-bound A / +0.5 | <span style="color: #c00000;">**78.0**</span> → 86.5 | <span style="color: #c00000;">**0.613**</span> → 0.616 | 0.00 → 0.00 | <span style="color: #c00000;">**4.98**</span> → 5.41 | 5.47 → <span style="color: #c00000;">**2.14**</span> | 14.58 → <span style="color: #c00000;">**1.12**</span> |
| Half-bound A / -0.5 | 93.1 → <span style="color: #c00000;">**92.5**</span> | <span style="color: #c00000;">**0.555**</span> → 0.587 | 0.00 → 0.00 | 4.10 → <span style="color: #c00000;">**0.23**</span> | 4.71 → <span style="color: #c00000;">**2.85**</span> | 14.33 → <span style="color: #c00000;">**1.62**</span> |
| Half-bound B / +0.5 | 97.2 → <span style="color: #c00000;">**87.2**</span> | <span style="color: #c00000;">**0.582**</span> → 0.608 | 0.00 → 0.00 | <span style="color: #c00000;">**4.77**</span> → 4.97 | <span style="color: #c00000;">**8.15**</span> → 8.33 | <span style="color: #c00000;">**0.80**</span> → 2.37 |
| Half-bound B / -0.5 | <span style="color: #c00000;">**87.3**</span> → 96.6 | 0.636 → <span style="color: #c00000;">**0.559**</span> | 0.00 → 0.00 | 6.18 → <span style="color: #c00000;">**0.39**</span> | 4.45 → <span style="color: #c00000;">**3.16**</span> | 17.08 → <span style="color: #c00000;">**14.26**</span> |
| Rotary A / +0.5 | 99.8 → <span style="color: #c00000;">**91.0**</span> | <span style="color: #c00000;">**0.545**</span> → 0.621 | 0.00 → 0.00 | <span style="color: #c00000;">**0.94**</span> → 2.95 | <span style="color: #c00000;">**0.63**</span> → 3.70 | 19.02 → <span style="color: #c00000;">**12.70**</span> |
| Rotary A / -0.5 | <span style="color: #c00000;">**90.0**</span> → 92.4 | 0.586 → <span style="color: #c00000;">**0.516**</span> | 0.00 → 0.00 | 10.17 → <span style="color: #c00000;">**1.42**</span> | 5.13 → <span style="color: #c00000;">**5.07**</span> | 25.84 → <span style="color: #c00000;">**4.49**</span> |
| Rotary B / +0.5 | <span style="color: #c00000;">**83.6**</span> → 87.5 | 0.596 → <span style="color: #c00000;">**0.533**</span> | 0.56 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**0.93**</span> → 2.65 | <span style="color: #c00000;">**5.07**</span> → 6.87 | 22.14 → <span style="color: #c00000;">**16.80**</span> |
| Rotary B / -0.5 | 92.7 → <span style="color: #c00000;">**89.3**</span> | 0.557 → <span style="color: #c00000;">**0.535**</span> | 0.00 → 0.00 | 4.97 → <span style="color: #c00000;">**2.06**</span> | <span style="color: #c00000;">**4.26**</span> → 5.64 | 6.15 → <span style="color: #c00000;">**4.73**</span> |
| Transverse A / +0.5 | <span style="color: #c00000;">**66.1**</span> → 75.1 | 0.629 → <span style="color: #c00000;">**0.560**</span> | 0.00 → 0.00 | 8.32 → <span style="color: #c00000;">**6.26**</span> | <span style="color: #c00000;">**0.35**</span> → 0.48 | 7.13 → <span style="color: #c00000;">**3.93**</span> |
| Transverse A / -0.5 | <span style="color: #c00000;">**73.9**</span> → 78.7 | 0.604 → <span style="color: #c00000;">**0.599**</span> | 0.00 → 0.00 | 5.36 → <span style="color: #c00000;">**5.23**</span> | <span style="color: #c00000;">**1.66**</span> → 3.01 | 8.98 → <span style="color: #c00000;">**2.12**</span> |
| Transverse B / +0.5 | 82.1 → <span style="color: #c00000;">**75.3**</span> | <span style="color: #c00000;">**0.592**</span> → 0.618 | 0.56 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**2.73**</span> → 3.82 | <span style="color: #c00000;">**4.19**</span> → 5.15 | 8.89 → <span style="color: #c00000;">**8.80**</span> |
| Transverse B / -0.5 | <span style="color: #c00000;">**71.4**</span> → 79.5 | 0.708 → <span style="color: #c00000;">**0.656**</span> | 0.00 → 0.00 | 8.72 → <span style="color: #c00000;">**6.48**</span> | 4.17 → <span style="color: #c00000;">**0.63**</span> | <span style="color: #c00000;">**10.79**</span> → 14.37 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Bound / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 8.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary A / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 83.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary B / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | <span style="color: #c00000;">**98.0**</span> → 0.0 | <span style="color: #c00000;">**98.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**98.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-1"></a>

#### X1, vx = ±1 m/s

<details>
<summary>Show all metrics for X1 at ±1 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | <span style="color: #c00000;">**0.056**</span> → 0.074 | <span style="color: #c00000;">**0.020**</span> → 0.049 | <span style="color: #c00000;">**5.61**</span> → 7.44 | <span style="color: #c00000;">**0.060**</span> → 0.102 | <span style="color: #c00000;">**26.5**</span> → 77.9 | <span style="color: #c00000;">**90.5**</span> → 28.6 |
| Trot / -1 | 0.086 → <span style="color: #c00000;">**0.060**</span> | 0.027 → <span style="color: #c00000;">**0.021**</span> | 8.63 → <span style="color: #c00000;">**5.98**</span> | <span style="color: #c00000;">**0.041**</span> → 0.043 | 10.4 → <span style="color: #c00000;">**2.2**</span> | 90.4 → <span style="color: #c00000;">**96.4**</span> |
| Bound / +1 | <span style="color: #c00000;">**0.069**</span> → 0.101 | 0.021 → <span style="color: #c00000;">**0.019**</span> | <span style="color: #c00000;">**6.91**</span> → 10.06 | 0.079 → <span style="color: #c00000;">**0.044**</span> | 44.9 → <span style="color: #c00000;">**42.2**</span> | <span style="color: #c00000;">**69.9**</span> → 63.9 |
| Bound / -1 | 0.116 → <span style="color: #c00000;">**0.060**</span> | 0.049 → <span style="color: #c00000;">**0.027**</span> | 11.56 → <span style="color: #c00000;">**6.05**</span> | 0.109 → <span style="color: #c00000;">**0.055**</span> | 124.2 → <span style="color: #c00000;">**47.9**</span> | -27.2 → <span style="color: #c00000;">**63.3**</span> |
| Half-bound A / +1 | <span style="color: #c00000;">**0.054**</span> → 0.081 | 0.032 → <span style="color: #c00000;">**0.030**</span> | <span style="color: #c00000;">**5.38**</span> → 8.11 | <span style="color: #c00000;">**0.059**</span> → 0.073 | <span style="color: #c00000;">**10.4**</span> → 25.9 | <span style="color: #c00000;">**95.8**</span> → 82.1 |
| Half-bound A / -1 | 0.074 → <span style="color: #c00000;">**0.057**</span> | 0.057 → <span style="color: #c00000;">**0.028**</span> | 7.44 → <span style="color: #c00000;">**5.72**</span> | 0.093 → <span style="color: #c00000;">**0.054**</span> | 72.2 → <span style="color: #c00000;">**23.7**</span> | 34.9 → <span style="color: #c00000;">**89.5**</span> |
| Half-bound B / +1 | <span style="color: #c00000;">**0.056**</span> → 0.101 | 0.044 → <span style="color: #c00000;">**0.033**</span> | <span style="color: #c00000;">**5.59**</span> → 10.14 | 0.132 → <span style="color: #c00000;">**0.054**</span> | 101.0 → <span style="color: #c00000;">**4.4**</span> | -2.0 → <span style="color: #c00000;">**87.4**</span> |
| Half-bound B / -1 | 0.189 → <span style="color: #c00000;">**0.071**</span> | <span style="color: #c00000;">**0.042**</span> → 0.057 | 18.85 → <span style="color: #c00000;">**7.06**</span> | 0.060 → <span style="color: #c00000;">**0.058**</span> | <span style="color: #c00000;">**3.8**</span> → 64.6 | <span style="color: #c00000;">**77.6**</span> → 41.1 |
| Rotary A / +1 | <span style="color: #c00000;">**0.076**</span> → 0.102 | <span style="color: #c00000;">**0.041**</span> → 0.061 | <span style="color: #c00000;">**7.60**</span> → 10.22 | 0.105 → <span style="color: #c00000;">**0.074**</span> | 125.8 → <span style="color: #c00000;">**10.1**</span> | -32.7 → <span style="color: #c00000;">**86.6**</span> |
| Rotary A / -1 | 0.078 → <span style="color: #c00000;">**0.058**</span> | <span style="color: #c00000;">**0.034**</span> → 0.066 | 7.80 → <span style="color: #c00000;">**5.77**</span> | 0.079 → <span style="color: #c00000;">**0.073**</span> | <span style="color: #c00000;">**33.3**</span> → 113.2 | <span style="color: #c00000;">**78.5**</span> → -19.6 |
| Rotary B / +1 | 0.087 → <span style="color: #c00000;">**0.075**</span> | 0.072 → <span style="color: #c00000;">**0.060**</span> | 8.73 → <span style="color: #c00000;">**7.49**</span> | 0.122 → <span style="color: #c00000;">**0.062**</span> | 23.4 → <span style="color: #c00000;">**6.6**</span> | <span style="color: #c00000;">**97.1**</span> → 91.7 |
| Rotary B / -1 | 0.095 → <span style="color: #c00000;">**0.073**</span> | 0.079 → <span style="color: #c00000;">**0.043**</span> | 9.54 → <span style="color: #c00000;">**7.35**</span> | 0.086 → <span style="color: #c00000;">**0.075**</span> | <span style="color: #c00000;">**14.3**</span> → 38.6 | <span style="color: #c00000;">**86.9**</span> → 74.5 |
| Transverse A / +1 | 0.083 → <span style="color: #c00000;">**0.075**</span> | <span style="color: #c00000;">**0.030**</span> → 0.033 | 8.29 → <span style="color: #c00000;">**7.53**</span> | <span style="color: #c00000;">**0.051**</span> → 0.058 | <span style="color: #c00000;">**12.8**</span> → 32.6 | <span style="color: #c00000;">**89.5**</span> → 78.8 |
| Transverse A / -1 | 0.075 → <span style="color: #c00000;">**0.067**</span> | <span style="color: #c00000;">**0.034**</span> → 0.045 | 7.48 → <span style="color: #c00000;">**6.71**</span> | <span style="color: #c00000;">**0.054**</span> → 0.082 | <span style="color: #c00000;">**19.6**</span> → 62.9 | <span style="color: #c00000;">**88.4**</span> → 51.7 |
| Transverse B / +1 | 0.075 → <span style="color: #c00000;">**0.073**</span> | <span style="color: #c00000;">**0.032**</span> → 0.043 | 7.49 → <span style="color: #c00000;">**7.25**</span> | 0.092 → <span style="color: #c00000;">**0.064**</span> | 52.8 → <span style="color: #c00000;">**40.0**</span> | 60.9 → <span style="color: #c00000;">**73.7**</span> |
| Transverse B / -1 | 0.097 → <span style="color: #c00000;">**0.068**</span> | 0.039 → <span style="color: #c00000;">**0.025**</span> | 9.74 → <span style="color: #c00000;">**6.82**</span> | <span style="color: #c00000;">**0.047**</span> → 0.059 | <span style="color: #c00000;">**22.0**</span> → 42.3 | <span style="color: #c00000;">**83.9**</span> → 69.3 |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 94.6 → <span style="color: #c00000;">**97.3**</span> | <span style="color: #c00000;">**7.0**</span> → 20.3 | 50.83 → 50.83 | 5.17 → 5.17 | 0.00 → 0.00 | 9.38 → <span style="color: #c00000;">**4.69**</span> | 1.00 → 1.00 |
| Trot / -1 | 95.5 → 95.5 | 5.5 → <span style="color: #c00000;">**5.1**</span> | 51.67 → 51.67 | 4.33 → 4.33 | 0.00 → 0.00 | 7.81 → 7.81 | 1.00 → 1.00 |
| Bound / +1 | 93.8 → <span style="color: #c00000;">**95.8**</span> | <span style="color: #c00000;">**15.0**</span> → 18.0 | 49.17 → <span style="color: #c00000;">**50.29**</span> | 6.83 → <span style="color: #c00000;">**5.71**</span> | 0.00 → 0.00 | 10.94 → <span style="color: #c00000;">**7.28**</span> | 1.00 → 1.00 |
| Bound / -1 | <span style="color: #c00000;">**97.3**</span> → 96.6 | <span style="color: #c00000;">**10.4**</span> → 23.3 | <span style="color: #c00000;">**52.50**</span> → 50.16 | <span style="color: #c00000;">**3.50**</span> → 5.84 | 0.00 → 0.00 | <span style="color: #c00000;">**4.69**</span> → 5.96 | 1.00 → 1.00 |
| Half-bound A / +1 | <span style="color: #c00000;">**94.6**</span> → 92.9 | <span style="color: #c00000;">**9.0**</span> → 19.0 | 50.78 → <span style="color: #c00000;">**52.60**</span> | 5.55 → <span style="color: #c00000;">**3.73**</span> | <span style="color: #c00000;">**0.00**</span> → 4.17 | 9.47 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Half-bound A / -1 | <span style="color: #c00000;">**95.5**</span> → 93.5 | <span style="color: #c00000;">**19.0**</span> → 21.3 | <span style="color: #c00000;">**53.33**</span> → 51.09 | <span style="color: #c00000;">**3.33**</span> → 4.91 | <span style="color: #c00000;">**2.08**</span> → 2.73 | <span style="color: #c00000;">**6.25**</span> → 9.38 | 1.00 → 1.00 |
| Half-bound B / +1 | 93.8 → <span style="color: #c00000;">**95.5**</span> | <span style="color: #c00000;">**12.9**</span> → 18.2 | 50.00 → <span style="color: #c00000;">**51.67**</span> | 6.33 → <span style="color: #c00000;">**4.33**</span> | 0.00 → 0.00 | 10.94 → <span style="color: #c00000;">**7.81**</span> | 1.00 → 1.00 |
| Half-bound B / -1 | 94.2 → <span style="color: #c00000;">**96.4**</span> | 29.0 → <span style="color: #c00000;">**24.0**</span> | <span style="color: #c00000;">**54.56**</span> → 52.53 | 4.22 → <span style="color: #c00000;">**3.81**</span> | 6.25 → <span style="color: #c00000;">**2.08**</span> | 5.52 → <span style="color: #c00000;">**4.69**</span> | 1.00 → 1.00 |
| Rotary A / +1 | 80.8 → <span style="color: #c00000;">**85.4**</span> | 55.7 → <span style="color: #c00000;">**44.7**</span> | 49.69 → <span style="color: #c00000;">**51.38**</span> | 6.31 → <span style="color: #c00000;">**4.62**</span> | 14.58 → <span style="color: #c00000;">**10.42**</span> | 22.66 → <span style="color: #c00000;">**17.72**</span> | 1.00 → 1.00 |
| Rotary A / -1 | 81.2 → <span style="color: #c00000;">**86.6**</span> | 40.5 → <span style="color: #c00000;">**38.0**</span> | 52.50 → <span style="color: #c00000;">**53.33**</span> | 4.17 → <span style="color: #c00000;">**3.33**</span> | 16.67 → <span style="color: #c00000;">**12.50**</span> | 20.31 → <span style="color: #c00000;">**14.06**</span> | 1.00 → 1.00 |
| Rotary B / +1 | <span style="color: #c00000;">**86.7**</span> → 84.9 | <span style="color: #c00000;">**35.9**</span> → 47.8 | 50.91 → <span style="color: #c00000;">**52.42**</span> | 5.79 → <span style="color: #c00000;">**3.91**</span> | <span style="color: #c00000;">**8.14**</span> → 12.30 | 17.24 → <span style="color: #c00000;">**17.19**</span> | 1.00 → 1.00 |
| Rotary B / -1 | <span style="color: #c00000;">**85.4**</span> → 83.9 | 48.0 → <span style="color: #c00000;">**41.3**</span> | 53.02 → <span style="color: #c00000;">**53.33**</span> | 4.69 → <span style="color: #c00000;">**3.33**</span> | <span style="color: #c00000;">**12.50**</span> → 14.58 | <span style="color: #c00000;">**16.21**</span> → 17.19 | 1.00 → 1.00 |
| Transverse A / +1 | <span style="color: #c00000;">**97.3**</span> → 96.4 | 12.0 → <span style="color: #c00000;">**11.5**</span> | <span style="color: #c00000;">**51.67**</span> → 51.35 | 4.67 → <span style="color: #c00000;">**4.65**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**4.69**</span> → 6.25 | 1.00 → 1.00 |
| Transverse A / -1 | <span style="color: #c00000;">**98.0**</span> → 95.5 | <span style="color: #c00000;">**11.0**</span> → 15.0 | <span style="color: #c00000;">**53.12**</span> → 50.00 | <span style="color: #c00000;">**3.21**</span> → 6.00 | 0.00 → 0.00 | <span style="color: #c00000;">**3.52**</span> → 7.81 | 1.00 → 1.00 |
| Transverse B / +1 | 93.8 → <span style="color: #c00000;">**97.3**</span> | 17.0 → <span style="color: #c00000;">**14.0**</span> | 49.17 → <span style="color: #c00000;">**51.67**</span> | 6.83 → <span style="color: #c00000;">**4.33**</span> | 0.00 → 0.00 | 10.94 → <span style="color: #c00000;">**4.69**</span> | 1.00 → 1.00 |
| Transverse B / -1 | 96.4 → <span style="color: #c00000;">**96.9**</span> | 11.0 → <span style="color: #c00000;">**10.0**</span> | 51.67 → <span style="color: #c00000;">**52.14**</span> | 4.67 → <span style="color: #c00000;">**3.86**</span> | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**5.37**</span> | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 89.2 → <span style="color: #c00000;">**74.0**</span> | <span style="color: #c00000;">**0.722**</span> → 0.742 | 3.06 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**5.07**</span> → 9.62 | 4.30 → <span style="color: #c00000;">**0.81**</span> | 12.13 → <span style="color: #c00000;">**2.58**</span> |
| Trot / -1 | 79.3 → <span style="color: #c00000;">**72.7**</span> | <span style="color: #c00000;">**0.750**</span> → 0.768 | 0.00 → 0.00 | <span style="color: #c00000;">**10.52**</span> → 11.10 | <span style="color: #c00000;">**2.49**</span> → 4.11 | 2.74 → <span style="color: #c00000;">**1.31**</span> |
| Bound / +1 | <span style="color: #c00000;">**143.8**</span> → 145.6 | 0.849 → <span style="color: #c00000;">**0.705**</span> | 0.00 → 0.00 | 6.22 → <span style="color: #c00000;">**0.24**</span> | 4.57 → <span style="color: #c00000;">**4.38**</span> | 19.46 → <span style="color: #c00000;">**16.64**</span> |
| Bound / -1 | 152.2 → <span style="color: #c00000;">**144.2**</span> | 0.802 → <span style="color: #c00000;">**0.713**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**4.26**</span> → 4.31 | <span style="color: #c00000;">**0.13**</span> → 3.94 | 17.58 → <span style="color: #c00000;">**1.36**</span> |
| Half-bound A / +1 | <span style="color: #c00000;">**127.8**</span> → 130.8 | 0.830 → <span style="color: #c00000;">**0.701**</span> | 0.83 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**5.59**</span> → 5.93 | 5.87 → <span style="color: #c00000;">**2.72**</span> | 18.04 → <span style="color: #c00000;">**0.80**</span> |
| Half-bound A / -1 | 148.8 → <span style="color: #c00000;">**146.0**</span> | <span style="color: #c00000;">**0.773**</span> → 0.825 | <span style="color: #c00000;">**0.00**</span> → 0.19 | <span style="color: #c00000;">**0.90**</span> → 2.17 | 2.40 → <span style="color: #c00000;">**1.77**</span> | 7.03 → <span style="color: #c00000;">**0.33**</span> |
| Half-bound B / +1 | 158.0 → <span style="color: #c00000;">**125.0**</span> | 0.842 → <span style="color: #c00000;">**0.770**</span> | 0.83 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**2.79**</span> → 3.09 | 11.73 → <span style="color: #c00000;">**7.17**</span> | <span style="color: #c00000;">**5.64**</span> → 14.94 |
| Half-bound B / -1 | <span style="color: #c00000;">**134.5**</span> → 141.4 | 0.795 → <span style="color: #c00000;">**0.715**</span> | 0.00 → 0.00 | 6.99 → <span style="color: #c00000;">**0.37**</span> | 5.02 → <span style="color: #c00000;">**2.33**</span> | 14.28 → <span style="color: #c00000;">**1.12**</span> |
| Rotary A / +1 | 157.5 → <span style="color: #c00000;">**125.5**</span> | 0.880 → <span style="color: #c00000;">**0.807**</span> | 0.00 → 0.00 | 4.77 → <span style="color: #c00000;">**3.42**</span> | 8.64 → <span style="color: #c00000;">**7.13**</span> | 24.72 → <span style="color: #c00000;">**21.72**</span> |
| Rotary A / -1 | 156.2 → <span style="color: #c00000;">**139.8**</span> | 0.791 → <span style="color: #c00000;">**0.676**</span> | 0.28 → <span style="color: #c00000;">**0.00**</span> | 10.79 → <span style="color: #c00000;">**0.63**</span> | 6.89 → <span style="color: #c00000;">**1.67**</span> | 19.79 → <span style="color: #c00000;">**11.90**</span> |
| Rotary B / +1 | 183.1 → <span style="color: #c00000;">**133.2**</span> | 0.976 → <span style="color: #c00000;">**0.745**</span> | 0.83 → <span style="color: #c00000;">**0.00**</span> | 26.00 → <span style="color: #c00000;">**1.01**</span> | 14.91 → <span style="color: #c00000;">**3.27**</span> | 30.69 → <span style="color: #c00000;">**27.93**</span> |
| Rotary B / -1 | 144.8 → <span style="color: #c00000;">**127.5**</span> | 0.758 → <span style="color: #c00000;">**0.711**</span> | 0.00 → 0.00 | 4.77 → <span style="color: #c00000;">**0.12**</span> | 4.25 → <span style="color: #c00000;">**0.96**</span> | <span style="color: #c00000;">**5.21**</span> → 6.33 |
| Transverse A / +1 | <span style="color: #c00000;">**108.0**</span> → 117.6 | 0.780 → <span style="color: #c00000;">**0.756**</span> | 1.11 → <span style="color: #c00000;">**0.00**</span> | 11.36 → <span style="color: #c00000;">**3.82**</span> | 3.59 → <span style="color: #c00000;">**1.57**</span> | 14.07 → <span style="color: #c00000;">**4.59**</span> |
| Transverse A / -1 | <span style="color: #c00000;">**120.7**</span> → 121.9 | 0.809 → <span style="color: #c00000;">**0.788**</span> | <span style="color: #c00000;">**0.28**</span> → 0.83 | <span style="color: #c00000;">**7.93**</span> → 9.77 | 4.23 → <span style="color: #c00000;">**0.36**</span> | 14.34 → <span style="color: #c00000;">**1.79**</span> |
| Transverse B / +1 | 130.3 → <span style="color: #c00000;">**115.4**</span> | 0.923 → <span style="color: #c00000;">**0.823**</span> | 2.78 → <span style="color: #c00000;">**0.00**</span> | 4.27 → <span style="color: #c00000;">**1.03**</span> | 4.66 → <span style="color: #c00000;">**3.12**</span> | 13.71 → <span style="color: #c00000;">**9.28**</span> |
| Transverse B / -1 | <span style="color: #c00000;">**113.8**</span> → 126.5 | <span style="color: #c00000;">**0.819**</span> → 0.880 | 0.00 → 0.00 | 7.48 → <span style="color: #c00000;">**4.20**</span> | 3.25 → <span style="color: #c00000;">**1.30**</span> | 8.45 → <span style="color: #c00000;">**7.34**</span> |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary A / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-1p5"></a>

#### X1, vx = ±1.5 m/s

<details>
<summary>Show all metrics for X1 at ±1.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 0.083 → <span style="color: #c00000;">**0.060**</span> | 0.056 → <span style="color: #c00000;">**0.040**</span> | 5.55 → <span style="color: #c00000;">**3.98**</span> | 0.082 → <span style="color: #c00000;">**0.081**</span> | <span style="color: #c00000;">**3.9**</span> → 47.1 | <span style="color: #c00000;">**101.3**</span> → 69.3 |
| Trot / -1.5 | <span style="color: #c00000;">**0.054**</span> → 0.066 | <span style="color: #c00000;">**0.036**</span> → 0.042 | <span style="color: #c00000;">**3.61**</span> → 4.37 | 0.077 → <span style="color: #c00000;">**0.060**</span> | 31.9 → <span style="color: #c00000;">**7.4**</span> | 84.2 → <span style="color: #c00000;">**98.7**</span> |
| Bound / +1.5 | 0.088 → <span style="color: #c00000;">**0.067**</span> | 0.060 → <span style="color: #c00000;">**0.059**</span> | 5.89 → <span style="color: #c00000;">**4.44**</span> | 0.121 → <span style="color: #c00000;">**0.107**</span> | 34.5 → <span style="color: #c00000;">**6.1**</span> | 80.2 → <span style="color: #c00000;">**98.0**</span> |
| Bound / -1.5 | 0.183 → <span style="color: #c00000;">**0.130**</span> | <span style="color: #c00000;">**0.046**</span> → 0.097 | 12.19 → <span style="color: #c00000;">**8.67**</span> | <span style="color: #c00000;">**0.081**</span> → 0.105 | 79.1 → <span style="color: #c00000;">**21.2**</span> | 23.3 → <span style="color: #c00000;">**82.1**</span> |
| Half-bound A / +1.5 | <span style="color: #c00000;">**0.084**</span> → 0.118 | 0.056 → <span style="color: #c00000;">**0.038**</span> | <span style="color: #c00000;">**5.57**</span> → 7.88 | 0.127 → <span style="color: #c00000;">**0.114**</span> | <span style="color: #c00000;">**42.6**</span> → 42.8 | <span style="color: #c00000;">**73.3**</span> → 66.9 |
| Half-bound A / -1.5 | 0.093 → <span style="color: #c00000;">**0.085**</span> | 0.061 → <span style="color: #c00000;">**0.052**</span> | 6.19 → <span style="color: #c00000;">**5.70**</span> | 0.084 → <span style="color: #c00000;">**0.077**</span> | 17.6 → <span style="color: #c00000;">**5.4**</span> | 90.1 → <span style="color: #c00000;">**95.5**</span> |
| Half-bound B / +1.5 | 0.112 → <span style="color: #c00000;">**0.080**</span> | <span style="color: #c00000;">**0.039**</span> → 0.056 | 7.47 → <span style="color: #c00000;">**5.33**</span> | 0.100 → <span style="color: #c00000;">**0.060**</span> | <span style="color: #c00000;">**19.9**</span> → 23.0 | 87.5 → <span style="color: #c00000;">**87.8**</span> |
| Half-bound B / -1.5 | 0.278 → <span style="color: #c00000;">**0.089**</span> | <span style="color: #c00000;">**0.049**</span> → 0.057 | 18.51 → <span style="color: #c00000;">**5.92**</span> | <span style="color: #c00000;">**0.092**</span> → 0.132 | <span style="color: #c00000;">**19.0**</span> → 38.4 | 75.8 → <span style="color: #c00000;">**76.9**</span> |
| Rotary A / +1.5 | 0.062 → <span style="color: #c00000;">**0.058**</span> | 0.083 → <span style="color: #c00000;">**0.074**</span> | 4.15 → <span style="color: #c00000;">**3.90**</span> | 0.096 → <span style="color: #c00000;">**0.082**</span> | 74.5 → <span style="color: #c00000;">**43.6**</span> | 26.7 → <span style="color: #c00000;">**72.8**</span> |
| Rotary A / -1.5 | <span style="color: #c00000;">**0.077**</span> → 0.092 | <span style="color: #c00000;">**0.054**</span> → 0.065 | <span style="color: #c00000;">**5.11**</span> → 6.11 | 0.106 → <span style="color: #c00000;">**0.063**</span> | <span style="color: #c00000;">**16.2**</span> → 103.0 | <span style="color: #c00000;">**91.8**</span> → -6.9 |
| Rotary B / +1.5 | 0.116 → <span style="color: #c00000;">**0.071**</span> | <span style="color: #c00000;">**0.060**</span> → 0.071 | 7.73 → <span style="color: #c00000;">**4.73**</span> | 0.121 → <span style="color: #c00000;">**0.111**</span> | <span style="color: #c00000;">**11.9**</span> → 43.3 | <span style="color: #c00000;">**101.5**</span> → 74.3 |
| Rotary B / -1.5 | 0.112 → <span style="color: #c00000;">**0.083**</span> | <span style="color: #c00000;">**0.067**</span> → 0.085 | 7.46 → <span style="color: #c00000;">**5.52**</span> | <span style="color: #c00000;">**0.118**</span> → 0.145 | <span style="color: #c00000;">**30.4**</span> → 40.4 | <span style="color: #c00000;">**80.4**</span> → 78.8 |
| Transverse A / +1.5 | 0.061 → <span style="color: #c00000;">**0.053**</span> | 0.085 → <span style="color: #c00000;">**0.043**</span> | 4.09 → <span style="color: #c00000;">**3.52**</span> | <span style="color: #c00000;">**0.086**</span> → 0.089 | <span style="color: #c00000;">**36.3**</span> → 54.8 | <span style="color: #c00000;">**76.2**</span> → 58.6 |
| Transverse A / -1.5 | <span style="color: #c00000;">**0.073**</span> → 0.087 | <span style="color: #c00000;">**0.057**</span> → 0.070 | <span style="color: #c00000;">**4.87**</span> → 5.80 | 0.098 → <span style="color: #c00000;">**0.096**</span> | <span style="color: #c00000;">**1.7**</span> → 6.6 | 95.9 → <span style="color: #c00000;">**101.4**</span> |
| Transverse B / +1.5 | 0.142 → <span style="color: #c00000;">**0.073**</span> | 0.060 → <span style="color: #c00000;">**0.034**</span> | 9.46 → <span style="color: #c00000;">**4.89**</span> | 0.140 → <span style="color: #c00000;">**0.070**</span> | 51.0 → <span style="color: #c00000;">**8.1**</span> | 65.4 → <span style="color: #c00000;">**94.7**</span> |
| Transverse B / -1.5 | 0.117 → <span style="color: #c00000;">**0.063**</span> | 0.039 → <span style="color: #c00000;">**0.025**</span> | 7.82 → <span style="color: #c00000;">**4.18**</span> | <span style="color: #c00000;">**0.070**</span> → 0.086 | <span style="color: #c00000;">**24.1**</span> → 37.6 | <span style="color: #c00000;">**84.0**</span> → 78.7 |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 92.9 → <span style="color: #c00000;">**93.8**</span> | <span style="color: #c00000;">**16.0**</span> → 22.7 | 46.67 → <span style="color: #c00000;">**47.81**</span> | 9.33 → <span style="color: #c00000;">**8.19**</span> | 0.00 → 0.00 | 12.50 → <span style="color: #c00000;">**10.89**</span> | 1.00 → 1.00 |
| Trot / -1.5 | 91.1 → 91.1 | 10.4 → 10.4 | 46.67 → 46.67 | 9.33 → 9.33 | 0.00 → 0.00 | 15.62 → 15.62 | 1.00 → 1.00 |
| Bound / +1.5 | 86.9 → <span style="color: #c00000;">**87.7**</span> | <span style="color: #c00000;">**35.1**</span> → 41.6 | <span style="color: #c00000;">**47.85**</span> → 46.76 | <span style="color: #c00000;">**8.15**</span> → 9.24 | 6.33 → <span style="color: #c00000;">**3.95**</span> | <span style="color: #c00000;">**18.16**</span> → 18.54 | <span style="color: #c00000;">**1.04**</span> → 1.06 |
| Bound / -1.5 | 89.6 → <span style="color: #c00000;">**91.1**</span> | 35.4 → <span style="color: #c00000;">**30.9**</span> | <span style="color: #c00000;">**48.30**</span> → 45.89 | <span style="color: #c00000;">**7.70**</span> → 10.11 | 3.71 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**15.49**</span> → 15.53 | 1.00 → 1.00 |
| Half-bound A / +1.5 | <span style="color: #c00000;">**92.0**</span> → 87.7 | <span style="color: #c00000;">**12.0**</span> → 30.9 | <span style="color: #c00000;">**50.00**</span> → 47.84 | <span style="color: #c00000;">**6.33**</span> → 8.16 | <span style="color: #c00000;">**2.08**</span> → 4.38 | <span style="color: #c00000;">**12.50**</span> → 18.28 | 1.00 → 1.00 |
| Half-bound A / -1.5 | <span style="color: #c00000;">**89.4**</span> → 88.0 | <span style="color: #c00000;">**29.1**</span> → 36.5 | <span style="color: #c00000;">**48.40**</span> → 47.08 | <span style="color: #c00000;">**7.60**</span> → 8.92 | <span style="color: #c00000;">**4.16**</span> → 4.17 | <span style="color: #c00000;">**15.50**</span> → 17.94 | 2.00 → <span style="color: #c00000;">**1.03**</span> |
| Half-bound B / +1.5 | 86.5 → <span style="color: #c00000;">**92.9**</span> | 43.5 → <span style="color: #c00000;">**21.0**</span> | 47.96 → <span style="color: #c00000;">**48.33**</span> | 8.04 → <span style="color: #c00000;">**8.00**</span> | 6.85 → <span style="color: #c00000;">**0.00**</span> | 18.47 → <span style="color: #c00000;">**12.50**</span> | 1.32 → <span style="color: #c00000;">**1.00**</span> |
| Half-bound B / -1.5 | 88.7 → <span style="color: #c00000;">**90.2**</span> | 30.0 → 30.0 | <span style="color: #c00000;">**50.30**</span> → 49.17 | <span style="color: #c00000;">**5.70**</span> → 6.83 | 6.32 → <span style="color: #c00000;">**4.17**</span> | 15.12 → <span style="color: #c00000;">**14.06**</span> | 1.36 → <span style="color: #c00000;">**1.00**</span> |
| Rotary A / +1.5 | 76.7 → <span style="color: #c00000;">**81.1**</span> | <span style="color: #c00000;">**23.6**</span> → 52.1 | 48.13 → <span style="color: #c00000;">**48.20**</span> | <span style="color: #c00000;">**7.87**</span> → 8.13 | 16.80 → <span style="color: #c00000;">**12.50**</span> | 28.13 → <span style="color: #c00000;">**23.68**</span> | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Rotary A / -1.5 | 77.5 → <span style="color: #c00000;">**80.8**</span> | <span style="color: #c00000;">**46.9**</span> → 55.6 | 48.05 → <span style="color: #c00000;">**50.06**</span> | 7.95 → <span style="color: #c00000;">**5.94**</span> | 16.56 → <span style="color: #c00000;">**15.21**</span> | 27.01 → <span style="color: #c00000;">**22.25**</span> | 1.14 → <span style="color: #c00000;">**1.00**</span> |
| Rotary B / +1.5 | <span style="color: #c00000;">**82.8**</span> → 71.4 | 44.7 → <span style="color: #c00000;">**18.0**</span> | <span style="color: #c00000;">**48.28**</span> → 47.50 | <span style="color: #c00000;">**7.72**</span> → 8.50 | <span style="color: #c00000;">**9.73**</span> → 22.92 | <span style="color: #c00000;">**22.73**</span> → 32.81 | 1.05 → <span style="color: #c00000;">**1.00**</span> |
| Rotary B / -1.5 | <span style="color: #c00000;">**79.5**</span> → 76.9 | 59.8 → <span style="color: #c00000;">**34.3**</span> | <span style="color: #c00000;">**48.94**</span> → 46.48 | <span style="color: #c00000;">**7.06**</span> → 9.77 | 15.46 → <span style="color: #c00000;">**14.28**</span> | <span style="color: #c00000;">**24.26**</span> → 29.73 | 1.21 → <span style="color: #c00000;">**1.00**</span> |
| Transverse A / +1.5 | <span style="color: #c00000;">**93.5**</span> → 90.5 | <span style="color: #c00000;">**19.0**</span> → 22.4 | <span style="color: #c00000;">**48.07**</span> → 45.26 | <span style="color: #c00000;">**7.93**</span> → 10.74 | 0.00 → 0.00 | <span style="color: #c00000;">**11.43**</span> → 16.70 | 1.00 → 1.00 |
| Transverse A / -1.5 | <span style="color: #c00000;">**93.5**</span> → 91.1 | 24.3 → <span style="color: #c00000;">**22.0**</span> | <span style="color: #c00000;">**47.44**</span> → 46.67 | <span style="color: #c00000;">**8.56**</span> → 9.33 | 0.00 → 0.00 | <span style="color: #c00000;">**11.42**</span> → 15.62 | 1.42 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / +1.5 | 89.6 → <span style="color: #c00000;">**93.5**</span> | 31.4 → <span style="color: #c00000;">**29.3**</span> | 46.98 → <span style="color: #c00000;">**49.75**</span> | 9.02 → <span style="color: #c00000;">**6.34**</span> | <span style="color: #c00000;">**2.05**</span> → 2.08 | 16.65 → <span style="color: #c00000;">**9.84**</span> | 1.05 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / -1.5 | 90.8 → <span style="color: #c00000;">**94.6**</span> | <span style="color: #c00000;">**14.6**</span> → 16.0 | 46.39 → <span style="color: #c00000;">**49.17**</span> | 9.61 → <span style="color: #c00000;">**6.83**</span> | 0.00 → 0.00 | 16.14 → <span style="color: #c00000;">**9.38**</span> | <span style="color: #c00000;">**1.19**</span> → 1.50 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 172.2 → <span style="color: #c00000;">**155.1**</span> | 1.080 → <span style="color: #c00000;">**1.069**</span> | 3.33 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**10.01**</span> → 14.94 | 0.88 → <span style="color: #c00000;">**0.39**</span> | 11.56 → <span style="color: #c00000;">**9.07**</span> |
| Trot / -1.5 | 195.0 → <span style="color: #c00000;">**163.8**</span> | 1.183 → <span style="color: #c00000;">**1.094**</span> | <span style="color: #c00000;">**0.00**</span> → 0.83 | <span style="color: #c00000;">**9.85**</span> → 11.52 | 4.59 → <span style="color: #c00000;">**3.06**</span> | 10.76 → <span style="color: #c00000;">**2.94**</span> |
| Bound / +1.5 | 263.7 → <span style="color: #c00000;">**229.4**</span> | 1.275 → <span style="color: #c00000;">**1.104**</span> | 1.62 → <span style="color: #c00000;">**0.29**</span> | 25.70 → <span style="color: #c00000;">**17.62**</span> | <span style="color: #c00000;">**4.02**</span> → 5.63 | <span style="color: #c00000;">**1.08**</span> → 2.81 |
| Bound / -1.5 | <span style="color: #c00000;">**238.8**</span> → 250.1 | 1.124 → <span style="color: #c00000;">**0.998**</span> | 1.06 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**0.72**</span> → 17.28 | <span style="color: #c00000;">**9.92**</span> → 15.08 | 5.35 → <span style="color: #c00000;">**2.20**</span> |
| Half-bound A / +1.5 | 245.4 → <span style="color: #c00000;">**233.3**</span> | 1.246 → <span style="color: #c00000;">**1.204**</span> | 2.50 → <span style="color: #c00000;">**0.83**</span> | <span style="color: #c00000;">**7.11**</span> → 12.88 | 9.68 → <span style="color: #c00000;">**8.38**</span> | <span style="color: #c00000;">**11.29**</span> → 14.58 |
| Half-bound A / -1.5 | 260.5 → <span style="color: #c00000;">**253.4**</span> | <span style="color: #c00000;">**1.133**</span> → 1.297 | <span style="color: #c00000;">**1.38**</span> → 1.84 | <span style="color: #c00000;">**5.51**</span> → 18.31 | <span style="color: #c00000;">**4.17**</span> → 5.50 | <span style="color: #c00000;">**2.24**</span> → 5.69 |
| Half-bound B / +1.5 | 235.0 → <span style="color: #c00000;">**198.8**</span> | 1.389 → <span style="color: #c00000;">**1.055**</span> | 4.94 → <span style="color: #c00000;">**1.39**</span> | <span style="color: #c00000;">**0.27**</span> → 8.05 | <span style="color: #c00000;">**11.54**</span> → 14.16 | <span style="color: #c00000;">**11.85**</span> → 18.25 |
| Half-bound B / -1.5 | <span style="color: #c00000;">**197.9**</span> → 223.3 | 1.119 → <span style="color: #c00000;">**1.043**</span> | 0.28 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**9.60**</span> → 10.05 | <span style="color: #c00000;">**3.12**</span> → 14.88 | 10.14 → <span style="color: #c00000;">**6.97**</span> |
| Rotary A / +1.5 | 241.0 → <span style="color: #c00000;">**218.1**</span> | 1.356 → <span style="color: #c00000;">**1.097**</span> | 3.59 → <span style="color: #c00000;">**1.39**</span> | 7.02 → <span style="color: #c00000;">**3.27**</span> | <span style="color: #c00000;">**1.09**</span> → 8.57 | <span style="color: #c00000;">**15.30**</span> → 23.87 |
| Rotary A / -1.5 | 305.6 → <span style="color: #c00000;">**244.5**</span> | 1.377 → <span style="color: #c00000;">**1.027**</span> | 2.78 → <span style="color: #c00000;">**0.07**</span> | <span style="color: #c00000;">**5.51**</span> → 9.91 | 16.24 → <span style="color: #c00000;">**5.24**</span> | <span style="color: #c00000;">**7.79**</span> → 17.23 |
| Rotary B / +1.5 | 287.1 → <span style="color: #c00000;">**224.2**</span> | 1.313 → <span style="color: #c00000;">**1.175**</span> | <span style="color: #c00000;">**2.42**</span> → 2.68 | 14.81 → <span style="color: #c00000;">**3.35**</span> | 13.09 → <span style="color: #c00000;">**0.93**</span> | <span style="color: #c00000;">**12.94**</span> → 20.13 |
| Rotary B / -1.5 | 249.1 → <span style="color: #c00000;">**240.0**</span> | 1.227 → <span style="color: #c00000;">**1.147**</span> | 1.27 → <span style="color: #c00000;">**0.77**</span> | 4.05 → <span style="color: #c00000;">**1.54**</span> | <span style="color: #c00000;">**10.29**</span> → 22.86 | <span style="color: #c00000;">**0.18**</span> → 5.79 |
| Transverse A / +1.5 | <span style="color: #c00000;">**180.3**</span> → 191.8 | 1.193 → 1.193 | 2.05 → <span style="color: #c00000;">**0.00**</span> | 10.00 → <span style="color: #c00000;">**1.76**</span> | 11.17 → <span style="color: #c00000;">**1.89**</span> | 15.95 → <span style="color: #c00000;">**13.99**</span> |
| Transverse A / -1.5 | 221.4 → <span style="color: #c00000;">**215.2**</span> | <span style="color: #c00000;">**1.161**</span> → 1.256 | <span style="color: #c00000;">**0.59**</span> → 0.83 | <span style="color: #c00000;">**9.87**</span> → 21.41 | 10.62 → <span style="color: #c00000;">**1.80**</span> | 3.36 → <span style="color: #c00000;">**1.14**</span> |
| Transverse B / +1.5 | 223.5 → <span style="color: #c00000;">**191.6**</span> | 1.326 → <span style="color: #c00000;">**1.220**</span> | 5.01 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**0.79**</span> → 9.48 | 9.34 → <span style="color: #c00000;">**7.15**</span> | <span style="color: #c00000;">**13.34**</span> → 16.31 |
| Transverse B / -1.5 | <span style="color: #c00000;">**185.0**</span> → 239.3 | <span style="color: #c00000;">**1.204**</span> → 1.238 | <span style="color: #c00000;">**0.56**</span> → 0.83 | <span style="color: #c00000;">**5.68**</span> → 10.74 | <span style="color: #c00000;">**0.62**</span> → 6.58 | <span style="color: #c00000;">**2.26**</span> → 2.37 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 0.0 → 0.0 | <span style="color: #c00000;">**43.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**96.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-2"></a>

#### X1, vx = ±2 m/s

<details>
<summary>Show all metrics for X1 at ±2 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | <span style="color: #c00000;">**0.113**</span> → 0.231 | <span style="color: #c00000;">**0.095**</span> → 0.165 | <span style="color: #c00000;">**5.64**</span> → 11.56 | <span style="color: #c00000;">**0.128**</span> → 0.296 | <span style="color: #c00000;">**9.3**</span> → 119.8 | <span style="color: #c00000;">**95.1**</span> → -27.1 |
| Trot / -2 | 0.147 → <span style="color: #c00000;">**0.109**</span> | 0.080 → <span style="color: #c00000;">**0.043**</span> | 7.37 → <span style="color: #c00000;">**5.44**</span> | 0.131 → <span style="color: #c00000;">**0.079**</span> | 88.9 → <span style="color: #c00000;">**13.8**</span> | 11.8 → <span style="color: #c00000;">**92.3**</span> |
| Bound / +2 | 0.198 → <span style="color: #c00000;">**0.124**</span> | <span style="color: #c00000;">**0.093**</span> → 0.106 | 9.89 → <span style="color: #c00000;">**6.20**</span> | 0.191 → <span style="color: #c00000;">**0.149**</span> | 89.6 → <span style="color: #c00000;">**69.9**</span> | 8.1 → <span style="color: #c00000;">**34.2**</span> |
| Bound / -2 | <span style="color: #c00000;">**0.271**</span> → 0.315 | <span style="color: #c00000;">**0.059**</span> → 0.086 | <span style="color: #c00000;">**13.56**</span> → 15.76 | <span style="color: #c00000;">**0.089**</span> → 0.177 | <span style="color: #c00000;">**58.7**</span> → 81.0 | <span style="color: #c00000;">**45.6**</span> → 20.7 |
| Half-bound A / +2 | 0.205 → <span style="color: #c00000;">**0.113**</span> | <span style="color: #c00000;">**0.071**</span> → 0.080 | 10.26 → <span style="color: #c00000;">**5.67**</span> | <span style="color: #c00000;">**0.145**</span> → 0.159 | <span style="color: #c00000;">**38.9**</span> → 73.6 | <span style="color: #c00000;">**69.6**</span> → 28.4 |
| Half-bound A / -2 | 0.153 → <span style="color: #c00000;">**0.152**</span> | 0.140 → <span style="color: #c00000;">**0.084**</span> | 7.65 → <span style="color: #c00000;">**7.62**</span> | 0.142 → <span style="color: #c00000;">**0.085**</span> | <span style="color: #c00000;">**7.1**</span> → 19.0 | <span style="color: #c00000;">**92.9**</span> → 87.7 |
| Half-bound B / +2 | <span style="color: #c00000;">**0.114**</span> → 0.128 | <span style="color: #c00000;">**0.068**</span> → 0.129 | <span style="color: #c00000;">**5.72**</span> → 6.39 | 0.191 → <span style="color: #c00000;">**0.106**</span> | <span style="color: #c00000;">**52.7**</span> → 66.1 | <span style="color: #c00000;">**60.7**</span> → 36.9 |
| Half-bound B / -2 | 0.247 → <span style="color: #c00000;">**0.172**</span> | 0.111 → <span style="color: #c00000;">**0.076**</span> | 12.37 → <span style="color: #c00000;">**8.60**</span> | 0.337 → <span style="color: #c00000;">**0.139**</span> | 109.1 → <span style="color: #c00000;">**34.8**</span> | -7.6 → <span style="color: #c00000;">**76.6**</span> |
| Rotary A / +2 | 0.261 → <span style="color: #c00000;">**0.135**</span> | 0.109 → <span style="color: #c00000;">**0.085**</span> | 13.07 → <span style="color: #c00000;">**6.75**</span> | 0.199 → <span style="color: #c00000;">**0.130**</span> | 39.8 → <span style="color: #c00000;">**4.3**</span> | 66.8 → <span style="color: #c00000;">**93.2**</span> |
| Rotary A / -2 | 0.176 → <span style="color: #c00000;">**0.156**</span> | 0.089 → <span style="color: #c00000;">**0.077**</span> | 8.81 → <span style="color: #c00000;">**7.82**</span> | 0.129 → <span style="color: #c00000;">**0.119**</span> | <span style="color: #c00000;">**4.3**</span> → 13.3 | <span style="color: #c00000;">**91.1**</span> → 89.6 |
| Rotary B / +2 | <span style="color: #c00000;">**0.159**</span> → 0.186 | 0.093 → <span style="color: #c00000;">**0.086**</span> | <span style="color: #c00000;">**7.93**</span> → 9.30 | <span style="color: #c00000;">**0.173**</span> → 0.201 | <span style="color: #c00000;">**9.2**</span> → 62.4 | <span style="color: #c00000;">**92.7**</span> → 45.0 |
| Rotary B / -2 | <span style="color: #c00000;">**0.105**</span> → 0.128 | 0.131 → <span style="color: #c00000;">**0.112**</span> | <span style="color: #c00000;">**5.23**</span> → 6.40 | <span style="color: #c00000;">**0.133**</span> → 0.225 | <span style="color: #c00000;">**51.4**</span> → 67.3 | <span style="color: #c00000;">**64.4**</span> → 37.9 |
| Transverse A / +2 | 0.288 → <span style="color: #c00000;">**0.117**</span> | 0.074 → <span style="color: #c00000;">**0.065**</span> | 14.42 → <span style="color: #c00000;">**5.84**</span> | 0.120 → <span style="color: #c00000;">**0.109**</span> | <span style="color: #c00000;">**3.5**</span> → 45.5 | <span style="color: #c00000;">**84.8**</span> → 65.9 |
| Transverse A / -2 | 0.149 → <span style="color: #c00000;">**0.091**</span> | <span style="color: #c00000;">**0.048**</span> → 0.066 | 7.43 → <span style="color: #c00000;">**4.55**</span> | 0.111 → <span style="color: #c00000;">**0.101**</span> | 23.6 → <span style="color: #c00000;">**21.3**</span> | 84.2 → <span style="color: #c00000;">**91.1**</span> |
| Transverse B / +2 | <span style="color: #c00000;">**0.102**</span> → 0.267 | <span style="color: #c00000;">**0.065**</span> → 0.081 | <span style="color: #c00000;">**5.11**</span> → 13.33 | 0.122 → <span style="color: #c00000;">**0.119**</span> | <span style="color: #c00000;">**11.4**</span> → 37.7 | <span style="color: #c00000;">**98.1**</span> → 68.6 |
| Transverse B / -2 | 0.175 → <span style="color: #c00000;">**0.136**</span> | 0.092 → <span style="color: #c00000;">**0.045**</span> | 8.75 → <span style="color: #c00000;">**6.80**</span> | 0.157 → <span style="color: #c00000;">**0.105**</span> | 69.6 → <span style="color: #c00000;">**14.4**</span> | 34.0 → <span style="color: #c00000;">**89.5**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 89.3 → 89.3 | <span style="color: #c00000;">**16.5**</span> → 28.6 | 43.37 → <span style="color: #c00000;">**44.71**</span> | 12.63 → <span style="color: #c00000;">**11.29**</span> | <span style="color: #c00000;">**0.00**</span> → 1.18 | 18.73 → <span style="color: #c00000;">**17.79**</span> | 1.39 → <span style="color: #c00000;">**1.19**</span> |
| Trot / -2 | 86.7 → <span style="color: #c00000;">**89.4**</span> | <span style="color: #c00000;">**14.7**</span> → 18.3 | 41.77 → <span style="color: #c00000;">**44.43**</span> | 14.23 → <span style="color: #c00000;">**11.57**</span> | <span style="color: #c00000;">**0.00**</span> → 0.03 | 23.34 → <span style="color: #c00000;">**18.46**</span> | <span style="color: #c00000;">**1.00**</span> → 1.12 |
| Bound / +2 | <span style="color: #c00000;">**81.6**</span> → 78.4 | <span style="color: #c00000;">**49.9**</span> → 72.5 | <span style="color: #c00000;">**44.54**</span> → 43.98 | <span style="color: #c00000;">**11.46**</span> → 12.02 | <span style="color: #c00000;">**8.35**</span> → 11.39 | <span style="color: #c00000;">**25.95**</span> → 29.27 | 1.43 → <span style="color: #c00000;">**1.04**</span> |
| Bound / -2 | 80.5 → <span style="color: #c00000;">**82.9**</span> | 51.9 → <span style="color: #c00000;">**46.1**</span> | 43.46 → <span style="color: #c00000;">**46.02**</span> | 12.54 → <span style="color: #c00000;">**9.98**</span> | <span style="color: #c00000;">**8.33**</span> → 8.72 | 27.88 → <span style="color: #c00000;">**23.42**</span> | 2.00 → <span style="color: #c00000;">**1.35**</span> |
| Half-bound A / +2 | <span style="color: #c00000;">**87.7**</span> → 77.2 | <span style="color: #c00000;">**21.1**</span> → 61.0 | <span style="color: #c00000;">**47.08**</span> → 41.23 | <span style="color: #c00000;">**8.92**</span> → 14.77 | <span style="color: #c00000;">**3.95**</span> → 8.33 | <span style="color: #c00000;">**18.53**</span> → 33.62 | 1.28 → <span style="color: #c00000;">**1.06**</span> |
| Half-bound A / -2 | <span style="color: #c00000;">**81.9**</span> → 80.0 | <span style="color: #c00000;">**36.9**</span> → 52.0 | <span style="color: #c00000;">**43.06**</span> → 42.51 | <span style="color: #c00000;">**12.94**</span> → 13.49 | <span style="color: #c00000;">**6.05**</span> → 6.63 | <span style="color: #c00000;">**27.09**</span> → 29.97 | 2.00 → <span style="color: #c00000;">**1.05**</span> |
| Half-bound B / +2 | 72.3 → <span style="color: #c00000;">**85.9**</span> | 73.2 → <span style="color: #c00000;">**43.1**</span> | 39.26 → <span style="color: #c00000;">**45.30**</span> | 16.74 → <span style="color: #c00000;">**10.84**</span> | 11.54 → <span style="color: #c00000;">**4.30**</span> | 39.73 → <span style="color: #c00000;">**21.45**</span> | 1.88 → <span style="color: #c00000;">**1.19**</span> |
| Half-bound B / -2 | 72.8 → <span style="color: #c00000;">**82.5**</span> | 67.0 → <span style="color: #c00000;">**36.5**</span> | 42.92 → <span style="color: #c00000;">**44.58**</span> | 13.08 → <span style="color: #c00000;">**11.42**</span> | 16.67 → <span style="color: #c00000;">**6.33**</span> | 35.16 → <span style="color: #c00000;">**25.86**</span> | 2.27 → <span style="color: #c00000;">**1.41**</span> |
| Rotary A / +2 | 73.4 → <span style="color: #c00000;">**75.8**</span> | <span style="color: #c00000;">**23.5**</span> → 69.3 | 44.51 → <span style="color: #c00000;">**47.48**</span> | 11.49 → <span style="color: #c00000;">**8.65**</span> | <span style="color: #c00000;">**16.33**</span> → 17.81 | 34.35 → <span style="color: #c00000;">**29.02**</span> | 2.01 → <span style="color: #c00000;">**1.90**</span> |
| Rotary A / -2 | 67.9 → <span style="color: #c00000;">**71.5**</span> | 48.9 → <span style="color: #c00000;">**44.2**</span> | 43.30 → <span style="color: #c00000;">**44.54**</span> | 12.70 → <span style="color: #c00000;">**11.46**</span> | 21.70 → <span style="color: #c00000;">**18.00**</span> | 39.87 → <span style="color: #c00000;">**36.41**</span> | 2.00 → <span style="color: #c00000;">**1.12**</span> |
| Rotary B / +2 | <span style="color: #c00000;">**78.4**</span> → 70.8 | 56.1 → <span style="color: #c00000;">**11.4**</span> | <span style="color: #c00000;">**45.25**</span> → 41.83 | <span style="color: #c00000;">**10.75**</span> → 14.17 | <span style="color: #c00000;">**11.89**</span> → 15.05 | <span style="color: #c00000;">**28.81**</span> → 39.86 | <span style="color: #c00000;">**1.36**</span> → 2.00 |
| Rotary B / -2 | <span style="color: #c00000;">**71.5**</span> → 68.1 | 45.7 → <span style="color: #c00000;">**42.1**</span> | 40.99 → <span style="color: #c00000;">**44.56**</span> | 15.01 → <span style="color: #c00000;">**11.44**</span> | <span style="color: #c00000;">**14.71**</span> → 22.99 | 38.84 → <span style="color: #c00000;">**38.52**</span> | 2.00 → <span style="color: #c00000;">**1.86**</span> |
| Transverse A / +2 | <span style="color: #c00000;">**90.1**</span> → 86.1 | <span style="color: #c00000;">**8.7**</span> → 31.7 | <span style="color: #c00000;">**49.05**</span> → 43.90 | <span style="color: #c00000;">**7.14**</span> → 12.10 | 2.72 → <span style="color: #c00000;">**2.36**</span> | <span style="color: #c00000;">**15.34**</span> → 22.58 | 1.44 → <span style="color: #c00000;">**1.24**</span> |
| Transverse A / -2 | <span style="color: #c00000;">**88.4**</span> → 84.3 | <span style="color: #c00000;">**18.0**</span> → 40.9 | <span style="color: #c00000;">**44.21**</span> → 43.65 | <span style="color: #c00000;">**11.79**</span> → 12.35 | <span style="color: #c00000;">**0.03**</span> → 4.16 | <span style="color: #c00000;">**20.25**</span> → 24.41 | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / +2 | 80.1 → <span style="color: #c00000;">**88.3**</span> | 43.7 → <span style="color: #c00000;">**37.5**</span> | 38.70 → <span style="color: #c00000;">**45.18**</span> | 17.30 → <span style="color: #c00000;">**10.82**</span> | 2.79 → <span style="color: #c00000;">**2.37**</span> | 32.66 → <span style="color: #c00000;">**18.66**</span> | 1.84 → <span style="color: #c00000;">**1.19**</span> |
| Transverse B / -2 | 81.4 → <span style="color: #c00000;">**88.9**</span> | 35.2 → <span style="color: #c00000;">**21.8**</span> | 43.09 → <span style="color: #c00000;">**44.67**</span> | 12.91 → <span style="color: #c00000;">**11.33**</span> | 6.79 → <span style="color: #c00000;">**0.00**</span> | 27.46 → <span style="color: #c00000;">**19.37**</span> | 1.67 → <span style="color: #c00000;">**1.00**</span> |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 265.2 → <span style="color: #c00000;">**243.6**</span> | <span style="color: #c00000;">**1.398**</span> → 1.429 | 4.49 → <span style="color: #c00000;">**0.29**</span> | 9.86 → <span style="color: #c00000;">**6.30**</span> | 1.91 → <span style="color: #c00000;">**1.43**</span> | <span style="color: #c00000;">**10.53**</span> → 11.61 |
| Trot / -2 | 266.6 → <span style="color: #c00000;">**258.6**</span> | 1.431 → <span style="color: #c00000;">**1.366**</span> | 1.64 → <span style="color: #c00000;">**0.83**</span> | 6.72 → <span style="color: #c00000;">**4.33**</span> | 4.09 → <span style="color: #c00000;">**1.08**</span> | 10.71 → <span style="color: #c00000;">**7.60**</span> |
| Bound / +2 | 365.4 → <span style="color: #c00000;">**277.3**</span> | 1.570 → <span style="color: #c00000;">**1.404**</span> | 4.15 → <span style="color: #c00000;">**2.70**</span> | 8.27 → <span style="color: #c00000;">**3.30**</span> | 3.69 → <span style="color: #c00000;">**2.02**</span> | <span style="color: #c00000;">**0.97**</span> → 7.37 |
| Bound / -2 | 336.9 → <span style="color: #c00000;">**303.8**</span> | 1.471 → <span style="color: #c00000;">**1.211**</span> | 3.24 → <span style="color: #c00000;">**1.37**</span> | <span style="color: #c00000;">**5.94**</span> → 14.10 | 16.37 → <span style="color: #c00000;">**15.17**</span> | <span style="color: #c00000;">**0.76**</span> → 12.60 |
| Half-bound A / +2 | 329.9 → <span style="color: #c00000;">**326.7**</span> | <span style="color: #c00000;">**1.492**</span> → 1.659 | 3.12 → <span style="color: #c00000;">**1.71**</span> | <span style="color: #c00000;">**8.33**</span> → 13.23 | <span style="color: #c00000;">**1.02**</span> → 6.72 | <span style="color: #c00000;">**11.71**</span> → 17.35 |
| Half-bound A / -2 | 328.4 → <span style="color: #c00000;">**303.0**</span> | <span style="color: #c00000;">**1.484**</span> → 1.587 | <span style="color: #c00000;">**2.10**</span> → 3.98 | <span style="color: #c00000;">**4.63**</span> → 27.21 | 9.50 → <span style="color: #c00000;">**6.08**</span> | <span style="color: #c00000;">**5.87**</span> → 7.26 |
| Half-bound B / +2 | 322.8 → <span style="color: #c00000;">**269.6**</span> | 1.558 → <span style="color: #c00000;">**1.407**</span> | 6.39 → <span style="color: #c00000;">**2.73**</span> | 4.17 → <span style="color: #c00000;">**2.50**</span> | 8.88 → <span style="color: #c00000;">**8.52**</span> | <span style="color: #c00000;">**12.21**</span> → 14.93 |
| Half-bound B / -2 | <span style="color: #c00000;">**265.4**</span> → 312.9 | 1.433 → <span style="color: #c00000;">**1.392**</span> | 3.67 → <span style="color: #c00000;">**2.32**</span> | 10.15 → <span style="color: #c00000;">**8.53**</span> | <span style="color: #c00000;">**3.67**</span> → 11.27 | <span style="color: #c00000;">**2.09**</span> → 8.66 |
| Rotary A / +2 | 287.5 → <span style="color: #c00000;">**280.2**</span> | 1.631 → <span style="color: #c00000;">**1.374**</span> | 5.05 → <span style="color: #c00000;">**2.26**</span> | 9.92 → <span style="color: #c00000;">**6.77**</span> | 6.53 → <span style="color: #c00000;">**4.61**</span> | <span style="color: #c00000;">**3.65**</span> → 23.49 |
| Rotary A / -2 | 350.6 → <span style="color: #c00000;">**345.9**</span> | 1.681 → <span style="color: #c00000;">**1.382**</span> | 4.56 → <span style="color: #c00000;">**2.70**</span> | <span style="color: #c00000;">**1.18**</span> → 7.37 | <span style="color: #c00000;">**5.46**</span> → 5.55 | <span style="color: #c00000;">**0.51**</span> → 8.40 |
| Rotary B / +2 | 418.7 → <span style="color: #c00000;">**308.0**</span> | 1.643 → <span style="color: #c00000;">**1.558**</span> | <span style="color: #c00000;">**3.49**</span> → 5.23 | 16.06 → <span style="color: #c00000;">**15.73**</span> | <span style="color: #c00000;">**5.63**</span> → 7.50 | <span style="color: #c00000;">**10.47**</span> → 24.14 |
| Rotary B / -2 | <span style="color: #c00000;">**348.8**</span> → 363.2 | <span style="color: #c00000;">**1.482**</span> → 1.554 | 4.18 → <span style="color: #c00000;">**2.73**</span> | <span style="color: #c00000;">**4.36**</span> → 6.80 | <span style="color: #c00000;">**12.21**</span> → 15.99 | <span style="color: #c00000;">**2.01**</span> → 11.28 |
| Transverse A / +2 | <span style="color: #c00000;">**245.7**</span> → 305.8 | <span style="color: #c00000;">**1.473**</span> → 1.639 | 3.32 → <span style="color: #c00000;">**0.58**</span> | <span style="color: #c00000;">**0.61**</span> → 4.79 | <span style="color: #c00000;">**7.68**</span> → 10.06 | 16.77 → <span style="color: #c00000;">**16.56**</span> |
| Transverse A / -2 | 297.8 → <span style="color: #c00000;">**293.0**</span> | <span style="color: #c00000;">**1.441**</span> → 1.502 | <span style="color: #c00000;">**1.47**</span> → 1.92 | <span style="color: #c00000;">**0.70**</span> → 23.41 | 12.58 → <span style="color: #c00000;">**2.98**</span> | 7.03 → <span style="color: #c00000;">**0.49**</span> |
| Transverse B / +2 | 315.3 → <span style="color: #c00000;">**269.4**</span> | <span style="color: #c00000;">**1.494**</span> → 1.545 | 5.83 → <span style="color: #c00000;">**2.03**</span> | <span style="color: #c00000;">**2.56**</span> → 6.82 | <span style="color: #c00000;">**2.73**</span> → 9.28 | <span style="color: #c00000;">**14.08**</span> → 20.78 |
| Transverse B / -2 | <span style="color: #c00000;">**267.5**</span> → 338.7 | 1.547 → <span style="color: #c00000;">**1.526**</span> | 3.18 → <span style="color: #c00000;">**2.14**</span> | 7.29 → <span style="color: #c00000;">**5.68**</span> | 6.54 → <span style="color: #c00000;">**5.82**</span> | <span style="color: #c00000;">**0.42**</span> → 7.00 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 10.0 | 100.0 → 100.0 |
| Trot / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 86.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 | 100.0 → 100.0 |
| Half-bound A / +2 | <span style="color: #c00000;">**55.0**</span> → 0.0 | <span style="color: #c00000;">**55.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**55.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 |
| Half-bound A / -2 | <span style="color: #c00000;">**9.0**</span> → 0.0 | <span style="color: #c00000;">**9.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**9.0**</span> → 0.0 | 0.0 → 0.0 | 9.0 → <span style="color: #c00000;">**100.0**</span> | 9.0 → <span style="color: #c00000;">**100.0**</span> |
| Half-bound B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary A / +2 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | 0.0 → 0.0 | 73.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**96.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +2 | 87.0 → <span style="color: #c00000;">**100.0**</span> | 87.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 87.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -2 | <span style="color: #c00000;">**98.0**</span> → 0.0 | <span style="color: #c00000;">**98.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**98.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +2 | <span style="color: #c00000;">**99.0**</span> → 94.0 | <span style="color: #c00000;">**99.0**</span> → 94.0 | 0.0 → 0.0 | 74.0 → <span style="color: #c00000;">**94.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -2 | 99.0 → <span style="color: #c00000;">**100.0**</span> | 99.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 99.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

### X1_go2bend: full gait/direction details

<a id="detail-x1-go2bend-0p5"></a>

#### X1_go2bend, vx = ±0.5 m/s

<details>
<summary>Show all metrics for X1_go2bend at ±0.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 0.027 → <span style="color: #c00000;">**0.025**</span> | 0.045 → <span style="color: #c00000;">**0.030**</span> | 5.30 → <span style="color: #c00000;">**4.93**</span> | <span style="color: #c00000;">**0.034**</span> → 0.063 | <span style="color: #c00000;">**10.0**</span> → 45.2 | <span style="color: #c00000;">**95.0**</span> → 70.9 |
| Trot / -0.5 | <span style="color: #c00000;">**0.027**</span> → 0.045 | <span style="color: #c00000;">**0.033**</span> → 0.048 | <span style="color: #c00000;">**5.33**</span> → 8.99 | <span style="color: #c00000;">**0.049**</span> → 0.061 | <span style="color: #c00000;">**7.7**</span> → 39.1 | <span style="color: #c00000;">**96.3**</span> → 77.3 |
| Bound / +0.5 | 0.131 → <span style="color: #c00000;">**0.080**</span> | <span style="color: #c00000;">**0.032**</span> → 0.037 | 26.20 → <span style="color: #c00000;">**16.07**</span> | 0.085 → <span style="color: #c00000;">**0.045**</span> | <span style="color: #c00000;">**18.9**</span> → 46.2 | <span style="color: #c00000;">**118.7**</span> → 80.1 |
| Bound / -0.5 | <span style="color: #c00000;">**0.069**</span> → 0.082 | 0.042 → <span style="color: #c00000;">**0.037**</span> | <span style="color: #c00000;">**13.71**</span> → 16.43 | 0.133 → <span style="color: #c00000;">**0.075**</span> | 113.0 → <span style="color: #c00000;">**14.1**</span> | -16.5 → <span style="color: #c00000;">**79.5**</span> |
| Half-bound A / +0.5 | 0.068 → <span style="color: #c00000;">**0.045**</span> | <span style="color: #c00000;">**0.043**</span> → 0.052 | 13.65 → <span style="color: #c00000;">**8.94**</span> | 0.126 → <span style="color: #c00000;">**0.083**</span> | <span style="color: #c00000;">**20.2**</span> → 53.3 | <span style="color: #c00000;">**105.5**</span> → 65.2 |
| Half-bound A / -0.5 | <span style="color: #c00000;">**0.074**</span> → 0.131 | 0.051 → <span style="color: #c00000;">**0.046**</span> | <span style="color: #c00000;">**14.70**</span> → 26.18 | 0.075 → <span style="color: #c00000;">**0.064**</span> | 65.2 → <span style="color: #c00000;">**57.0**</span> | 44.8 → <span style="color: #c00000;">**45.3**</span> |
| Half-bound B / +0.5 | 0.072 → <span style="color: #c00000;">**0.047**</span> | 0.065 → <span style="color: #c00000;">**0.035**</span> | 14.39 → <span style="color: #c00000;">**9.36**</span> | 0.094 → <span style="color: #c00000;">**0.054**</span> | <span style="color: #c00000;">**2.4**</span> → 48.5 | <span style="color: #c00000;">**111.0**</span> → 65.9 |
| Half-bound B / -0.5 | 0.051 → <span style="color: #c00000;">**0.050**</span> | <span style="color: #c00000;">**0.033**</span> → 0.052 | 10.19 → <span style="color: #c00000;">**10.02**</span> | 0.089 → <span style="color: #c00000;">**0.044**</span> | <span style="color: #c00000;">**23.1**</span> → 30.6 | <span style="color: #c00000;">**99.1**</span> → 77.9 |
| Rotary A / +0.5 | <span style="color: #c00000;">**0.039**</span> → 0.061 | <span style="color: #c00000;">**0.058**</span> → 0.068 | <span style="color: #c00000;">**7.72**</span> → 12.18 | <span style="color: #c00000;">**0.111**</span> → 0.121 | <span style="color: #c00000;">**82.4**</span> → 122.6 | <span style="color: #c00000;">**25.3**</span> → -28.7 |
| Rotary A / -0.5 | 0.076 → <span style="color: #c00000;">**0.035**</span> | <span style="color: #c00000;">**0.045**</span> → 0.055 | 15.21 → <span style="color: #c00000;">**6.91**</span> | <span style="color: #c00000;">**0.055**</span> → 0.069 | <span style="color: #c00000;">**8.2**</span> → 9.4 | 111.9 → <span style="color: #c00000;">**100.2**</span> |
| Rotary B / +0.5 | 0.039 → <span style="color: #c00000;">**0.036**</span> | 0.082 → <span style="color: #c00000;">**0.077**</span> | 7.81 → <span style="color: #c00000;">**7.17**</span> | 0.117 → <span style="color: #c00000;">**0.100**</span> | 83.5 → <span style="color: #c00000;">**71.6**</span> | 23.6 → <span style="color: #c00000;">**37.8**</span> |
| Rotary B / -0.5 | <span style="color: #c00000;">**0.065**</span> → 0.171 | 0.065 → <span style="color: #c00000;">**0.055**</span> | <span style="color: #c00000;">**12.97**</span> → 34.15 | 0.100 → <span style="color: #c00000;">**0.075**</span> | 94.7 → <span style="color: #c00000;">**48.3**</span> | -2.4 → <span style="color: #c00000;">**48.0**</span> |
| Transverse A / +0.5 | <span style="color: #c00000;">**0.044**</span> → 0.068 | <span style="color: #c00000;">**0.029**</span> → 0.034 | <span style="color: #c00000;">**8.86**</span> → 13.52 | 0.078 → <span style="color: #c00000;">**0.065**</span> | 68.6 → <span style="color: #c00000;">**53.3**</span> | 42.1 → <span style="color: #c00000;">**63.3**</span> |
| Transverse A / -0.5 | <span style="color: #c00000;">**0.059**</span> → 0.088 | <span style="color: #c00000;">**0.022**</span> → 0.037 | <span style="color: #c00000;">**11.75**</span> → 17.56 | 0.111 → <span style="color: #c00000;">**0.051**</span> | 105.6 → <span style="color: #c00000;">**35.3**</span> | -11.6 → <span style="color: #c00000;">**72.9**</span> |
| Transverse B / +0.5 | 0.061 → <span style="color: #c00000;">**0.053**</span> | <span style="color: #c00000;">**0.014**</span> → 0.025 | 12.29 → <span style="color: #c00000;">**10.67**</span> | 0.069 → <span style="color: #c00000;">**0.042**</span> | 9.5 → <span style="color: #c00000;">**3.7**</span> | <span style="color: #c00000;">**104.2**</span> → 94.8 |
| Transverse B / -0.5 | <span style="color: #c00000;">**0.041**</span> → 0.059 | 0.040 → <span style="color: #c00000;">**0.036**</span> | <span style="color: #c00000;">**8.10**</span> → 11.87 | 0.100 → <span style="color: #c00000;">**0.047**</span> | 48.2 → <span style="color: #c00000;">**20.3**</span> | 67.9 → <span style="color: #c00000;">**84.8**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 97.3 → <span style="color: #c00000;">**100.0**</span> | <span style="color: #c00000;">**10.0**</span> → 13.0 | 52.50 → <span style="color: #c00000;">**54.17**</span> | 3.50 → <span style="color: #c00000;">**2.17**</span> | 0.00 → 0.00 | 4.69 → <span style="color: #c00000;">**0.00**</span> | 1.00 → 1.00 |
| Trot / -0.5 | 96.4 → <span style="color: #c00000;">**98.2**</span> | <span style="color: #c00000;">**10.4**</span> → 21.0 | 51.67 → <span style="color: #c00000;">**52.50**</span> | 4.33 → <span style="color: #c00000;">**3.83**</span> | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**3.12**</span> | 1.00 → 1.00 |
| Bound / +0.5 | 91.1 → <span style="color: #c00000;">**91.4**</span> | 43.0 → 43.0 | 51.67 → <span style="color: #c00000;">**52.79**</span> | 6.33 → <span style="color: #c00000;">**4.12**</span> | 6.25 → 6.25 | 10.94 → <span style="color: #c00000;">**10.40**</span> | 1.00 → 1.00 |
| Bound / -0.5 | 91.8 → <span style="color: #c00000;">**92.0**</span> | 35.6 → <span style="color: #c00000;">**30.0**</span> | 52.53 → <span style="color: #c00000;">**53.33**</span> | <span style="color: #c00000;">**3.47**</span> → 4.67 | <span style="color: #c00000;">**5.34**</span> → 6.25 | 10.30 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.5 → <span style="color: #c00000;">**99.1**</span> | 25.0 → <span style="color: #c00000;">**8.3**</span> | 52.50 → <span style="color: #c00000;">**56.41**</span> | 5.50 → <span style="color: #c00000;">**1.93**</span> | 0.00 → 0.00 | 7.81 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |
| Half-bound A / -0.5 | <span style="color: #c00000;">**96.0**</span> → 95.5 | 20.0 → 20.0 | 55.00 → <span style="color: #c00000;">**56.67**</span> | <span style="color: #c00000;">**2.17**</span> → 3.33 | <span style="color: #c00000;">**3.12**</span> → 4.17 | 4.69 → 4.69 | 1.00 → 1.00 |
| Half-bound B / +0.5 | 92.9 → <span style="color: #c00000;">**96.4**</span> | 22.0 → <span style="color: #c00000;">**10.8**</span> | <span style="color: #c00000;">**55.31**</span> → 54.17 | <span style="color: #c00000;">**2.31**</span> → 3.83 | 6.25 → <span style="color: #c00000;">**0.00**</span> | 7.81 → <span style="color: #c00000;">**6.25**</span> | 1.00 → 1.00 |
| Half-bound B / -0.5 | 90.2 → <span style="color: #c00000;">**97.3**</span> | 30.0 → <span style="color: #c00000;">**14.9**</span> | 54.17 → <span style="color: #c00000;">**55.01**</span> | 4.17 → <span style="color: #c00000;">**3.35**</span> | 8.33 → <span style="color: #c00000;">**0.00**</span> | 10.94 → <span style="color: #c00000;">**4.69**</span> | 1.00 → 1.00 |
| Rotary A / +0.5 | 77.7 → <span style="color: #c00000;">**88.4**</span> | 33.5 → <span style="color: #c00000;">**31.0**</span> | 61.67 → <span style="color: #c00000;">**60.00**</span> | 8.33 → <span style="color: #c00000;">**4.00**</span> | 31.25 → <span style="color: #c00000;">**16.67**</span> | 15.62 → <span style="color: #c00000;">**7.81**</span> | 1.00 → 1.00 |
| Rotary A / -0.5 | 78.6 → <span style="color: #c00000;">**85.2**</span> | <span style="color: #c00000;">**24.0**</span> → 45.6 | 52.50 → <span style="color: #c00000;">**54.64**</span> | 5.50 → <span style="color: #c00000;">**3.82**</span> | 18.75 → <span style="color: #c00000;">**14.68**</span> | 23.44 → <span style="color: #c00000;">**14.81**</span> | 1.00 → 1.00 |
| Rotary B / +0.5 | <span style="color: #c00000;">**90.2**</span> → 87.5 | <span style="color: #c00000;">**29.4**</span> → 41.0 | 59.17 → <span style="color: #c00000;">**57.50**</span> | 5.83 → <span style="color: #c00000;">**1.50**</span> | 14.58 → 14.58 | <span style="color: #c00000;">**6.25**</span> → 10.94 | 1.00 → 1.00 |
| Rotary B / -0.5 | <span style="color: #c00000;">**84.9**</span> → 84.4 | 45.7 → <span style="color: #c00000;">**45.5**</span> | <span style="color: #c00000;">**55.62**</span> → 55.44 | <span style="color: #c00000;">**3.48**</span> → 4.06 | <span style="color: #c00000;">**15.67**</span> → 16.67 | <span style="color: #c00000;">**14.66**</span> → 14.79 | 1.00 → 1.00 |
| Transverse A / +0.5 | 95.5 → <span style="color: #c00000;">**99.1**</span> | 17.0 → <span style="color: #c00000;">**14.0**</span> | 52.50 → <span style="color: #c00000;">**53.96**</span> | 3.83 → <span style="color: #c00000;">**2.38**</span> | 2.08 → <span style="color: #c00000;">**0.00**</span> | 6.25 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |
| Transverse A / -0.5 | 98.2 → 98.2 | 16.0 → 16.0 | 52.50 → <span style="color: #c00000;">**55.71**</span> | <span style="color: #c00000;">**3.50**</span> → 4.05 | <span style="color: #c00000;">**0.00**</span> → 2.08 | 3.12 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |
| Transverse B / +0.5 | 97.3 → 97.3 | 19.0 → <span style="color: #c00000;">**15.0**</span> | 52.50 → <span style="color: #c00000;">**54.17**</span> | <span style="color: #c00000;">**3.50**</span> → 3.83 | <span style="color: #c00000;">**0.00**</span> → 2.08 | 4.69 → <span style="color: #c00000;">**3.12**</span> | 1.00 → 1.00 |
| Transverse B / -0.5 | 97.3 → <span style="color: #c00000;">**99.1**</span> | 12.0 → <span style="color: #c00000;">**7.0**</span> | 53.33 → <span style="color: #c00000;">**55.83**</span> | 3.00 → <span style="color: #c00000;">**2.83**</span> | 0.00 → 0.00 | 4.69 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 34.5 → <span style="color: #c00000;">**31.7**</span> | 0.646 → <span style="color: #c00000;">**0.535**</span> | 2.22 → <span style="color: #c00000;">**0.83**</span> | 24.49 → <span style="color: #c00000;">**15.78**</span> | <span style="color: #c00000;">**1.24**</span> → 9.72 | <span style="color: #c00000;">**5.90**</span> → 14.90 |
| Trot / -0.5 | 38.7 → <span style="color: #c00000;">**30.0**</span> | 0.591 → <span style="color: #c00000;">**0.565**</span> | 2.50 → <span style="color: #c00000;">**0.00**</span> | 20.13 → <span style="color: #c00000;">**11.57**</span> | <span style="color: #c00000;">**2.04**</span> → 4.08 | 3.99 → <span style="color: #c00000;">**2.74**</span> |
| Bound / +0.5 | <span style="color: #c00000;">**110.4**</span> → 140.5 | 0.972 → <span style="color: #c00000;">**0.796**</span> | <span style="color: #c00000;">**2.18**</span> → 2.20 | 46.12 → <span style="color: #c00000;">**32.69**</span> | <span style="color: #c00000;">**13.46**</span> → 21.04 | <span style="color: #c00000;">**1.69**</span> → 2.98 |
| Bound / -0.5 | 141.8 → <span style="color: #c00000;">**132.2**</span> | 0.702 → <span style="color: #c00000;">**0.695**</span> | 3.15 → <span style="color: #c00000;">**1.11**</span> | 32.26 → <span style="color: #c00000;">**20.45**</span> | <span style="color: #c00000;">**12.45**</span> → 14.72 | <span style="color: #c00000;">**0.31**</span> → 15.68 |
| Half-bound A / +0.5 | <span style="color: #c00000;">**93.1**</span> → 103.4 | 0.816 → <span style="color: #c00000;">**0.760**</span> | <span style="color: #c00000;">**2.78**</span> → 3.61 | 44.14 → <span style="color: #c00000;">**32.02**</span> | 13.02 → <span style="color: #c00000;">**8.77**</span> | 5.43 → <span style="color: #c00000;">**2.26**</span> |
| Half-bound A / -0.5 | 118.3 → <span style="color: #c00000;">**94.9**</span> | 0.712 → <span style="color: #c00000;">**0.697**</span> | 2.72 → <span style="color: #c00000;">**2.22**</span> | 36.41 → <span style="color: #c00000;">**27.86**</span> | <span style="color: #c00000;">**10.90**</span> → 13.39 | 8.90 → <span style="color: #c00000;">**7.10**</span> |
| Half-bound B / +0.5 | <span style="color: #c00000;">**93.9**</span> → 121.9 | 0.812 → <span style="color: #c00000;">**0.732**</span> | 5.24 → <span style="color: #c00000;">**1.39**</span> | <span style="color: #c00000;">**22.63**</span> → 32.96 | 14.10 → <span style="color: #c00000;">**12.51**</span> | 8.12 → <span style="color: #c00000;">**0.81**</span> |
| Half-bound B / -0.5 | <span style="color: #c00000;">**96.8**</span> → 117.2 | 0.746 → <span style="color: #c00000;">**0.613**</span> | 3.89 → <span style="color: #c00000;">**1.67**</span> | 28.25 → <span style="color: #c00000;">**17.03**</span> | <span style="color: #c00000;">**8.98**</span> → 16.55 | <span style="color: #c00000;">**0.81**</span> → 5.49 |
| Rotary A / +0.5 | <span style="color: #c00000;">**68.7**</span> → 110.7 | 0.814 → <span style="color: #c00000;">**0.604**</span> | <span style="color: #c00000;">**1.11**</span> → 1.39 | <span style="color: #c00000;">**15.45**</span> → 22.95 | <span style="color: #c00000;">**9.00**</span> → 15.36 | 15.59 → <span style="color: #c00000;">**7.01**</span> |
| Rotary A / -0.5 | <span style="color: #c00000;">**139.4**</span> → 146.1 | 0.748 → <span style="color: #c00000;">**0.673**</span> | <span style="color: #c00000;">**1.94**</span> → 2.50 | <span style="color: #c00000;">**14.01**</span> → 22.68 | <span style="color: #c00000;">**3.18**</span> → 18.48 | 11.65 → <span style="color: #c00000;">**0.70**</span> |
| Rotary B / +0.5 | <span style="color: #c00000;">**88.9**</span> → 110.1 | <span style="color: #c00000;">**0.701**</span> → 0.722 | 3.89 → 3.89 | <span style="color: #c00000;">**31.84**</span> → 37.21 | 11.45 → <span style="color: #c00000;">**5.74**</span> | <span style="color: #c00000;">**11.96**</span> → 15.46 |
| Rotary B / -0.5 | 120.3 → <span style="color: #c00000;">**90.7**</span> | <span style="color: #c00000;">**0.665**</span> → 0.738 | 3.47 → <span style="color: #c00000;">**3.06**</span> | 28.58 → <span style="color: #c00000;">**21.49**</span> | <span style="color: #c00000;">**13.02**</span> → 14.70 | <span style="color: #c00000;">**0.96**</span> → 14.89 |
| Transverse A / +0.5 | <span style="color: #c00000;">**70.6**</span> → 76.1 | 0.731 → <span style="color: #c00000;">**0.659**</span> | <span style="color: #c00000;">**1.39**</span> → 2.78 | 10.48 → <span style="color: #c00000;">**9.45**</span> | <span style="color: #c00000;">**8.97**</span> → 9.43 | 18.56 → <span style="color: #c00000;">**14.20**</span> |
| Transverse A / -0.5 | <span style="color: #c00000;">**83.2**</span> → 87.2 | 0.619 → <span style="color: #c00000;">**0.602**</span> | 2.22 → <span style="color: #c00000;">**0.28**</span> | <span style="color: #c00000;">**10.03**</span> → 18.30 | <span style="color: #c00000;">**8.38**</span> → 11.94 | 11.38 → <span style="color: #c00000;">**6.43**</span> |
| Transverse B / +0.5 | <span style="color: #c00000;">**69.6**</span> → 83.2 | 0.640 → <span style="color: #c00000;">**0.561**</span> | 4.17 → <span style="color: #c00000;">**1.39**</span> | <span style="color: #c00000;">**5.81**</span> → 6.60 | <span style="color: #c00000;">**3.77**</span> → 12.16 | <span style="color: #c00000;">**5.95**</span> → 16.81 |
| Transverse B / -0.5 | 82.6 → <span style="color: #c00000;">**79.3**</span> | 0.645 → <span style="color: #c00000;">**0.601**</span> | 2.78 → <span style="color: #c00000;">**1.39**</span> | 23.47 → <span style="color: #c00000;">**14.19**</span> | <span style="color: #c00000;">**6.70**</span> → 12.59 | <span style="color: #c00000;">**1.19**</span> → 21.77 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -0.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound A / +0.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary A / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-go2bend-1"></a>

#### X1_go2bend, vx = ±1 m/s

<details>
<summary>Show all metrics for X1_go2bend at ±1 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | <span style="color: #c00000;">**0.034**</span> → 0.042 | 0.037 → <span style="color: #c00000;">**0.031**</span> | <span style="color: #c00000;">**3.40**</span> → 4.19 | <span style="color: #c00000;">**0.051**</span> → 0.052 | <span style="color: #c00000;">**3.8**</span> → 23.1 | <span style="color: #c00000;">**100.4**</span> → 94.2 |
| Trot / -1 | 0.084 → <span style="color: #c00000;">**0.063**</span> | 0.048 → <span style="color: #c00000;">**0.028**</span> | 8.45 → <span style="color: #c00000;">**6.31**</span> | <span style="color: #c00000;">**0.071**</span> → 0.073 | 47.7 → <span style="color: #c00000;">**33.4**</span> | 60.8 → <span style="color: #c00000;">**82.0**</span> |
| Bound / +1 | 0.181 → <span style="color: #c00000;">**0.163**</span> | <span style="color: #c00000;">**0.034**</span> → 0.046 | 18.14 → <span style="color: #c00000;">**16.30**</span> | 0.168 → <span style="color: #c00000;">**0.094**</span> | 100.9 → <span style="color: #c00000;">**6.7**</span> | -7.9 → <span style="color: #c00000;">**114.5**</span> |
| Bound / -1 | 0.069 → <span style="color: #c00000;">**0.054**</span> | 0.041 → <span style="color: #c00000;">**0.039**</span> | 6.85 → <span style="color: #c00000;">**5.40**</span> | 0.125 → <span style="color: #c00000;">**0.086**</span> | 95.3 → <span style="color: #c00000;">**12.8**</span> | 3.6 → <span style="color: #c00000;">**94.2**</span> |
| Half-bound A / +1 | <span style="color: #c00000;">**0.063**</span> → 0.082 | <span style="color: #c00000;">**0.031**</span> → 0.057 | <span style="color: #c00000;">**6.26**</span> → 8.15 | <span style="color: #c00000;">**0.094**</span> → 0.129 | <span style="color: #c00000;">**28.2**</span> → 117.3 | <span style="color: #c00000;">**91.3**</span> → -21.7 |
| Half-bound A / -1 | <span style="color: #c00000;">**0.090**</span> → 0.104 | <span style="color: #c00000;">**0.047**</span> → 0.072 | <span style="color: #c00000;">**8.98**</span> → 10.40 | <span style="color: #c00000;">**0.047**</span> → 0.084 | <span style="color: #c00000;">**14.0**</span> → 52.7 | <span style="color: #c00000;">**92.3**</span> → 60.0 |
| Half-bound B / +1 | <span style="color: #c00000;">**0.078**</span> → 0.084 | 0.060 → <span style="color: #c00000;">**0.052**</span> | <span style="color: #c00000;">**7.85**</span> → 8.44 | 0.093 → <span style="color: #c00000;">**0.076**</span> | <span style="color: #c00000;">**1.0**</span> → 5.8 | <span style="color: #c00000;">**95.3**</span> → 105.0 |
| Half-bound B / -1 | <span style="color: #c00000;">**0.070**</span> → 0.084 | <span style="color: #c00000;">**0.045**</span> → 0.066 | <span style="color: #c00000;">**6.96**</span> → 8.44 | 0.131 → <span style="color: #c00000;">**0.079**</span> | <span style="color: #c00000;">**49.5**</span> → 53.4 | <span style="color: #c00000;">**66.3**</span> → 53.4 |
| Rotary A / +1 | <span style="color: #c00000;">**0.064**</span> → 0.078 | <span style="color: #c00000;">**0.051**</span> → 0.070 | <span style="color: #c00000;">**6.43**</span> → 7.75 | 0.094 → <span style="color: #c00000;">**0.088**</span> | <span style="color: #c00000;">**43.2**</span> → 83.2 | <span style="color: #c00000;">**71.4**</span> → 20.4 |
| Rotary A / -1 | <span style="color: #c00000;">**0.075**</span> → 0.080 | <span style="color: #c00000;">**0.043**</span> → 0.044 | <span style="color: #c00000;">**7.45**</span> → 7.98 | <span style="color: #c00000;">**0.107**</span> → 0.141 | 74.7 → <span style="color: #c00000;">**67.7**</span> | 35.5 → <span style="color: #c00000;">**41.5**</span> |
| Rotary B / +1 | <span style="color: #c00000;">**0.083**</span> → 0.116 | <span style="color: #c00000;">**0.064**</span> → 0.087 | <span style="color: #c00000;">**8.27**</span> → 11.56 | <span style="color: #c00000;">**0.118**</span> → 0.191 | <span style="color: #c00000;">**7.9**</span> → 125.8 | <span style="color: #c00000;">**105.2**</span> → -37.8 |
| Rotary B / -1 | <span style="color: #c00000;">**0.068**</span> → 0.147 | <span style="color: #c00000;">**0.052**</span> → 0.081 | <span style="color: #c00000;">**6.76**</span> → 14.71 | <span style="color: #c00000;">**0.055**</span> → 0.078 | <span style="color: #c00000;">**19.5**</span> → 20.8 | <span style="color: #c00000;">**90.1**</span> → 82.1 |
| Transverse A / +1 | <span style="color: #c00000;">**0.040**</span> → 0.075 | <span style="color: #c00000;">**0.032**</span> → 0.060 | <span style="color: #c00000;">**4.02**</span> → 7.51 | 0.077 → <span style="color: #c00000;">**0.059**</span> | 69.3 → <span style="color: #c00000;">**29.6**</span> | 41.2 → <span style="color: #c00000;">**91.1**</span> |
| Transverse A / -1 | <span style="color: #c00000;">**0.075**</span> → 0.107 | 0.052 → <span style="color: #c00000;">**0.039**</span> | <span style="color: #c00000;">**7.47**</span> → 10.73 | 0.111 → <span style="color: #c00000;">**0.055**</span> | 105.5 → <span style="color: #c00000;">**9.7**</span> | -7.5 → <span style="color: #c00000;">**90.2**</span> |
| Transverse B / +1 | <span style="color: #c00000;">**0.064**</span> → 0.078 | <span style="color: #c00000;">**0.027**</span> → 0.057 | <span style="color: #c00000;">**6.42**</span> → 7.78 | <span style="color: #c00000;">**0.071**</span> → 0.078 | <span style="color: #c00000;">**1.6**</span> → 13.9 | 103.1 → <span style="color: #c00000;">**101.2**</span> |
| Transverse B / -1 | <span style="color: #c00000;">**0.060**</span> → 0.068 | <span style="color: #c00000;">**0.055**</span> → 0.063 | <span style="color: #c00000;">**6.03**</span> → 6.84 | 0.088 → <span style="color: #c00000;">**0.077**</span> | 29.1 → <span style="color: #c00000;">**18.6**</span> | 87.8 → <span style="color: #c00000;">**91.6**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 96.4 → 96.4 | 13.0 → <span style="color: #c00000;">**10.0**</span> | 50.83 → <span style="color: #c00000;">**51.67**</span> | 5.17 → <span style="color: #c00000;">**4.33**</span> | 0.00 → 0.00 | 6.25 → 6.25 | 1.00 → 1.00 |
| Trot / -1 | 96.4 → <span style="color: #c00000;">**97.3**</span> | <span style="color: #c00000;">**6.1**</span> → 15.7 | <span style="color: #c00000;">**52.29**</span> → 51.67 | <span style="color: #c00000;">**3.71**</span> → 4.33 | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**4.69**</span> | 1.00 → 1.00 |
| Bound / +1 | 88.4 → <span style="color: #c00000;">**89.5**</span> | 58.0 → <span style="color: #c00000;">**46.2**</span> | <span style="color: #c00000;">**50.83**</span> → 50.81 | 8.83 → <span style="color: #c00000;">**6.15**</span> | 8.33 → <span style="color: #c00000;">**6.71**</span> | 14.06 → <span style="color: #c00000;">**13.33**</span> | 1.00 → 1.00 |
| Bound / -1 | 88.4 → 88.4 | 44.9 → <span style="color: #c00000;">**40.0**</span> | 50.83 → <span style="color: #c00000;">**52.50**</span> | 5.17 → <span style="color: #c00000;">**3.83**</span> | <span style="color: #c00000;">**6.25**</span> → 8.33 | 15.62 → <span style="color: #c00000;">**14.06**</span> | 1.00 → 1.00 |
| Half-bound A / +1 | 93.8 → <span style="color: #c00000;">**97.3**</span> | 26.0 → <span style="color: #c00000;">**16.0**</span> | 50.83 → <span style="color: #c00000;">**56.67**</span> | 7.17 → <span style="color: #c00000;">**2.00**</span> | <span style="color: #c00000;">**0.00**</span> → 4.17 | 10.94 → <span style="color: #c00000;">**1.56**</span> | 1.00 → 1.00 |
| Half-bound A / -1 | 93.8 → 93.8 | <span style="color: #c00000;">**25.0**</span> → 30.0 | 51.67 → 51.67 | <span style="color: #c00000;">**4.33**</span> → 4.67 | 2.08 → 2.08 | 9.38 → 9.38 | 1.00 → 1.00 |
| Half-bound B / +1 | 93.2 → <span style="color: #c00000;">**93.8**</span> | 25.5 → <span style="color: #c00000;">**17.0**</span> | <span style="color: #c00000;">**52.81**</span> → 50.00 | <span style="color: #c00000;">**3.85**</span> → 6.33 | 4.17 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**8.79**</span> → 10.94 | 1.00 → 1.00 |
| Half-bound B / -1 | 86.6 → <span style="color: #c00000;">**94.6**</span> | 45.0 → <span style="color: #c00000;">**25.0**</span> | 48.33 → <span style="color: #c00000;">**52.50**</span> | 8.00 → <span style="color: #c00000;">**3.83**</span> | 6.25 → <span style="color: #c00000;">**2.08**</span> | 18.75 → <span style="color: #c00000;">**7.81**</span> | 1.00 → 1.00 |
| Rotary A / +1 | 71.5 → <span style="color: #c00000;">**84.8**</span> | 47.7 → <span style="color: #c00000;">**31.0**</span> | <span style="color: #c00000;">**55.86**</span> → 55.00 | 7.53 → <span style="color: #c00000;">**3.00**</span> | 31.25 → <span style="color: #c00000;">**14.58**</span> | 26.51 → <span style="color: #c00000;">**15.62**</span> | <span style="color: #c00000;">**1.00**</span> → 2.00 |
| Rotary A / -1 | 76.8 → <span style="color: #c00000;">**77.5**</span> | 46.0 → <span style="color: #c00000;">**40.6**</span> | 50.83 → <span style="color: #c00000;">**51.54**</span> | 5.50 → <span style="color: #c00000;">**4.80**</span> | 20.83 → 20.83 | 25.00 → <span style="color: #c00000;">**23.68**</span> | <span style="color: #c00000;">**1.00**</span> → 1.44 |
| Rotary B / +1 | 85.3 → <span style="color: #c00000;">**86.2**</span> | <span style="color: #c00000;">**29.4**</span> → 42.5 | 54.58 → <span style="color: #c00000;">**56.67**</span> | 5.42 → <span style="color: #c00000;">**3.33**</span> | <span style="color: #c00000;">**14.58**</span> → 16.67 | 14.84 → <span style="color: #c00000;">**11.72**</span> | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Rotary B / -1 | 81.4 → <span style="color: #c00000;">**83.0**</span> | 63.8 → <span style="color: #c00000;">**54.0**</span> | <span style="color: #c00000;">**52.79**</span> → 51.67 | <span style="color: #c00000;">**3.21**</span> → 4.67 | 18.10 → <span style="color: #c00000;">**14.58**</span> | 18.99 → <span style="color: #c00000;">**18.75**</span> | 1.00 → 1.00 |
| Transverse A / +1 | 94.6 → <span style="color: #c00000;">**95.5**</span> | 22.0 → 22.0 | 51.67 → <span style="color: #c00000;">**52.50**</span> | 4.33 → <span style="color: #c00000;">**3.83**</span> | 2.08 → 2.08 | 7.81 → <span style="color: #c00000;">**6.25**</span> | 1.00 → 1.00 |
| Transverse A / -1 | 96.4 → <span style="color: #c00000;">**98.2**</span> | 26.0 → <span style="color: #c00000;">**21.0**</span> | 50.83 → <span style="color: #c00000;">**53.33**</span> | 5.17 → <span style="color: #c00000;">**3.33**</span> | 0.00 → 0.00 | 6.25 → <span style="color: #c00000;">**3.12**</span> | 1.00 → 1.00 |
| Transverse B / +1 | <span style="color: #c00000;">**94.1**</span> → 93.8 | 29.8 → <span style="color: #c00000;">**29.0**</span> | <span style="color: #c00000;">**50.47**</span> → 50.00 | <span style="color: #c00000;">**5.53**</span> → 8.00 | <span style="color: #c00000;">**0.00**</span> → 2.08 | 10.25 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Transverse B / -1 | 95.5 → 95.5 | <span style="color: #c00000;">**17.0**</span> → 21.0 | <span style="color: #c00000;">**51.67**</span> → 50.83 | <span style="color: #c00000;">**4.67**</span> → 5.50 | 0.00 → 0.00 | 7.81 → 7.81 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 80.4 → <span style="color: #c00000;">**72.6**</span> | 0.849 → <span style="color: #c00000;">**0.708**</span> | 2.22 → <span style="color: #c00000;">**1.67**</span> | 3.38 → <span style="color: #c00000;">**1.19**</span> | <span style="color: #c00000;">**1.08**</span> → 2.40 | 12.01 → <span style="color: #c00000;">**11.47**</span> |
| Trot / -1 | 74.4 → <span style="color: #c00000;">**69.6**</span> | <span style="color: #c00000;">**0.651**</span> → 0.752 | 2.64 → <span style="color: #c00000;">**0.00**</span> | 17.25 → <span style="color: #c00000;">**9.43**</span> | <span style="color: #c00000;">**1.15**</span> → 3.93 | <span style="color: #c00000;">**0.32**</span> → 4.56 |
| Bound / +1 | <span style="color: #c00000;">**184.1**</span> → 241.7 | 1.198 → <span style="color: #c00000;">**1.113**</span> | <span style="color: #c00000;">**2.78**</span> → 2.85 | <span style="color: #c00000;">**33.72**</span> → 41.85 | <span style="color: #c00000;">**11.55**</span> → 19.02 | <span style="color: #c00000;">**7.88**</span> → 7.90 |
| Bound / -1 | 208.8 → <span style="color: #c00000;">**193.1**</span> | 0.954 → <span style="color: #c00000;">**0.887**</span> | 4.44 → <span style="color: #c00000;">**2.50**</span> | 21.21 → <span style="color: #c00000;">**10.58**</span> | <span style="color: #c00000;">**1.56**</span> → 12.53 | 4.91 → <span style="color: #c00000;">**2.11**</span> |
| Half-bound A / +1 | <span style="color: #c00000;">**152.1**</span> → 162.3 | 1.003 → <span style="color: #c00000;">**0.827**</span> | <span style="color: #c00000;">**2.78**</span> → 4.17 | <span style="color: #c00000;">**29.44**</span> → 32.90 | 10.93 → <span style="color: #c00000;">**4.15**</span> | 8.75 → <span style="color: #c00000;">**1.15**</span> |
| Half-bound A / -1 | 172.7 → <span style="color: #c00000;">**154.7**</span> | 0.899 → <span style="color: #c00000;">**0.851**</span> | 4.17 → <span style="color: #c00000;">**2.22**</span> | 25.47 → <span style="color: #c00000;">**11.29**</span> | <span style="color: #c00000;">**9.06**</span> → 11.57 | 5.26 → <span style="color: #c00000;">**0.78**</span> |
| Half-bound B / +1 | <span style="color: #c00000;">**147.0**</span> → 182.8 | 1.035 → <span style="color: #c00000;">**1.002**</span> | 5.00 → <span style="color: #c00000;">**1.94**</span> | <span style="color: #c00000;">**21.31**</span> → 35.07 | <span style="color: #c00000;">**8.60**</span> → 16.29 | <span style="color: #c00000;">**6.64**</span> → 6.66 |
| Half-bound B / -1 | <span style="color: #c00000;">**125.8**</span> → 175.9 | <span style="color: #c00000;">**0.767**</span> → 0.831 | 3.61 → <span style="color: #c00000;">**3.06**</span> | <span style="color: #c00000;">**10.28**</span> → 15.00 | 7.65 → <span style="color: #c00000;">**6.04**</span> | <span style="color: #c00000;">**0.43**</span> → 8.46 |
| Rotary A / +1 | <span style="color: #c00000;">**142.6**</span> → 160.6 | 1.041 → <span style="color: #c00000;">**0.908**</span> | 2.22 → <span style="color: #c00000;">**1.39**</span> | <span style="color: #c00000;">**13.48**</span> → 27.35 | <span style="color: #c00000;">**11.43**</span> → 14.48 | 9.50 → <span style="color: #c00000;">**2.85**</span> |
| Rotary A / -1 | 200.1 → <span style="color: #c00000;">**183.7**</span> | 0.922 → <span style="color: #c00000;">**0.887**</span> | 5.00 → <span style="color: #c00000;">**1.94**</span> | 14.08 → <span style="color: #c00000;">**3.22**</span> | 9.50 → <span style="color: #c00000;">**2.45**</span> | <span style="color: #c00000;">**2.26**</span> → 6.90 |
| Rotary B / +1 | <span style="color: #c00000;">**161.4**</span> → 179.2 | 0.965 → <span style="color: #c00000;">**0.857**</span> | <span style="color: #c00000;">**2.64**</span> → 3.89 | <span style="color: #c00000;">**21.22**</span> → 40.44 | 10.68 → <span style="color: #c00000;">**3.51**</span> | <span style="color: #c00000;">**15.75**</span> → 18.48 |
| Rotary B / -1 | 198.8 → <span style="color: #c00000;">**154.2**</span> | 0.938 → <span style="color: #c00000;">**0.860**</span> | 4.44 → <span style="color: #c00000;">**3.33**</span> | 20.59 → <span style="color: #c00000;">**7.46**</span> | <span style="color: #c00000;">**3.17**</span> → 12.99 | <span style="color: #c00000;">**0.20**</span> → 8.68 |
| Transverse A / +1 | <span style="color: #c00000;">**94.8**</span> → 104.5 | 0.845 → <span style="color: #c00000;">**0.828**</span> | <span style="color: #c00000;">**1.67**</span> → 2.78 | <span style="color: #c00000;">**2.90**</span> → 8.86 | 6.41 → <span style="color: #c00000;">**5.90**</span> | 17.66 → <span style="color: #c00000;">**14.14**</span> |
| Transverse A / -1 | <span style="color: #c00000;">**124.6**</span> → 130.6 | <span style="color: #c00000;">**0.728**</span> → 0.776 | 3.06 → <span style="color: #c00000;">**0.56**</span> | 12.60 → <span style="color: #c00000;">**6.16**</span> | <span style="color: #c00000;">**9.61**</span> → 10.27 | 6.89 → <span style="color: #c00000;">**1.73**</span> |
| Transverse B / +1 | <span style="color: #c00000;">**101.9**</span> → 129.2 | 0.861 → <span style="color: #c00000;">**0.839**</span> | 4.72 → <span style="color: #c00000;">**1.67**</span> | 4.99 → <span style="color: #c00000;">**0.66**</span> | <span style="color: #c00000;">**0.66**</span> → 11.39 | <span style="color: #c00000;">**14.05**</span> → 19.22 |
| Transverse B / -1 | <span style="color: #c00000;">**115.4**</span> → 118.8 | 0.741 → <span style="color: #c00000;">**0.704**</span> | 2.78 → <span style="color: #c00000;">**1.11**</span> | <span style="color: #c00000;">**8.61**</span> → 8.73 | 9.06 → <span style="color: #c00000;">**7.55**</span> | <span style="color: #c00000;">**1.20**</span> → 13.78 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Bound / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-go2bend-1p5"></a>

#### X1_go2bend, vx = ±1.5 m/s

<details>
<summary>Show all metrics for X1_go2bend at ±1.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 0.058 → <span style="color: #c00000;">**0.056**</span> | 0.071 → <span style="color: #c00000;">**0.056**</span> | 3.85 → <span style="color: #c00000;">**3.72**</span> | 0.064 → <span style="color: #c00000;">**0.056**</span> | 7.5 → <span style="color: #c00000;">**5.1**</span> | <span style="color: #c00000;">**100.1**</span> → 100.8 |
| Trot / -1.5 | 0.174 → <span style="color: #c00000;">**0.103**</span> | 0.043 → <span style="color: #c00000;">**0.030**</span> | 11.58 → <span style="color: #c00000;">**6.87**</span> | 0.087 → <span style="color: #c00000;">**0.051**</span> | 57.5 → <span style="color: #c00000;">**4.2**</span> | 49.1 → <span style="color: #c00000;">**95.3**</span> |
| Bound / +1.5 | <span style="color: #c00000;">**0.057**</span> → 0.078 | <span style="color: #c00000;">**0.054**</span> → 0.061 | <span style="color: #c00000;">**3.83**</span> → 5.17 | 0.093 → <span style="color: #c00000;">**0.089**</span> | 25.4 → <span style="color: #c00000;">**5.1**</span> | 90.2 → <span style="color: #c00000;">**103.4**</span> |
| Bound / -1.5 | 0.179 → <span style="color: #c00000;">**0.066**</span> | 0.154 → <span style="color: #c00000;">**0.061**</span> | 11.96 → <span style="color: #c00000;">**4.42**</span> | 0.155 → <span style="color: #c00000;">**0.127**</span> | 98.5 → <span style="color: #c00000;">**52.4**</span> | 5.2 → <span style="color: #c00000;">**62.6**</span> |
| Half-bound A / +1.5 | <span style="color: #c00000;">**0.058**</span> → 0.076 | <span style="color: #c00000;">**0.048**</span> → 0.090 | <span style="color: #c00000;">**3.89**</span> → 5.03 | <span style="color: #c00000;">**0.107**</span> → 0.110 | 30.0 → <span style="color: #c00000;">**8.3**</span> | 87.3 → <span style="color: #c00000;">**102.7**</span> |
| Half-bound A / -1.5 | 0.126 → <span style="color: #c00000;">**0.079**</span> | 0.090 → <span style="color: #c00000;">**0.042**</span> | 8.37 → <span style="color: #c00000;">**5.26**</span> | 0.151 → <span style="color: #c00000;">**0.070**</span> | 118.8 → <span style="color: #c00000;">**15.5**</span> | -20.9 → <span style="color: #c00000;">**93.0**</span> |
| Half-bound B / +1.5 | 0.073 → <span style="color: #c00000;">**0.052**</span> | <span style="color: #c00000;">**0.061**</span> → 0.076 | 4.86 → <span style="color: #c00000;">**3.47**</span> | 0.128 → <span style="color: #c00000;">**0.071**</span> | 24.0 → <span style="color: #c00000;">**2.2**</span> | 91.5 → <span style="color: #c00000;">**98.7**</span> |
| Half-bound B / -1.5 | <span style="color: #c00000;">**0.094**</span> → 0.150 | <span style="color: #c00000;">**0.038**</span> → 0.073 | <span style="color: #c00000;">**6.27**</span> → 10.02 | 0.130 → <span style="color: #c00000;">**0.096**</span> | 28.1 → <span style="color: #c00000;">**16.9**</span> | <span style="color: #c00000;">**87.0**</span> → 85.7 |
| Rotary A / +1.5 | 0.080 → <span style="color: #c00000;">**0.076**</span> | <span style="color: #c00000;">**0.060**</span> → 0.094 | 5.33 → <span style="color: #c00000;">**5.04**</span> | 0.120 → <span style="color: #c00000;">**0.092**</span> | 73.9 → <span style="color: #c00000;">**60.9**</span> | 29.7 → <span style="color: #c00000;">**53.5**</span> |
| Rotary A / -1.5 | 0.100 → <span style="color: #c00000;">**0.083**</span> | 0.084 → <span style="color: #c00000;">**0.074**</span> | 6.65 → <span style="color: #c00000;">**5.51**</span> | 0.105 → <span style="color: #c00000;">**0.081**</span> | 61.6 → <span style="color: #c00000;">**28.9**</span> | 50.8 → <span style="color: #c00000;">**87.3**</span> |
| Rotary B / +1.5 | <span style="color: #c00000;">**0.072**</span> → 0.106 | <span style="color: #c00000;">**0.070**</span> → 0.076 | <span style="color: #c00000;">**4.78**</span> → 7.05 | 0.143 → <span style="color: #c00000;">**0.078**</span> | 78.1 → <span style="color: #c00000;">**25.6**</span> | 28.1 → <span style="color: #c00000;">**86.7**</span> |
| Rotary B / -1.5 | 0.145 → <span style="color: #c00000;">**0.086**</span> | <span style="color: #c00000;">**0.062**</span> → 0.084 | 9.66 → <span style="color: #c00000;">**5.74**</span> | <span style="color: #c00000;">**0.114**</span> → 0.115 | 42.8 → <span style="color: #c00000;">**4.4**</span> | 69.9 → <span style="color: #c00000;">**95.3**</span> |
| Transverse A / +1.5 | 0.093 → 0.093 | <span style="color: #c00000;">**0.069**</span> → 0.115 | 6.20 → <span style="color: #c00000;">**6.19**</span> | 0.079 → <span style="color: #c00000;">**0.067**</span> | <span style="color: #c00000;">**2.1**</span> → 20.0 | 95.1 → <span style="color: #c00000;">**95.9**</span> |
| Transverse A / -1.5 | 0.096 → <span style="color: #c00000;">**0.088**</span> | 0.062 → <span style="color: #c00000;">**0.031**</span> | 6.41 → <span style="color: #c00000;">**5.84**</span> | 0.088 → <span style="color: #c00000;">**0.055**</span> | 73.2 → <span style="color: #c00000;">**24.7**</span> | 34.5 → <span style="color: #c00000;">**87.5**</span> |
| Transverse B / +1.5 | 0.105 → <span style="color: #c00000;">**0.079**</span> | 0.080 → <span style="color: #c00000;">**0.064**</span> | 6.98 → <span style="color: #c00000;">**5.27**</span> | <span style="color: #c00000;">**0.066**</span> → 0.099 | <span style="color: #c00000;">**9.0**</span> → 17.4 | 104.3 → <span style="color: #c00000;">**97.6**</span> |
| Transverse B / -1.5 | <span style="color: #c00000;">**0.092**</span> → 0.106 | 0.051 → <span style="color: #c00000;">**0.037**</span> | <span style="color: #c00000;">**6.15**</span> → 7.04 | 0.138 → <span style="color: #c00000;">**0.086**</span> | <span style="color: #c00000;">**18.3**</span> → 43.1 | <span style="color: #c00000;">**92.6**</span> → 69.3 |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 92.0 → <span style="color: #c00000;">**95.6**</span> | 18.0 → <span style="color: #c00000;">**13.0**</span> | 46.67 → <span style="color: #c00000;">**50.03**</span> | 9.33 → <span style="color: #c00000;">**5.97**</span> | 0.00 → 0.00 | 14.06 → <span style="color: #c00000;">**7.76**</span> | <span style="color: #c00000;">**1.00**</span> → 1.03 |
| Trot / -1.5 | 94.6 → 94.6 | <span style="color: #c00000;">**15.7**</span> → 20.0 | <span style="color: #c00000;">**49.14**</span> → 48.33 | <span style="color: #c00000;">**6.86**</span> → 7.67 | 0.00 → 0.00 | 9.42 → <span style="color: #c00000;">**9.38**</span> | 1.00 → 1.00 |
| Bound / +1.5 | 82.8 → <span style="color: #c00000;">**83.8**</span> | 65.3 → <span style="color: #c00000;">**64.2**</span> | 49.01 → 49.01 | <span style="color: #c00000;">**6.99**</span> → 7.27 | 10.55 → <span style="color: #c00000;">**10.35**</span> | 22.27 → <span style="color: #c00000;">**20.65**</span> | <span style="color: #c00000;">**1.06**</span> → 1.12 |
| Bound / -1.5 | <span style="color: #c00000;">**83.0**</span> → 80.1 | <span style="color: #c00000;">**60.0**</span> → 66.8 | <span style="color: #c00000;">**49.17**</span> → 46.38 | <span style="color: #c00000;">**6.83**</span> → 9.62 | 10.42 → 10.42 | <span style="color: #c00000;">**21.88**</span> → 27.10 | 1.00 → 1.00 |
| Half-bound A / +1.5 | <span style="color: #c00000;">**89.1**</span> → 88.0 | <span style="color: #c00000;">**26.3**</span> → 31.0 | 49.14 → <span style="color: #c00000;">**50.44**</span> | 7.19 → <span style="color: #c00000;">**5.89**</span> | <span style="color: #c00000;">**4.36**</span> → 6.25 | <span style="color: #c00000;">**15.82**</span> → 16.31 | 2.22 → <span style="color: #c00000;">**2.00**</span> |
| Half-bound A / -1.5 | <span style="color: #c00000;">**90.2**</span> → 88.4 | <span style="color: #c00000;">**40.0**</span> → 49.4 | <span style="color: #c00000;">**49.17**</span> → 47.50 | <span style="color: #c00000;">**6.83**</span> → 8.50 | 4.17 → 4.17 | <span style="color: #c00000;">**14.06**</span> → 17.19 | 1.00 → 1.00 |
| Half-bound B / +1.5 | 84.3 → <span style="color: #c00000;">**89.2**</span> | 38.3 → <span style="color: #c00000;">**36.0**</span> | <span style="color: #c00000;">**47.84**</span> → 46.57 | <span style="color: #c00000;">**8.16**</span> → 9.77 | 8.33 → <span style="color: #c00000;">**2.08**</span> | 21.24 → <span style="color: #c00000;">**17.38**</span> | <span style="color: #c00000;">**1.97**</span> → 2.07 |
| Half-bound B / -1.5 | 83.0 → <span style="color: #c00000;">**89.1**</span> | <span style="color: #c00000;">**31.5**</span> → 38.1 | 46.12 → <span style="color: #c00000;">**46.89**</span> | 9.88 → <span style="color: #c00000;">**9.11**</span> | 8.33 → <span style="color: #c00000;">**2.21**</span> | 23.44 → <span style="color: #c00000;">**17.38**</span> | 1.00 → 1.00 |
| Rotary A / +1.5 | 66.3 → <span style="color: #c00000;">**80.9**</span> | <span style="color: #c00000;">**34.8**</span> → 37.2 | 50.21 → <span style="color: #c00000;">**51.17**</span> | 7.86 → <span style="color: #c00000;">**5.41**</span> | 31.25 → <span style="color: #c00000;">**16.30**</span> | 35.55 → <span style="color: #c00000;">**21.19**</span> | <span style="color: #c00000;">**1.50**</span> → 2.00 |
| Rotary A / -1.5 | <span style="color: #c00000;">**73.2**</span> → 69.1 | <span style="color: #c00000;">**52.7**</span> → 63.6 | <span style="color: #c00000;">**46.67**</span> → 46.13 | <span style="color: #c00000;">**9.33**</span> → 9.87 | <span style="color: #c00000;">**20.83**</span> → 25.00 | <span style="color: #c00000;">**31.25**</span> → 35.38 | 1.00 → 1.00 |
| Rotary B / +1.5 | <span style="color: #c00000;">**80.4**</span> → 79.0 | <span style="color: #c00000;">**49.8**</span> → 57.0 | <span style="color: #c00000;">**51.63**</span> → 51.25 | 4.77 → <span style="color: #c00000;">**4.75**</span> | <span style="color: #c00000;">**17.18**</span> → 18.75 | <span style="color: #c00000;">**21.45**</span> → 22.66 | 2.09 → <span style="color: #c00000;">**2.00**</span> |
| Rotary B / -1.5 | <span style="color: #c00000;">**77.3**</span> → 75.0 | <span style="color: #c00000;">**46.1**</span> → 49.1 | <span style="color: #c00000;">**50.65**</span> → 48.31 | <span style="color: #c00000;">**5.35**</span> → 7.69 | <span style="color: #c00000;">**20.04**</span> → 20.83 | <span style="color: #c00000;">**24.75**</span> → 28.17 | <span style="color: #c00000;">**1.00**</span> → 1.97 |
| Transverse A / +1.5 | 89.1 → <span style="color: #c00000;">**92.7**</span> | <span style="color: #c00000;">**37.0**</span> → 43.7 | 46.53 → <span style="color: #c00000;">**50.05**</span> | 9.47 → <span style="color: #c00000;">**6.08**</span> | 2.08 → 2.08 | 17.44 → <span style="color: #c00000;">**11.14**</span> | 1.76 → <span style="color: #c00000;">**1.13**</span> |
| Transverse A / -1.5 | <span style="color: #c00000;">**92.9**</span> → 92.0 | <span style="color: #c00000;">**26.0**</span> → 36.0 | <span style="color: #c00000;">**47.50**</span> → 46.67 | <span style="color: #c00000;">**8.50**</span> → 9.33 | 0.00 → 0.00 | <span style="color: #c00000;">**12.50**</span> → 14.06 | 1.00 → 1.00 |
| Transverse B / +1.5 | 90.2 → <span style="color: #c00000;">**91.1**</span> | 47.0 → <span style="color: #c00000;">**31.5**</span> | 46.99 → <span style="color: #c00000;">**47.33**</span> | 9.01 → <span style="color: #c00000;">**8.70**</span> | <span style="color: #c00000;">**0.00**</span> → 1.65 | 17.19 → <span style="color: #c00000;">**14.29**</span> | <span style="color: #c00000;">**1.00**</span> → 2.27 |
| Transverse B / -1.5 | 90.2 → <span style="color: #c00000;">**91.1**</span> | 32.0 → <span style="color: #c00000;">**31.0**</span> | <span style="color: #c00000;">**47.50**</span> → 45.84 | <span style="color: #c00000;">**8.50**</span> → 10.16 | 2.08 → <span style="color: #c00000;">**0.00**</span> | 15.62 → 15.62 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | <span style="color: #c00000;">**151.8**</span> → 157.0 | 1.013 → <span style="color: #c00000;">**0.963**</span> | <span style="color: #c00000;">**2.50**</span> → 3.33 | <span style="color: #c00000;">**4.34**</span> → 8.22 | 4.15 → <span style="color: #c00000;">**3.81**</span> | 16.21 → <span style="color: #c00000;">**9.00**</span> |
| Trot / -1.5 | 136.0 → <span style="color: #c00000;">**135.8**</span> | <span style="color: #c00000;">**0.758**</span> → 0.881 | 3.89 → <span style="color: #c00000;">**0.00**</span> | 5.76 → <span style="color: #c00000;">**4.30**</span> | 7.39 → <span style="color: #c00000;">**5.37**</span> | <span style="color: #c00000;">**3.15**</span> → 10.22 |
| Bound / +1.5 | <span style="color: #c00000;">**283.3**</span> → 327.5 | 1.554 → <span style="color: #c00000;">**1.384**</span> | 4.70 → <span style="color: #c00000;">**4.18**</span> | <span style="color: #c00000;">**20.58**</span> → 25.90 | <span style="color: #c00000;">**12.11**</span> → 17.84 | 9.41 → <span style="color: #c00000;">**6.48**</span> |
| Bound / -1.5 | 245.8 → <span style="color: #c00000;">**230.7**</span> | <span style="color: #c00000;">**1.036**</span> → 1.156 | 6.39 → <span style="color: #c00000;">**3.89**</span> | 8.88 → <span style="color: #c00000;">**1.88**</span> | 5.19 → <span style="color: #c00000;">**3.75**</span> | 8.38 → <span style="color: #c00000;">**2.09**</span> |
| Half-bound A / +1.5 | <span style="color: #c00000;">**221.1**</span> → 231.5 | 1.213 → <span style="color: #c00000;">**1.137**</span> | <span style="color: #c00000;">**3.59**</span> → 3.89 | <span style="color: #c00000;">**3.82**</span> → 17.75 | <span style="color: #c00000;">**6.73**</span> → 9.27 | 22.12 → <span style="color: #c00000;">**1.86**</span> |
| Half-bound A / -1.5 | 224.0 → <span style="color: #c00000;">**199.0**</span> | <span style="color: #c00000;">**1.000**</span> → 1.096 | 5.28 → <span style="color: #c00000;">**2.50**</span> | 15.51 → <span style="color: #c00000;">**1.36**</span> | <span style="color: #c00000;">**5.51**</span> → 8.61 | 5.14 → <span style="color: #c00000;">**2.90**</span> |
| Half-bound B / +1.5 | <span style="color: #c00000;">**214.8**</span> → 253.1 | 1.349 → <span style="color: #c00000;">**1.241**</span> | 5.00 → <span style="color: #c00000;">**3.33**</span> | <span style="color: #c00000;">**10.85**</span> → 12.99 | <span style="color: #c00000;">**7.95**</span> → 13.77 | <span style="color: #c00000;">**9.11**</span> → 11.63 |
| Half-bound B / -1.5 | <span style="color: #c00000;">**177.2**</span> → 252.7 | <span style="color: #c00000;">**0.940**</span> → 1.051 | <span style="color: #c00000;">**3.61**</span> → 3.98 | <span style="color: #c00000;">**2.49**</span> → 11.68 | <span style="color: #c00000;">**11.17**</span> → 13.46 | 7.42 → <span style="color: #c00000;">**2.82**</span> |
| Rotary A / +1.5 | <span style="color: #c00000;">**209.6**</span> → 245.0 | 1.249 → <span style="color: #c00000;">**1.202**</span> | <span style="color: #c00000;">**2.78**</span> → 3.17 | <span style="color: #c00000;">**0.44**</span> → 13.84 | <span style="color: #c00000;">**0.02**</span> → 8.77 | 13.90 → <span style="color: #c00000;">**4.06**</span> |
| Rotary A / -1.5 | 258.4 → <span style="color: #c00000;">**229.0**</span> | <span style="color: #c00000;">**1.103**</span> → 1.191 | 6.39 → <span style="color: #c00000;">**3.06**</span> | 5.01 → <span style="color: #c00000;">**3.86**</span> | 2.17 → <span style="color: #c00000;">**2.04**</span> | 4.80 → <span style="color: #c00000;">**2.58**</span> |
| Rotary B / +1.5 | 339.4 → <span style="color: #c00000;">**222.8**</span> | 1.309 → <span style="color: #c00000;">**1.117**</span> | 4.00 → <span style="color: #c00000;">**3.89**</span> | 35.66 → <span style="color: #c00000;">**27.31**</span> | 6.51 → <span style="color: #c00000;">**0.97**</span> | <span style="color: #c00000;">**0.43**</span> → 13.90 |
| Rotary B / -1.5 | 245.1 → <span style="color: #c00000;">**221.7**</span> | <span style="color: #c00000;">**1.070**</span> → 1.211 | 5.83 → <span style="color: #c00000;">**3.61**</span> | 14.70 → <span style="color: #c00000;">**0.70**</span> | <span style="color: #c00000;">**5.31**</span> → 6.17 | 7.22 → <span style="color: #c00000;">**2.27**</span> |
| Transverse A / +1.5 | <span style="color: #c00000;">**156.3**</span> → 189.2 | <span style="color: #c00000;">**0.995**</span> → 1.066 | 1.99 → <span style="color: #c00000;">**1.90**</span> | <span style="color: #c00000;">**2.21**</span> → 11.09 | 1.24 → <span style="color: #c00000;">**0.40**</span> | 13.62 → <span style="color: #c00000;">**6.45**</span> |
| Transverse A / -1.5 | <span style="color: #c00000;">**177.3**</span> → 177.9 | <span style="color: #c00000;">**0.795**</span> → 0.955 | 3.33 → <span style="color: #c00000;">**0.83**</span> | 7.62 → <span style="color: #c00000;">**0.90**</span> | <span style="color: #c00000;">**6.64**</span> → 8.94 | 1.30 → <span style="color: #c00000;">**1.06**</span> |
| Transverse B / +1.5 | <span style="color: #c00000;">**171.4**</span> → 218.0 | 1.133 → <span style="color: #c00000;">**1.093**</span> | 4.44 → <span style="color: #c00000;">**0.56**</span> | 10.20 → <span style="color: #c00000;">**6.81**</span> | <span style="color: #c00000;">**1.65**</span> → 2.62 | 19.43 → <span style="color: #c00000;">**16.70**</span> |
| Transverse B / -1.5 | 177.0 → <span style="color: #c00000;">**164.2**</span> | 0.930 → <span style="color: #c00000;">**0.799**</span> | 3.06 → <span style="color: #c00000;">**1.39**</span> | <span style="color: #c00000;">**4.47**</span> → 8.09 | <span style="color: #c00000;">**2.62**</span> → 12.70 | 4.47 → <span style="color: #c00000;">**1.93**</span> |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-x1-go2bend-2"></a>

#### X1_go2bend, vx = ±2 m/s

<details>
<summary>Show all metrics for X1_go2bend at ±2 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 0.088 → <span style="color: #c00000;">**0.075**</span> | 0.065 → <span style="color: #c00000;">**0.058**</span> | 4.40 → <span style="color: #c00000;">**3.76**</span> | 0.082 → <span style="color: #c00000;">**0.074**</span> | 4.7 → <span style="color: #c00000;">**3.5**</span> | 96.8 → <span style="color: #c00000;">**98.6**</span> |
| Trot / -2 | 0.358 → <span style="color: #c00000;">**0.247**</span> | 0.079 → <span style="color: #c00000;">**0.044**</span> | 17.91 → <span style="color: #c00000;">**12.36**</span> | 0.104 → <span style="color: #c00000;">**0.068**</span> | 62.3 → <span style="color: #c00000;">**3.6**</span> | 42.9 → <span style="color: #c00000;">**88.3**</span> |
| Bound / +2 | <span style="color: #c00000;">**0.078**</span> → 0.080 | 0.079 → <span style="color: #c00000;">**0.063**</span> | <span style="color: #c00000;">**3.91**</span> → 4.02 | 0.129 → <span style="color: #c00000;">**0.103**</span> | 11.3 → <span style="color: #c00000;">**3.4**</span> | 96.9 → <span style="color: #c00000;">**100.9**</span> |
| Bound / -2 | 0.375 → <span style="color: #c00000;">**0.229**</span> | 0.167 → <span style="color: #c00000;">**0.116**</span> | 18.73 → <span style="color: #c00000;">**11.44**</span> | <span style="color: #c00000;">**0.157**</span> → 0.172 | <span style="color: #c00000;">**74.0**</span> → 80.7 | <span style="color: #c00000;">**32.8**</span> → 23.6 |
| Half-bound A / +2 | <span style="color: #c00000;">**0.078**</span> → 0.151 | <span style="color: #c00000;">**0.105**</span> → 0.130 | <span style="color: #c00000;">**3.89**</span> → 7.54 | <span style="color: #c00000;">**0.118**</span> → 0.123 | 19.8 → <span style="color: #c00000;">**18.6**</span> | 95.4 → <span style="color: #c00000;">**101.2**</span> |
| Half-bound A / -2 | 0.251 → <span style="color: #c00000;">**0.188**</span> | 0.087 → <span style="color: #c00000;">**0.063**</span> | 12.53 → <span style="color: #c00000;">**9.39**</span> | 0.131 → <span style="color: #c00000;">**0.086**</span> | 74.0 → <span style="color: #c00000;">**9.4**</span> | 31.4 → <span style="color: #c00000;">**90.4**</span> |
| Half-bound B / +2 | 0.152 → <span style="color: #c00000;">**0.080**</span> | 0.094 → <span style="color: #c00000;">**0.077**</span> | 7.62 → <span style="color: #c00000;">**4.00**</span> | 0.153 → <span style="color: #c00000;">**0.084**</span> | <span style="color: #c00000;">**27.9**</span> → 28.6 | <span style="color: #c00000;">**91.7**</span> → 88.4 |
| Half-bound B / -2 | 0.267 → <span style="color: #c00000;">**0.262**</span> | <span style="color: #c00000;">**0.055**</span> → 0.083 | 13.33 → <span style="color: #c00000;">**13.11**</span> | 0.149 → <span style="color: #c00000;">**0.109**</span> | <span style="color: #c00000;">**2.3**</span> → 59.1 | <span style="color: #c00000;">**87.6**</span> → 44.6 |
| Rotary A / +2 | 0.091 → <span style="color: #c00000;">**0.079**</span> | 0.113 → <span style="color: #c00000;">**0.050**</span> | 4.54 → <span style="color: #c00000;">**3.94**</span> | 0.130 → <span style="color: #c00000;">**0.071**</span> | 76.8 → <span style="color: #c00000;">**7.3**</span> | 24.3 → <span style="color: #c00000;">**99.4**</span> |
| Rotary A / -2 | 0.260 → <span style="color: #c00000;">**0.182**</span> | 0.148 → <span style="color: #c00000;">**0.081**</span> | 12.99 → <span style="color: #c00000;">**9.09**</span> | 0.165 → <span style="color: #c00000;">**0.109**</span> | <span style="color: #c00000;">**22.6**</span> → 69.7 | <span style="color: #c00000;">**78.0**</span> → 37.2 |
| Rotary B / +2 | <span style="color: #c00000;">**0.087**</span> → 0.111 | <span style="color: #c00000;">**0.063**</span> → 0.119 | <span style="color: #c00000;">**4.36**</span> → 5.54 | 0.136 → <span style="color: #c00000;">**0.105**</span> | <span style="color: #c00000;">**13.3**</span> → 20.3 | 96.2 → <span style="color: #c00000;">**97.4**</span> |
| Rotary B / -2 | 0.328 → <span style="color: #c00000;">**0.201**</span> | 0.108 → <span style="color: #c00000;">**0.094**</span> | 16.41 → <span style="color: #c00000;">**10.06**</span> | 0.144 → <span style="color: #c00000;">**0.143**</span> | 42.7 → <span style="color: #c00000;">**19.7**</span> | 64.9 → <span style="color: #c00000;">**85.4**</span> |
| Transverse A / +2 | <span style="color: #c00000;">**0.079**</span> → 0.096 | 0.122 → <span style="color: #c00000;">**0.101**</span> | <span style="color: #c00000;">**3.93**</span> → 4.80 | <span style="color: #c00000;">**0.136**</span> → 0.174 | <span style="color: #c00000;">**82.0**</span> → 96.7 | <span style="color: #c00000;">**14.4**</span> → -0.7 |
| Transverse A / -2 | <span style="color: #c00000;">**0.195**</span> → 0.203 | 0.093 → <span style="color: #c00000;">**0.045**</span> | <span style="color: #c00000;">**9.74**</span> → 10.13 | 0.139 → <span style="color: #c00000;">**0.062**</span> | 108.0 → <span style="color: #c00000;">**4.8**</span> | -16.2 → <span style="color: #c00000;">**90.3**</span> |
| Transverse B / +2 | 0.146 → <span style="color: #c00000;">**0.077**</span> | 0.141 → <span style="color: #c00000;">**0.086**</span> | 7.32 → <span style="color: #c00000;">**3.83**</span> | <span style="color: #c00000;">**0.101**</span> → 0.106 | <span style="color: #c00000;">**28.1**</span> → 53.9 | <span style="color: #c00000;">**90.0**</span> → 63.1 |
| Transverse B / -2 | 0.265 → <span style="color: #c00000;">**0.216**</span> | 0.090 → <span style="color: #c00000;">**0.051**</span> | 13.23 → <span style="color: #c00000;">**10.78**</span> | 0.171 → <span style="color: #c00000;">**0.096**</span> | <span style="color: #c00000;">**20.7**</span> → 51.6 | <span style="color: #c00000;">**80.5**</span> → 55.7 |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 90.5 → <span style="color: #c00000;">**92.6**</span> | 23.6 → <span style="color: #c00000;">**16.3**</span> | 44.51 → <span style="color: #c00000;">**46.41**</span> | 11.49 → <span style="color: #c00000;">**9.59**</span> | 0.00 → 0.00 | 16.55 → <span style="color: #c00000;">**12.99**</span> | <span style="color: #c00000;">**1.50**</span> → 1.88 |
| Trot / -2 | 88.5 → <span style="color: #c00000;">**91.1**</span> | 20.7 → <span style="color: #c00000;">**20.0**</span> | 43.46 → <span style="color: #c00000;">**45.42**</span> | 12.54 → <span style="color: #c00000;">**10.58**</span> | 0.00 → 0.00 | 20.07 → <span style="color: #c00000;">**15.62**</span> | 1.00 → 1.00 |
| Bound / +2 | <span style="color: #c00000;">**74.2**</span> → 73.3 | <span style="color: #c00000;">**75.9**</span> → 87.8 | <span style="color: #c00000;">**43.70**</span> → 42.79 | <span style="color: #c00000;">**12.30**</span> → 13.21 | <span style="color: #c00000;">**13.87**</span> → 14.84 | <span style="color: #c00000;">**34.72**</span> → 35.55 | <span style="color: #c00000;">**2.00**</span> → 2.09 |
| Bound / -2 | <span style="color: #c00000;">**77.1**</span> → 75.9 | 75.1 → <span style="color: #c00000;">**62.3**</span> | <span style="color: #c00000;">**45.31**</span> → 42.45 | <span style="color: #c00000;">**10.69**</span> → 13.55 | 12.50 → <span style="color: #c00000;">**10.29**</span> | <span style="color: #c00000;">**30.66**</span> → 34.38 | 2.00 → <span style="color: #c00000;">**1.03**</span> |
| Half-bound A / +2 | <span style="color: #c00000;">**81.8**</span> → 79.7 | <span style="color: #c00000;">**35.3**</span> → 44.6 | 43.20 → <span style="color: #c00000;">**44.52**</span> | 12.80 → <span style="color: #c00000;">**11.48**</span> | <span style="color: #c00000;">**5.21**</span> → 9.58 | <span style="color: #c00000;">**28.03**</span> → 28.40 | 2.00 → 2.00 |
| Half-bound A / -2 | <span style="color: #c00000;">**82.6**</span> → 81.7 | 54.5 → <span style="color: #c00000;">**40.0**</span> | <span style="color: #c00000;">**43.75**</span> → 42.92 | <span style="color: #c00000;">**12.25**</span> → 13.08 | 6.25 → 6.25 | <span style="color: #c00000;">**25.78**</span> → 27.34 | 1.00 → 1.00 |
| Half-bound B / +2 | 75.3 → <span style="color: #c00000;">**81.6**</span> | <span style="color: #c00000;">**50.2**</span> → 57.4 | 42.29 → <span style="color: #c00000;">**43.94**</span> | 13.71 → <span style="color: #c00000;">**12.06**</span> | 11.07 → <span style="color: #c00000;">**6.65**</span> | 34.91 → <span style="color: #c00000;">**27.28**</span> | 2.09 → <span style="color: #c00000;">**2.06**</span> |
| Half-bound B / -2 | 75.0 → <span style="color: #c00000;">**79.1**</span> | <span style="color: #c00000;">**48.0**</span> → 58.1 | <span style="color: #c00000;">**42.50**</span> → 42.03 | <span style="color: #c00000;">**13.50**</span> → 13.97 | 12.50 → <span style="color: #c00000;">**8.23**</span> | 34.38 → <span style="color: #c00000;">**30.49**</span> | 1.00 → 1.00 |
| Rotary A / +2 | 60.5 → <span style="color: #c00000;">**70.7**</span> | <span style="color: #c00000;">**19.9**</span> → 43.8 | 42.44 → <span style="color: #c00000;">**44.95**</span> | 13.56 → <span style="color: #c00000;">**11.05**</span> | 28.14 → <span style="color: #c00000;">**20.54**</span> | 48.03 → <span style="color: #c00000;">**35.90**</span> | <span style="color: #c00000;">**2.01**</span> → 2.08 |
| Rotary A / -2 | <span style="color: #c00000;">**65.5**</span> → 65.4 | <span style="color: #c00000;">**42.8**</span> → 66.4 | 41.12 → <span style="color: #c00000;">**42.78**</span> | 15.21 → <span style="color: #c00000;">**13.22**</span> | <span style="color: #c00000;">**22.92**</span> → 25.00 | 43.21 → <span style="color: #c00000;">**41.71**</span> | 2.00 → <span style="color: #c00000;">**1.29**</span> |
| Rotary B / +2 | <span style="color: #c00000;">**73.2**</span> → 68.5 | <span style="color: #c00000;">**62.0**</span> → 75.8 | <span style="color: #c00000;">**46.17**</span> → 44.73 | <span style="color: #c00000;">**9.83**</span> → 11.27 | <span style="color: #c00000;">**20.26**</span> → 23.82 | <span style="color: #c00000;">**31.76**</span> → 37.24 | 2.00 → 2.00 |
| Rotary B / -2 | <span style="color: #c00000;">**70.8**</span> → 68.2 | <span style="color: #c00000;">**2.0**</span> → 28.8 | <span style="color: #c00000;">**46.08**</span> → 43.39 | <span style="color: #c00000;">**9.92**</span> → 12.61 | <span style="color: #c00000;">**20.83**</span> → 22.43 | <span style="color: #c00000;">**35.48**</span> → 38.79 | <span style="color: #c00000;">**1.00**</span> → 1.24 |
| Transverse A / +2 | 84.8 → 84.8 | <span style="color: #c00000;">**44.5**</span> → 48.9 | <span style="color: #c00000;">**42.49**</span> → 42.39 | <span style="color: #c00000;">**13.51**</span> → 13.61 | <span style="color: #c00000;">**2.08**</span> → 2.32 | 25.04 → <span style="color: #c00000;">**24.89**</span> | <span style="color: #c00000;">**1.85**</span> → 2.47 |
| Transverse A / -2 | 86.6 → <span style="color: #c00000;">**87.8**</span> | 37.0 → <span style="color: #c00000;">**30.3**</span> | <span style="color: #c00000;">**44.17**</span> → 43.62 | <span style="color: #c00000;">**11.83**</span> → 12.38 | 2.08 → <span style="color: #c00000;">**0.00**</span> | 21.88 → <span style="color: #c00000;">**21.34**</span> | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / +2 | 85.3 → <span style="color: #c00000;">**87.3**</span> | 55.7 → <span style="color: #c00000;">**31.0**</span> | 43.01 → <span style="color: #c00000;">**43.12**</span> | 12.99 → <span style="color: #c00000;">**12.88**</span> | 0.48 → <span style="color: #c00000;">**0.00**</span> | 25.33 → <span style="color: #c00000;">**22.27**</span> | <span style="color: #c00000;">**1.84**</span> → 2.00 |
| Transverse B / -2 | <span style="color: #c00000;">**83.9**</span> → 83.2 | 42.0 → <span style="color: #c00000;">**37.7**</span> | <span style="color: #c00000;">**41.67**</span> → 41.02 | <span style="color: #c00000;">**14.33**</span> → 14.98 | 2.08 → 2.08 | <span style="color: #c00000;">**26.56**</span> → 27.77 | 2.00 → <span style="color: #c00000;">**1.00**</span> |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | <span style="color: #c00000;">**235.7**</span> → 266.1 | <span style="color: #c00000;">**1.197**</span> → 1.246 | 3.19 → <span style="color: #c00000;">**3.14**</span> | <span style="color: #c00000;">**3.19**</span> → 15.89 | 7.92 → <span style="color: #c00000;">**7.75**</span> | 11.72 → <span style="color: #c00000;">**4.96**</span> |
| Trot / -2 | 199.3 → <span style="color: #c00000;">**186.3**</span> | <span style="color: #c00000;">**0.843**</span> → 0.933 | 3.89 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**0.29**</span> → 2.65 | <span style="color: #c00000;">**0.67**</span> → 5.58 | <span style="color: #c00000;">**8.52**</span> → 12.36 |
| Bound / +2 | <span style="color: #c00000;">**352.9**</span> → 354.5 | 1.704 → <span style="color: #c00000;">**1.574**</span> | 4.32 → <span style="color: #c00000;">**4.17**</span> | <span style="color: #c00000;">**7.42**</span> → 9.35 | <span style="color: #c00000;">**0.17**</span> → 2.12 | 8.38 → <span style="color: #c00000;">**8.21**</span> |
| Bound / -2 | <span style="color: #c00000;">**281.5**</span> → 296.4 | <span style="color: #c00000;">**1.147**</span> → 1.324 | 6.45 → <span style="color: #c00000;">**4.05**</span> | <span style="color: #c00000;">**8.98**</span> → 13.16 | 6.15 → <span style="color: #c00000;">**1.63**</span> | 10.73 → <span style="color: #c00000;">**2.54**</span> |
| Half-bound A / +2 | <span style="color: #c00000;">**352.0**</span> → 372.7 | 1.520 → <span style="color: #c00000;">**1.505**</span> | 4.24 → <span style="color: #c00000;">**3.23**</span> | <span style="color: #c00000;">**0.46**</span> → 8.89 | <span style="color: #c00000;">**2.23**</span> → 6.92 | 12.88 → <span style="color: #c00000;">**3.41**</span> |
| Half-bound A / -2 | <span style="color: #c00000;">**250.3**</span> → 252.9 | <span style="color: #c00000;">**1.040**</span> → 1.281 | 4.72 → <span style="color: #c00000;">**3.61**</span> | <span style="color: #c00000;">**9.22**</span> → 9.96 | 7.12 → <span style="color: #c00000;">**2.37**</span> | <span style="color: #c00000;">**2.86**</span> → 3.82 |
| Half-bound B / +2 | 328.1 → <span style="color: #c00000;">**301.0**</span> | 1.613 → <span style="color: #c00000;">**1.313**</span> | 5.73 → <span style="color: #c00000;">**2.90**</span> | <span style="color: #c00000;">**0.04**</span> → 4.17 | 5.00 → <span style="color: #c00000;">**2.60**</span> | <span style="color: #c00000;">**4.63**</span> → 15.26 |
| Half-bound B / -2 | <span style="color: #c00000;">**263.5**</span> → 278.7 | 1.095 → <span style="color: #c00000;">**1.043**</span> | <span style="color: #c00000;">**4.17**</span> → 4.55 | 15.77 → <span style="color: #c00000;">**1.07**</span> | <span style="color: #c00000;">**0.67**</span> → 8.01 | 3.49 → <span style="color: #c00000;">**3.13**</span> |
| Rotary A / +2 | 372.2 → <span style="color: #c00000;">**313.7**</span> | 1.591 → <span style="color: #c00000;">**1.339**</span> | 3.46 → <span style="color: #c00000;">**3.41**</span> | 9.15 → <span style="color: #c00000;">**0.34**</span> | 5.81 → <span style="color: #c00000;">**1.24**</span> | <span style="color: #c00000;">**7.95**</span> → 13.24 |
| Rotary A / -2 | 295.2 → <span style="color: #c00000;">**293.9**</span> | 1.297 → <span style="color: #c00000;">**1.249**</span> | 6.91 → <span style="color: #c00000;">**3.52**</span> | <span style="color: #c00000;">**1.77**</span> → 15.88 | 2.26 → <span style="color: #c00000;">**0.93**</span> | 10.57 → <span style="color: #c00000;">**1.65**</span> |
| Rotary B / +2 | 423.1 → <span style="color: #c00000;">**347.0**</span> | 1.649 → <span style="color: #c00000;">**1.516**</span> | 4.73 → <span style="color: #c00000;">**3.61**</span> | 15.06 → <span style="color: #c00000;">**14.83**</span> | <span style="color: #c00000;">**3.31**</span> → 8.88 | <span style="color: #c00000;">**0.69**</span> → 12.25 |
| Rotary B / -2 | 276.8 → <span style="color: #c00000;">**265.7**</span> | <span style="color: #c00000;">**1.129**</span> → 1.369 | 6.39 → <span style="color: #c00000;">**4.60**</span> | <span style="color: #c00000;">**11.42**</span> → 15.71 | 10.00 → <span style="color: #c00000;">**1.15**</span> | 9.67 → <span style="color: #c00000;">**0.31**</span> |
| Transverse A / +2 | <span style="color: #c00000;">**290.0**</span> → 322.0 | <span style="color: #c00000;">**1.247**</span> → 1.311 | 2.80 → <span style="color: #c00000;">**1.92**</span> | 14.73 → <span style="color: #c00000;">**11.79**</span> | <span style="color: #c00000;">**3.07**</span> → 5.87 | 20.60 → <span style="color: #c00000;">**2.29**</span> |
| Transverse A / -2 | 241.4 → <span style="color: #c00000;">**239.3**</span> | <span style="color: #c00000;">**0.942**</span> → 1.080 | 3.61 → <span style="color: #c00000;">**1.67**</span> | 4.57 → <span style="color: #c00000;">**1.41**</span> | 7.27 → <span style="color: #c00000;">**1.14**</span> | <span style="color: #c00000;">**0.16**</span> → 3.88 |
| Transverse B / +2 | <span style="color: #c00000;">**292.2**</span> → 297.9 | 1.431 → <span style="color: #c00000;">**1.260**</span> | 5.39 → <span style="color: #c00000;">**1.12**</span> | 5.08 → <span style="color: #c00000;">**1.40**</span> | <span style="color: #c00000;">**1.55**</span> → 3.60 | <span style="color: #c00000;">**7.40**</span> → 14.62 |
| Transverse B / -2 | 243.1 → <span style="color: #c00000;">**234.4**</span> | 1.051 → <span style="color: #c00000;">**0.882**</span> | 3.61 → <span style="color: #c00000;">**1.14**</span> | <span style="color: #c00000;">**3.77**</span> → 6.29 | <span style="color: #c00000;">**5.38**</span> → 8.39 | 5.74 → <span style="color: #c00000;">**0.41**</span> |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +2 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +2 | <span style="color: #c00000;">**78.0**</span> → 7.0 | <span style="color: #c00000;">**78.0**</span> → 7.0 | 0.0 → 0.0 | <span style="color: #c00000;">**78.0**</span> → 7.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 49.0 | 100.0 → 100.0 |
| Transverse A / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Transverse B / +2 | 19.0 → <span style="color: #c00000;">**94.0**</span> | 19.0 → <span style="color: #c00000;">**94.0**</span> | 0.0 → 0.0 | 19.0 → <span style="color: #c00000;">**94.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -2 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

### Go2: full gait/direction details

<a id="detail-go2-0p5"></a>

#### Go2, vx = ±0.5 m/s

<details>
<summary>Show all metrics for Go2 at ±0.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | <span style="color: #c00000;">**0.032**</span> → 0.044 | 0.039 → <span style="color: #c00000;">**0.035**</span> | <span style="color: #c00000;">**6.41**</span> → 8.79 | <span style="color: #c00000;">**0.074**</span> → 0.078 | 22.7 → <span style="color: #c00000;">**22.0**</span> | <span style="color: #c00000;">**96.6**</span> → 87.1 |
| Trot / -0.5 | 0.065 → <span style="color: #c00000;">**0.018**</span> | 0.068 → <span style="color: #c00000;">**0.055**</span> | 12.92 → <span style="color: #c00000;">**3.51**</span> | <span style="color: #c00000;">**0.046**</span> → 0.063 | <span style="color: #c00000;">**2.5**</span> → 33.6 | <span style="color: #c00000;">**87.3**</span> → 87.1 |
| Bound / +0.5 | <span style="color: #c00000;">**0.037**</span> → 0.050 | <span style="color: #c00000;">**0.031**</span> → 0.035 | <span style="color: #c00000;">**7.49**</span> → 9.90 | <span style="color: #c00000;">**0.042**</span> → 0.074 | <span style="color: #c00000;">**51.0**</span> → 75.6 | <span style="color: #c00000;">**66.4**</span> → 30.4 |
| Bound / -0.5 | <span style="color: #c00000;">**0.045**</span> → 0.052 | 0.033 → <span style="color: #c00000;">**0.032**</span> | <span style="color: #c00000;">**9.07**</span> → 10.32 | <span style="color: #c00000;">**0.068**</span> → 0.071 | 52.6 → <span style="color: #c00000;">**9.9**</span> | 57.6 → <span style="color: #c00000;">**93.3**</span> |
| Half-bound A / +0.5 | 0.034 → 0.034 | 0.065 → <span style="color: #c00000;">**0.053**</span> | <span style="color: #c00000;">**6.72**</span> → 6.79 | <span style="color: #c00000;">**0.048**</span> → 0.064 | <span style="color: #c00000;">**7.1**</span> → 24.8 | <span style="color: #c00000;">**100.4**</span> → 90.8 |
| Half-bound A / -0.5 | <span style="color: #c00000;">**0.039**</span> → 0.083 | 0.071 → <span style="color: #c00000;">**0.063**</span> | <span style="color: #c00000;">**7.83**</span> → 16.56 | 0.068 → <span style="color: #c00000;">**0.057**</span> | 30.9 → <span style="color: #c00000;">**2.4**</span> | <span style="color: #c00000;">**87.3**</span> → 84.1 |
| Half-bound B / +0.5 | <span style="color: #c00000;">**0.041**</span> → 0.043 | 0.065 → <span style="color: #c00000;">**0.056**</span> | <span style="color: #c00000;">**8.30**</span> → 8.61 | 0.087 → <span style="color: #c00000;">**0.048**</span> | 66.7 → <span style="color: #c00000;">**40.2**</span> | 37.0 → <span style="color: #c00000;">**75.5**</span> |
| Half-bound B / -0.5 | <span style="color: #c00000;">**0.039**</span> → 0.059 | <span style="color: #c00000;">**0.047**</span> → 0.065 | <span style="color: #c00000;">**7.76**</span> → 11.83 | 0.052 → 0.052 | 77.1 → <span style="color: #c00000;">**23.2**</span> | 28.4 → <span style="color: #c00000;">**81.7**</span> |
| Rotary A / +0.5 | <span style="color: #c00000;">**0.051**</span> → 0.059 | <span style="color: #c00000;">**0.119**</span> → 0.120 | <span style="color: #c00000;">**10.29**</span> → 11.83 | <span style="color: #c00000;">**0.061**</span> → 0.072 | 52.1 → <span style="color: #c00000;">**13.9**</span> | 57.6 → <span style="color: #c00000;">**88.1**</span> |
| Rotary A / -0.5 | <span style="color: #c00000;">**0.043**</span> → 0.055 | <span style="color: #c00000;">**0.104**</span> → 0.137 | <span style="color: #c00000;">**8.62**</span> → 11.07 | 0.072 → <span style="color: #c00000;">**0.053**</span> | 98.9 → <span style="color: #c00000;">**8.0**</span> | -8.4 → <span style="color: #c00000;">**89.9**</span> |
| Rotary B / +0.5 | <span style="color: #c00000;">**0.042**</span> → 0.056 | 0.126 → <span style="color: #c00000;">**0.123**</span> | <span style="color: #c00000;">**8.34**</span> → 11.19 | 0.091 → <span style="color: #c00000;">**0.062**</span> | 26.9 → <span style="color: #c00000;">**8.5**</span> | 90.4 → <span style="color: #c00000;">**96.1**</span> |
| Rotary B / -0.5 | 0.054 → <span style="color: #c00000;">**0.040**</span> | <span style="color: #c00000;">**0.115**</span> → 0.123 | 10.82 → <span style="color: #c00000;">**7.99**</span> | 0.132 → <span style="color: #c00000;">**0.064**</span> | 101.1 → <span style="color: #c00000;">**2.7**</span> | 0.8 → <span style="color: #c00000;">**98.8**</span> |
| Transverse A / +0.5 | <span style="color: #c00000;">**0.071**</span> → 0.081 | 0.026 → <span style="color: #c00000;">**0.018**</span> | <span style="color: #c00000;">**14.22**</span> → 16.23 | <span style="color: #c00000;">**0.037**</span> → 0.091 | <span style="color: #c00000;">**7.9**</span> → 22.6 | <span style="color: #c00000;">**87.5**</span> → 79.2 |
| Transverse A / -0.5 | <span style="color: #c00000;">**0.032**</span> → 0.051 | 0.056 → <span style="color: #c00000;">**0.029**</span> | <span style="color: #c00000;">**6.32**</span> → 10.12 | <span style="color: #c00000;">**0.038**</span> → 0.060 | <span style="color: #c00000;">**19.3**</span> → 46.6 | <span style="color: #c00000;">**97.5**</span> → 63.6 |
| Transverse B / +0.5 | 0.035 → <span style="color: #c00000;">**0.023**</span> | 0.028 → <span style="color: #c00000;">**0.018**</span> | 7.01 → <span style="color: #c00000;">**4.66**</span> | <span style="color: #c00000;">**0.042**</span> → 0.070 | 29.9 → <span style="color: #c00000;">**3.8**</span> | 83.7 → <span style="color: #c00000;">**100.7**</span> |
| Transverse B / -0.5 | 0.047 → <span style="color: #c00000;">**0.023**</span> | <span style="color: #c00000;">**0.026**</span> → 0.076 | 9.34 → <span style="color: #c00000;">**4.54**</span> | <span style="color: #c00000;">**0.059**</span> → 0.065 | 46.1 → <span style="color: #c00000;">**24.1**</span> | 64.4 → <span style="color: #c00000;">**98.9**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 99.0 → <span style="color: #c00000;">**100.0**</span> | 10.0 → 10.0 | 54.81 → <span style="color: #c00000;">**55.77**</span> | 2.88 → <span style="color: #c00000;">**1.92**</span> | 0.00 → 0.00 | 1.79 → <span style="color: #c00000;">**0.00**</span> | 1.00 → 1.00 |
| Trot / -0.5 | 96.2 → <span style="color: #c00000;">**99.0**</span> | <span style="color: #c00000;">**3.3**</span> → 5.1 | 53.52 → <span style="color: #c00000;">**55.77**</span> | 2.48 → <span style="color: #c00000;">**1.92**</span> | 0.00 → 0.00 | 6.58 → <span style="color: #c00000;">**1.79**</span> | 1.00 → 1.00 |
| Bound / +0.5 | 92.7 → <span style="color: #c00000;">**92.8**</span> | 30.6 → <span style="color: #c00000;">**27.1**</span> | 56.73 → <span style="color: #c00000;">**56.18**</span> | <span style="color: #c00000;">**6.73**</span> → 8.10 | 10.00 → <span style="color: #c00000;">**7.88**</span> | <span style="color: #c00000;">**5.36**</span> → 6.65 | 1.00 → 1.00 |
| Bound / -0.5 | 93.8 → <span style="color: #c00000;">**96.3**</span> | 10.0 → <span style="color: #c00000;">**4.9**</span> | 53.85 → <span style="color: #c00000;">**58.15**</span> | 5.77 → <span style="color: #c00000;">**4.35**</span> | <span style="color: #c00000;">**2.50**</span> → 3.75 | 8.93 → <span style="color: #c00000;">**3.62**</span> | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.8 → <span style="color: #c00000;">**96.9**</span> | 19.3 → <span style="color: #c00000;">**14.4**</span> | 58.65 → 58.65 | 4.81 → 4.81 | 4.88 → 4.88 | 3.64 → <span style="color: #c00000;">**1.82**</span> | 1.00 → 1.00 |
| Half-bound A / -0.5 | 96.9 → 96.9 | 13.8 → <span style="color: #c00000;">**8.8**</span> | <span style="color: #c00000;">**56.73**</span> → 59.62 | 4.81 → <span style="color: #c00000;">**4.69**</span> | <span style="color: #c00000;">**2.44**</span> → 4.88 | 3.64 → <span style="color: #c00000;">**1.82**</span> | 1.00 → 1.00 |
| Half-bound B / +0.5 | 92.7 → <span style="color: #c00000;">**96.9**</span> | 24.4 → <span style="color: #c00000;">**14.4**</span> | 59.62 → <span style="color: #c00000;">**57.69**</span> | 7.69 → <span style="color: #c00000;">**3.85**</span> | 9.76 → <span style="color: #c00000;">**4.88**</span> | 5.45 → <span style="color: #c00000;">**1.82**</span> | 1.00 → 1.00 |
| Half-bound B / -0.5 | <span style="color: #c00000;">**95.8**</span> → 94.8 | <span style="color: #c00000;">**5.0**</span> → 15.0 | 60.55 → <span style="color: #c00000;">**59.59**</span> | 5.68 → <span style="color: #c00000;">**4.67**</span> | 7.32 → 7.32 | <span style="color: #c00000;">**1.87**</span> → 3.68 | 1.00 → 1.00 |
| Rotary A / +0.5 | 96.9 → 96.9 | <span style="color: #c00000;">**10.6**</span> → 18.2 | 60.58 → <span style="color: #c00000;">**59.62**</span> | 5.65 → <span style="color: #c00000;">**4.69**</span> | 7.14 → 7.14 | 0.00 → 0.00 | 1.00 → 1.00 |
| Rotary A / -0.5 | 96.9 → <span style="color: #c00000;">**97.9**</span> | 13.8 → <span style="color: #c00000;">**8.8**</span> | 59.62 → <span style="color: #c00000;">**56.73**</span> | 3.62 → <span style="color: #c00000;">**2.88**</span> | 4.76 → <span style="color: #c00000;">**2.38**</span> | 1.85 → 1.85 | 1.00 → 1.00 |
| Rotary B / +0.5 | <span style="color: #c00000;">**96.9**</span> → 95.8 | <span style="color: #c00000;">**15.0**</span> → 18.2 | <span style="color: #c00000;">**59.62**</span> → 61.54 | 5.77 → <span style="color: #c00000;">**5.54**</span> | <span style="color: #c00000;">**7.14**</span> → 9.52 | 0.00 → 0.00 | 1.00 → 1.00 |
| Rotary B / -0.5 | 95.8 → <span style="color: #c00000;">**96.9**</span> | 17.6 → <span style="color: #c00000;">**12.6**</span> | 58.65 → 58.65 | 3.73 → 3.73 | 4.76 → 4.76 | 3.70 → <span style="color: #c00000;">**1.85**</span> | 1.00 → 1.00 |
| Transverse A / +0.5 | 96.9 → 96.9 | 11.8 → 11.8 | 56.73 → 56.73 | 4.81 → 4.81 | 4.76 → 4.76 | 1.85 → 1.85 | 1.00 → 1.00 |
| Transverse A / -0.5 | 96.9 → 96.9 | 11.2 → <span style="color: #c00000;">**9.4**</span> | <span style="color: #c00000;">**55.77**</span> → 58.17 | <span style="color: #c00000;">**3.85**</span> → 5.29 | 4.76 → 4.76 | 1.85 → 1.85 | 1.00 → 1.00 |
| Transverse B / +0.5 | 94.3 → <span style="color: #c00000;">**94.8**</span> | 14.0 → <span style="color: #c00000;">**12.4**</span> | 58.17 → <span style="color: #c00000;">**57.69**</span> | <span style="color: #c00000;">**4.33**</span> → 5.77 | 8.33 → <span style="color: #c00000;">**7.14**</span> | 3.70 → 3.70 | 1.00 → 1.00 |
| Transverse B / -0.5 | <span style="color: #c00000;">**95.8**</span> → 94.8 | <span style="color: #c00000;">**10.0**</span> → 16.2 | <span style="color: #c00000;">**55.77**</span> → 56.73 | 5.77 → <span style="color: #c00000;">**4.81**</span> | <span style="color: #c00000;">**4.76**</span> → 7.14 | 3.70 → 3.70 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 30.6 → <span style="color: #c00000;">**24.8**</span> | 0.300 → <span style="color: #c00000;">**0.278**</span> | 0.00 → 0.00 | 24.55 → <span style="color: #c00000;">**1.80**</span> | 23.90 → <span style="color: #c00000;">**14.95**</span> | 29.55 → <span style="color: #c00000;">**15.79**</span> |
| Trot / -0.5 | <span style="color: #c00000;">**24.4**</span> → 25.9 | 0.298 → <span style="color: #c00000;">**0.284**</span> | 0.00 → 0.00 | 34.03 → <span style="color: #c00000;">**7.12**</span> | 32.22 → <span style="color: #c00000;">**16.28**</span> | 49.85 → <span style="color: #c00000;">**14.20**</span> |
| Bound / +0.5 | 87.3 → <span style="color: #c00000;">**79.7**</span> | <span style="color: #c00000;">**0.307**</span> → 0.330 | 0.96 → <span style="color: #c00000;">**0.00**</span> | 24.18 → <span style="color: #c00000;">**17.60**</span> | <span style="color: #c00000;">**31.89**</span> → 33.99 | <span style="color: #c00000;">**30.11**</span> → 36.26 |
| Bound / -0.5 | 108.7 → <span style="color: #c00000;">**98.8**</span> | <span style="color: #c00000;">**0.272**</span> → 0.288 | 0.64 → <span style="color: #c00000;">**0.00**</span> | 12.62 → <span style="color: #c00000;">**5.68**</span> | 29.43 → <span style="color: #c00000;">**25.35**</span> | 34.48 → <span style="color: #c00000;">**18.18**</span> |
| Half-bound A / +0.5 | 84.8 → <span style="color: #c00000;">**77.6**</span> | 0.295 → <span style="color: #c00000;">**0.272**</span> | 1.60 → <span style="color: #c00000;">**0.00**</span> | 26.93 → <span style="color: #c00000;">**10.26**</span> | 29.76 → <span style="color: #c00000;">**27.08**</span> | 40.90 → <span style="color: #c00000;">**34.47**</span> |
| Half-bound A / -0.5 | 93.0 → <span style="color: #c00000;">**73.1**</span> | <span style="color: #c00000;">**0.259**</span> → 0.272 | 1.60 → <span style="color: #c00000;">**0.00**</span> | 11.48 → <span style="color: #c00000;">**1.25**</span> | 32.76 → <span style="color: #c00000;">**30.58**</span> | 41.53 → <span style="color: #c00000;">**33.17**</span> |
| Half-bound B / +0.5 | 80.0 → <span style="color: #c00000;">**74.0**</span> | <span style="color: #c00000;">**0.284**</span> → 0.303 | 2.24 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**22.01**</span> → 24.70 | 33.91 → <span style="color: #c00000;">**32.75**</span> | 39.56 → <span style="color: #c00000;">**33.47**</span> |
| Half-bound B / -0.5 | 95.6 → <span style="color: #c00000;">**77.3**</span> | <span style="color: #c00000;">**0.251**</span> → 0.269 | 1.60 → <span style="color: #c00000;">**0.00**</span> | 10.28 → <span style="color: #c00000;">**2.62**</span> | 29.56 → <span style="color: #c00000;">**27.50**</span> | 33.28 → <span style="color: #c00000;">**31.87**</span> |
| Rotary A / +0.5 | 74.1 → <span style="color: #c00000;">**67.9**</span> | <span style="color: #c00000;">**0.243**</span> → 0.253 | 1.92 → <span style="color: #c00000;">**0.00**</span> | 27.06 → <span style="color: #c00000;">**26.54**</span> | <span style="color: #c00000;">**31.37**</span> → 34.70 | 55.99 → <span style="color: #c00000;">**49.15**</span> |
| Rotary A / -0.5 | 101.9 → <span style="color: #c00000;">**70.9**</span> | 0.258 → <span style="color: #c00000;">**0.253**</span> | 1.60 → <span style="color: #c00000;">**0.00**</span> | 15.54 → <span style="color: #c00000;">**8.51**</span> | <span style="color: #c00000;">**26.29**</span> → 32.81 | <span style="color: #c00000;">**43.51**</span> → 54.87 |
| Rotary B / +0.5 | 68.3 → <span style="color: #c00000;">**63.5**</span> | <span style="color: #c00000;">**0.263**</span> → 0.271 | 1.28 → <span style="color: #c00000;">**0.00**</span> | 27.15 → <span style="color: #c00000;">**13.31**</span> | 35.41 → <span style="color: #c00000;">**33.81**</span> | <span style="color: #c00000;">**56.35**</span> → 56.67 |
| Rotary B / -0.5 | 91.1 → <span style="color: #c00000;">**73.5**</span> | 0.264 → <span style="color: #c00000;">**0.254**</span> | 1.28 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**2.70**</span> → 8.85 | <span style="color: #c00000;">**29.02**</span> → 29.90 | <span style="color: #c00000;">**39.07**</span> → 49.41 |
| Transverse A / +0.5 | <span style="color: #c00000;">**62.9**</span> → 65.6 | 0.282 → <span style="color: #c00000;">**0.276**</span> | 0.64 → <span style="color: #c00000;">**0.00**</span> | 21.98 → <span style="color: #c00000;">**18.78**</span> | 25.84 → <span style="color: #c00000;">**25.16**</span> | 30.11 → <span style="color: #c00000;">**29.58**</span> |
| Transverse A / -0.5 | 78.3 → <span style="color: #c00000;">**63.3**</span> | 0.293 → <span style="color: #c00000;">**0.282**</span> | 0.96 → <span style="color: #c00000;">**0.00**</span> | 15.49 → <span style="color: #c00000;">**0.09**</span> | 30.35 → <span style="color: #c00000;">**27.97**</span> | 44.18 → <span style="color: #c00000;">**31.67**</span> |
| Transverse B / +0.5 | 62.4 → <span style="color: #c00000;">**59.7**</span> | <span style="color: #c00000;">**0.282**</span> → 0.292 | 1.60 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**30.29**</span> → 32.79 | <span style="color: #c00000;">**26.89**</span> → 28.77 | <span style="color: #c00000;">**34.31**</span> → 39.74 |
| Transverse B / -0.5 | 65.3 → <span style="color: #c00000;">**63.8**</span> | <span style="color: #c00000;">**0.289**</span> → 0.294 | 1.60 → <span style="color: #c00000;">**0.00**</span> | 14.59 → <span style="color: #c00000;">**7.60**</span> | 30.11 → <span style="color: #c00000;">**28.53**</span> | 35.31 → <span style="color: #c00000;">**32.25**</span> |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |
| Rotary B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-go2-1"></a>

#### Go2, vx = ±1 m/s

<details>
<summary>Show all metrics for Go2 at ±1 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 0.048 → <span style="color: #c00000;">**0.033**</span> | <span style="color: #c00000;">**0.029**</span> → 0.032 | 4.76 → <span style="color: #c00000;">**3.26**</span> | 0.096 → <span style="color: #c00000;">**0.082**</span> | <span style="color: #c00000;">**28.7**</span> → 43.5 | <span style="color: #c00000;">**90.7**</span> → 75.9 |
| Trot / -1 | <span style="color: #c00000;">**0.031**</span> → 0.042 | 0.072 → <span style="color: #c00000;">**0.044**</span> | <span style="color: #c00000;">**3.08**</span> → 4.19 | 0.068 → <span style="color: #c00000;">**0.050**</span> | 43.2 → <span style="color: #c00000;">**16.9**</span> | 74.8 → <span style="color: #c00000;">**92.5**</span> |
| Bound / +1 | <span style="color: #c00000;">**0.078**</span> → 0.081 | <span style="color: #c00000;">**0.040**</span> → 0.041 | <span style="color: #c00000;">**7.79**</span> → 8.11 | 0.078 → <span style="color: #c00000;">**0.070**</span> | 53.2 → <span style="color: #c00000;">**34.1**</span> | 62.3 → <span style="color: #c00000;">**87.9**</span> |
| Bound / -1 | 0.161 → <span style="color: #c00000;">**0.112**</span> | 0.056 → <span style="color: #c00000;">**0.054**</span> | 16.07 → <span style="color: #c00000;">**11.23**</span> | <span style="color: #c00000;">**0.052**</span> → 0.098 | <span style="color: #c00000;">**5.2**</span> → 56.7 | <span style="color: #c00000;">**83.0**</span> → 51.4 |
| Half-bound A / +1 | 0.092 → <span style="color: #c00000;">**0.064**</span> | 0.071 → <span style="color: #c00000;">**0.069**</span> | 9.25 → <span style="color: #c00000;">**6.36**</span> | <span style="color: #c00000;">**0.046**</span> → 0.079 | <span style="color: #c00000;">**4.0**</span> → 60.2 | <span style="color: #c00000;">**104.2**</span> → 50.0 |
| Half-bound A / -1 | 0.099 → <span style="color: #c00000;">**0.067**</span> | 0.086 → <span style="color: #c00000;">**0.074**</span> | 9.90 → <span style="color: #c00000;">**6.71**</span> | 0.101 → <span style="color: #c00000;">**0.055**</span> | 70.4 → <span style="color: #c00000;">**0.9**</span> | 33.1 → <span style="color: #c00000;">**94.3**</span> |
| Half-bound B / +1 | <span style="color: #c00000;">**0.078**</span> → 0.090 | <span style="color: #c00000;">**0.044**</span> → 0.047 | <span style="color: #c00000;">**7.77**</span> → 8.95 | 0.134 → <span style="color: #c00000;">**0.076**</span> | 90.5 → <span style="color: #c00000;">**52.0**</span> | 8.3 → <span style="color: #c00000;">**62.9**</span> |
| Half-bound B / -1 | 0.084 → <span style="color: #c00000;">**0.062**</span> | 0.087 → <span style="color: #c00000;">**0.082**</span> | 8.36 → <span style="color: #c00000;">**6.16**</span> | <span style="color: #c00000;">**0.054**</span> → 0.073 | 63.0 → <span style="color: #c00000;">**41.2**</span> | 47.8 → <span style="color: #c00000;">**74.1**</span> |
| Rotary A / +1 | <span style="color: #c00000;">**0.059**</span> → 0.086 | 0.118 → <span style="color: #c00000;">**0.111**</span> | <span style="color: #c00000;">**5.91**</span> → 8.59 | <span style="color: #c00000;">**0.051**</span> → 0.114 | <span style="color: #c00000;">**4.5**</span> → 39.0 | <span style="color: #c00000;">**101.7**</span> → 77.5 |
| Rotary A / -1 | 0.123 → <span style="color: #c00000;">**0.070**</span> | <span style="color: #c00000;">**0.114**</span> → 0.131 | 12.35 → <span style="color: #c00000;">**7.00**</span> | <span style="color: #c00000;">**0.047**</span> → 0.094 | <span style="color: #c00000;">**76.9**</span> → 114.5 | <span style="color: #c00000;">**21.5**</span> → -21.4 |
| Rotary B / +1 | 0.068 → <span style="color: #c00000;">**0.050**</span> | 0.132 → <span style="color: #c00000;">**0.118**</span> | 6.82 → <span style="color: #c00000;">**5.03**</span> | 0.081 → <span style="color: #c00000;">**0.065**</span> | 28.8 → <span style="color: #c00000;">**1.3**</span> | 90.1 → <span style="color: #c00000;">**100.1**</span> |
| Rotary B / -1 | 0.143 → <span style="color: #c00000;">**0.059**</span> | <span style="color: #c00000;">**0.118**</span> → 0.157 | 14.32 → <span style="color: #c00000;">**5.92**</span> | <span style="color: #c00000;">**0.084**</span> → 0.126 | <span style="color: #c00000;">**15.6**</span> → 17.8 | 84.3 → <span style="color: #c00000;">**98.6**</span> |
| Transverse A / +1 | 0.073 → <span style="color: #c00000;">**0.070**</span> | <span style="color: #c00000;">**0.023**</span> → 0.031 | 7.32 → <span style="color: #c00000;">**7.00**</span> | <span style="color: #c00000;">**0.071**</span> → 0.108 | 23.6 → <span style="color: #c00000;">**23.4**</span> | <span style="color: #c00000;">**89.9**</span> → 87.1 |
| Transverse A / -1 | 0.060 → <span style="color: #c00000;">**0.048**</span> | 0.072 → <span style="color: #c00000;">**0.033**</span> | 6.02 → <span style="color: #c00000;">**4.76**</span> | <span style="color: #c00000;">**0.063**</span> → 0.092 | <span style="color: #c00000;">**23.7**</span> → 58.1 | <span style="color: #c00000;">**86.8**</span> → 52.3 |
| Transverse B / +1 | 0.070 → <span style="color: #c00000;">**0.055**</span> | <span style="color: #c00000;">**0.018**</span> → 0.021 | 7.04 → <span style="color: #c00000;">**5.49**</span> | <span style="color: #c00000;">**0.054**</span> → 0.087 | <span style="color: #c00000;">**35.7**</span> → 39.4 | <span style="color: #c00000;">**79.9**</span> → 75.3 |
| Transverse B / -1 | 0.061 → <span style="color: #c00000;">**0.032**</span> | <span style="color: #c00000;">**0.066**</span> → 0.075 | 6.08 → <span style="color: #c00000;">**3.22**</span> | 0.089 → <span style="color: #c00000;">**0.064**</span> | 74.0 → <span style="color: #c00000;">**18.6**</span> | 34.3 → <span style="color: #c00000;">**93.3**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 97.9 → <span style="color: #c00000;">**100.0**</span> | 10.0 → 10.0 | 53.85 → <span style="color: #c00000;">**55.77**</span> | 3.85 → <span style="color: #c00000;">**1.92**</span> | 0.00 → 0.00 | 3.57 → <span style="color: #c00000;">**0.00**</span> | 1.00 → 1.00 |
| Trot / -1 | 94.8 → 94.8 | 10.1 → <span style="color: #c00000;">**2.1**</span> | 50.96 → <span style="color: #c00000;">**52.49**</span> | 5.04 → <span style="color: #c00000;">**3.51**</span> | 0.00 → 0.00 | 8.93 → 8.93 | 1.00 → 1.00 |
| Bound / +1 | <span style="color: #c00000;">**87.3**</span> → 84.4 | <span style="color: #c00000;">**23.5**</span> → 26.2 | 57.26 → <span style="color: #c00000;">**55.77**</span> | <span style="color: #c00000;">**5.61**</span> → 7.69 | <span style="color: #c00000;">**15.63**</span> → 20.00 | <span style="color: #c00000;">**10.66**</span> → 12.50 | <span style="color: #c00000;">**1.53**</span> → 2.00 |
| Bound / -1 | <span style="color: #c00000;">**91.7**</span> → 91.6 | <span style="color: #c00000;">**9.9**</span> → 14.8 | <span style="color: #c00000;">**54.81**</span> → 57.67 | <span style="color: #c00000;">**4.81**</span> → 5.79 | <span style="color: #c00000;">**5.00**</span> → 10.00 | 10.71 → <span style="color: #c00000;">**7.19**</span> | 1.00 → 1.00 |
| Half-bound A / +1 | 94.6 → <span style="color: #c00000;">**96.9**</span> | 26.2 → <span style="color: #c00000;">**11.2**</span> | <span style="color: #c00000;">**57.44**</span> → 58.65 | 5.24 → <span style="color: #c00000;">**3.73**</span> | 5.24 → <span style="color: #c00000;">**4.88**</span> | 5.45 → <span style="color: #c00000;">**1.82**</span> | <span style="color: #c00000;">**1.00**</span> → 2.00 |
| Half-bound A / -1 | 94.8 → <span style="color: #c00000;">**95.8**</span> | 13.8 → <span style="color: #c00000;">**10.0**</span> | <span style="color: #c00000;">**56.73**</span> → 57.67 | 6.73 → <span style="color: #c00000;">**2.74**</span> | 4.88 → 4.88 | 5.45 → <span style="color: #c00000;">**3.68**</span> | 1.00 → 1.00 |
| Half-bound B / +1 | 90.0 → <span style="color: #c00000;">**94.8**</span> | 36.2 → <span style="color: #c00000;">**26.2**</span> | 56.35 → <span style="color: #c00000;">**55.77**</span> | 8.18 → <span style="color: #c00000;">**6.85**</span> | 11.16 → <span style="color: #c00000;">**4.88**</span> | 9.09 → <span style="color: #c00000;">**5.45**</span> | 1.00 → 1.00 |
| Half-bound B / -1 | 93.8 → 93.8 | <span style="color: #c00000;">**10.0**</span> → 15.0 | <span style="color: #c00000;">**54.81**</span> → 57.69 | 6.73 → <span style="color: #c00000;">**3.85**</span> | <span style="color: #c00000;">**4.88**</span> → 7.32 | 7.27 → <span style="color: #c00000;">**5.45**</span> | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Rotary A / +1 | 91.7 → <span style="color: #c00000;">**94.8**</span> | <span style="color: #c00000;">**5.6**</span> → 15.6 | 57.69 → <span style="color: #c00000;">**56.73**</span> | <span style="color: #c00000;">**3.85**</span> → 4.81 | 9.52 → <span style="color: #c00000;">**4.76**</span> | 7.41 → <span style="color: #c00000;">**5.56**</span> | 2.00 → 2.00 |
| Rotary A / -1 | 92.7 → <span style="color: #c00000;">**93.8**</span> | 18.8 → 18.8 | <span style="color: #c00000;">**55.77**</span> → 54.81 | 5.77 → <span style="color: #c00000;">**3.96**</span> | 7.14 → <span style="color: #c00000;">**4.76**</span> | 7.41 → 7.41 | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Rotary B / +1 | 93.1 → <span style="color: #c00000;">**93.8**</span> | 10.6 → <span style="color: #c00000;">**10.0**</span> | 58.10 → <span style="color: #c00000;">**56.73**</span> | 3.48 → <span style="color: #c00000;">**2.88**</span> | 9.52 → <span style="color: #c00000;">**7.14**</span> | <span style="color: #c00000;">**4.77**</span> → 5.56 | 2.00 → 2.00 |
| Rotary B / -1 | 93.7 → <span style="color: #c00000;">**94.8**</span> | 22.6 → <span style="color: #c00000;">**12.9**</span> | <span style="color: #c00000;">**56.73**</span> → 54.43 | 4.81 → <span style="color: #c00000;">**4.80**</span> | 5.96 → <span style="color: #c00000;">**4.76**</span> | 6.48 → <span style="color: #c00000;">**5.56**</span> | <span style="color: #c00000;">**1.00**</span> → 2.00 |
| Transverse A / +1 | <span style="color: #c00000;">**96.9**</span> → 94.8 | 26.8 → <span style="color: #c00000;">**22.4**</span> | 54.81 → <span style="color: #c00000;">**55.77**</span> | 5.88 → <span style="color: #c00000;">**3.85**</span> | <span style="color: #c00000;">**4.76**</span> → 7.14 | <span style="color: #c00000;">**1.85**</span> → 3.70 | 1.00 → 1.00 |
| Transverse A / -1 | <span style="color: #c00000;">**95.6**</span> → 94.8 | 16.2 → <span style="color: #c00000;">**11.2**</span> | 53.06 → <span style="color: #c00000;">**53.85**</span> | 5.10 → <span style="color: #c00000;">**4.92**</span> | <span style="color: #c00000;">**4.01**</span> → 4.76 | <span style="color: #c00000;">**4.63**</span> → 5.56 | 1.00 → 1.00 |
| Transverse B / +1 | 91.7 → <span style="color: #c00000;">**96.9**</span> | <span style="color: #c00000;">**17.4**</span> → 21.8 | 56.73 → 56.73 | 6.73 → <span style="color: #c00000;">**4.81**</span> | 9.52 → <span style="color: #c00000;">**4.76**</span> | 7.41 → <span style="color: #c00000;">**1.85**</span> | 1.00 → 1.00 |
| Transverse B / -1 | <span style="color: #c00000;">**93.8**</span> → 90.6 | <span style="color: #c00000;">**11.2**</span> → 22.4 | 52.88 → <span style="color: #c00000;">**53.85**</span> | 5.88 → <span style="color: #c00000;">**5.77**</span> | <span style="color: #c00000;">**4.76**</span> → 9.52 | <span style="color: #c00000;">**7.41**</span> → 9.26 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 66.4 → <span style="color: #c00000;">**62.0**</span> | 0.364 → <span style="color: #c00000;">**0.322**</span> | 0.00 → 0.00 | <span style="color: #c00000;">**7.33**</span> → 15.98 | <span style="color: #c00000;">**14.07**</span> → 16.72 | <span style="color: #c00000;">**15.74**</span> → 24.41 |
| Trot / -1 | <span style="color: #c00000;">**58.0**</span> → 59.0 | 0.358 → <span style="color: #c00000;">**0.325**</span> | 0.00 → 0.00 | 27.23 → <span style="color: #c00000;">**23.78**</span> | 30.20 → <span style="color: #c00000;">**19.55**</span> | 52.19 → <span style="color: #c00000;">**30.60**</span> |
| Bound / +1 | 122.9 → <span style="color: #c00000;">**114.2**</span> | 0.436 → <span style="color: #c00000;">**0.397**</span> | 1.07 → <span style="color: #c00000;">**0.00**</span> | 9.00 → <span style="color: #c00000;">**3.79**</span> | <span style="color: #c00000;">**23.10**</span> → 24.17 | <span style="color: #c00000;">**20.44**</span> → 24.93 |
| Bound / -1 | 134.6 → <span style="color: #c00000;">**131.9**</span> | <span style="color: #c00000;">**0.385**</span> → 0.422 | 0.96 → <span style="color: #c00000;">**0.64**</span> | 9.22 → <span style="color: #c00000;">**1.16**</span> | 29.44 → <span style="color: #c00000;">**25.49**</span> | 35.39 → <span style="color: #c00000;">**22.28**</span> |
| Half-bound A / +1 | 120.0 → <span style="color: #c00000;">**113.2**</span> | 0.450 → <span style="color: #c00000;">**0.358**</span> | 1.92 → <span style="color: #c00000;">**0.00**</span> | 23.43 → <span style="color: #c00000;">**3.81**</span> | 25.03 → <span style="color: #c00000;">**24.19**</span> | 36.19 → <span style="color: #c00000;">**29.65**</span> |
| Half-bound A / -1 | 126.3 → <span style="color: #c00000;">**116.7**</span> | 0.383 → <span style="color: #c00000;">**0.375**</span> | 1.60 → <span style="color: #c00000;">**0.00**</span> | 13.61 → <span style="color: #c00000;">**3.22**</span> | 34.67 → <span style="color: #c00000;">**30.95**</span> | 50.37 → <span style="color: #c00000;">**39.25**</span> |
| Half-bound B / +1 | <span style="color: #c00000;">**103.8**</span> → 106.2 | <span style="color: #c00000;">**0.408**</span> → 0.429 | 2.25 → <span style="color: #c00000;">**0.32**</span> | <span style="color: #c00000;">**12.37**</span> → 26.17 | <span style="color: #c00000;">**27.66**</span> → 31.25 | <span style="color: #c00000;">**38.35**</span> → 39.30 |
| Half-bound B / -1 | <span style="color: #c00000;">**119.6**</span> → 122.5 | <span style="color: #c00000;">**0.347**</span> → 0.387 | 1.92 → <span style="color: #c00000;">**0.96**</span> | 15.46 → <span style="color: #c00000;">**0.88**</span> | 32.90 → <span style="color: #c00000;">**29.80**</span> | 46.06 → <span style="color: #c00000;">**29.36**</span> |
| Rotary A / +1 | <span style="color: #c00000;">**90.5**</span> → 99.0 | <span style="color: #c00000;">**0.358**</span> → 0.373 | 1.60 → <span style="color: #c00000;">**0.32**</span> | <span style="color: #c00000;">**19.92**</span> → 24.27 | <span style="color: #c00000;">**25.29**</span> → 26.18 | 50.73 → <span style="color: #c00000;">**33.93**</span> |
| Rotary A / -1 | 118.7 → <span style="color: #c00000;">**118.4**</span> | 0.375 → <span style="color: #c00000;">**0.367**</span> | 1.92 → <span style="color: #c00000;">**0.64**</span> | 22.93 → <span style="color: #c00000;">**1.27**</span> | <span style="color: #c00000;">**30.14**</span> → 30.88 | 49.14 → <span style="color: #c00000;">**48.34**</span> |
| Rotary B / +1 | <span style="color: #c00000;">**91.8**</span> → 107.7 | 0.355 → <span style="color: #c00000;">**0.338**</span> | 0.96 → <span style="color: #c00000;">**0.00**</span> | 16.57 → <span style="color: #c00000;">**14.56**</span> | 25.44 → <span style="color: #c00000;">**23.43**</span> | 42.38 → <span style="color: #c00000;">**40.53**</span> |
| Rotary B / -1 | 123.4 → <span style="color: #c00000;">**106.9**</span> | 0.381 → <span style="color: #c00000;">**0.370**</span> | 1.28 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**9.58**</span> → 15.08 | 33.40 → <span style="color: #c00000;">**32.57**</span> | <span style="color: #c00000;">**48.68**</span> → 55.07 |
| Transverse A / +1 | 83.3 → <span style="color: #c00000;">**79.3**</span> | 0.377 → <span style="color: #c00000;">**0.356**</span> | 1.28 → <span style="color: #c00000;">**0.96**</span> | 20.17 → <span style="color: #c00000;">**17.82**</span> | <span style="color: #c00000;">**21.12**</span> → 21.41 | 28.87 → <span style="color: #c00000;">**28.30**</span> |
| Transverse A / -1 | 114.1 → <span style="color: #c00000;">**97.8**</span> | 0.414 → <span style="color: #c00000;">**0.372**</span> | 1.28 → <span style="color: #c00000;">**0.64**</span> | 12.81 → <span style="color: #c00000;">**7.22**</span> | 33.08 → <span style="color: #c00000;">**26.88**</span> | 53.54 → <span style="color: #c00000;">**37.65**</span> |
| Transverse B / +1 | 80.0 → <span style="color: #c00000;">**79.3**</span> | <span style="color: #c00000;">**0.372**</span> → 0.384 | 1.92 → <span style="color: #c00000;">**0.32**</span> | <span style="color: #c00000;">**17.32**</span> → 29.41 | <span style="color: #c00000;">**22.69**</span> → 26.76 | <span style="color: #c00000;">**25.70**</span> → 38.27 |
| Transverse B / -1 | 95.1 → <span style="color: #c00000;">**87.4**</span> | <span style="color: #c00000;">**0.353**</span> → 0.381 | 1.60 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**9.08**</span> → 13.89 | <span style="color: #c00000;">**28.08**</span> → 33.11 | <span style="color: #c00000;">**39.27**</span> → 44.57 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

</details>

<a id="detail-go2-1p5"></a>

#### Go2, vx = ±1.5 m/s

<details>
<summary>Show all metrics for Go2 at ±1.5 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | <span style="color: #c00000;">**0.095**</span> → 0.111 | 0.031 → <span style="color: #c00000;">**0.023**</span> | <span style="color: #c00000;">**6.35**</span> → 7.42 | <span style="color: #c00000;">**0.092**</span> → 0.117 | <span style="color: #c00000;">**20.9**</span> → 49.4 | <span style="color: #c00000;">**87.5**</span> → 60.8 |
| Trot / -1.5 | <span style="color: #c00000;">**0.165**</span> → 0.172 | 0.181 → <span style="color: #c00000;">**0.098**</span> | <span style="color: #c00000;">**10.99**</span> → 11.46 | 0.106 → <span style="color: #c00000;">**0.097**</span> | 97.0 → <span style="color: #c00000;">**75.8**</span> | 7.9 → <span style="color: #c00000;">**30.9**</span> |
| Bound / +1.5 | 0.172 → <span style="color: #c00000;">**0.149**</span> | 0.065 → <span style="color: #c00000;">**0.048**</span> | 11.50 → <span style="color: #c00000;">**9.90**</span> | <span style="color: #c00000;">**0.072**</span> → 0.073 | <span style="color: #c00000;">**1.4**</span> → 5.0 | 92.3 → <span style="color: #c00000;">**94.1**</span> |
| Bound / -1.5 | 0.389 → <span style="color: #c00000;">**0.124**</span> | <span style="color: #c00000;">**0.089**</span> → 0.119 | 25.94 → <span style="color: #c00000;">**8.23**</span> | <span style="color: #c00000;">**0.063**</span> → 0.177 | <span style="color: #c00000;">**20.2**</span> → 128.9 | <span style="color: #c00000;">**67.3**</span> → -37.5 |
| Half-bound A / +1.5 | 0.170 → <span style="color: #c00000;">**0.127**</span> | <span style="color: #c00000;">**0.065**</span> → 0.087 | 11.31 → <span style="color: #c00000;">**8.49**</span> | 0.097 → <span style="color: #c00000;">**0.072**</span> | 33.8 → <span style="color: #c00000;">**12.0**</span> | 79.8 → <span style="color: #c00000;">**94.3**</span> |
| Half-bound A / -1.5 | 0.324 → <span style="color: #c00000;">**0.253**</span> | 0.120 → <span style="color: #c00000;">**0.094**</span> | 21.60 → <span style="color: #c00000;">**16.84**</span> | <span style="color: #c00000;">**0.095**</span> → 0.098 | <span style="color: #c00000;">**9.4**</span> → 64.4 | <span style="color: #c00000;">**75.2**</span> → 40.3 |
| Half-bound B / +1.5 | <span style="color: #c00000;">**0.170**</span> → 0.221 | <span style="color: #c00000;">**0.049**</span> → 0.051 | <span style="color: #c00000;">**11.37**</span> → 14.73 | 0.115 → <span style="color: #c00000;">**0.096**</span> | <span style="color: #c00000;">**31.8**</span> → 36.1 | <span style="color: #c00000;">**79.0**</span> → 70.6 |
| Half-bound B / -1.5 | 0.286 → <span style="color: #c00000;">**0.142**</span> | <span style="color: #c00000;">**0.148**</span> → 0.218 | 19.06 → <span style="color: #c00000;">**9.44**</span> | <span style="color: #c00000;">**0.123**</span> → 0.272 | <span style="color: #c00000;">**98.7**</span> → 112.1 | <span style="color: #c00000;">**3.3**</span> → -15.0 |
| Rotary A / +1.5 | <span style="color: #c00000;">**0.145**</span> → 0.209 | 0.110 → <span style="color: #c00000;">**0.097**</span> | <span style="color: #c00000;">**9.69**</span> → 13.94 | <span style="color: #c00000;">**0.118**</span> → 0.166 | <span style="color: #c00000;">**17.8**</span> → 89.3 | <span style="color: #c00000;">**89.0**</span> → 8.4 |
| Rotary A / -1.5 | 0.315 → <span style="color: #c00000;">**0.115**</span> | <span style="color: #c00000;">**0.124**</span> → 0.133 | 20.99 → <span style="color: #c00000;">**7.68**</span> | <span style="color: #c00000;">**0.082**</span> → 0.150 | <span style="color: #c00000;">**111.5**</span> → 129.4 | <span style="color: #c00000;">**-14.1**</span> → -39.3 |
| Rotary B / +1.5 | 0.115 → 0.115 | 0.143 → <span style="color: #c00000;">**0.109**</span> | 7.67 → 7.67 | 0.102 → <span style="color: #c00000;">**0.073**</span> | 63.8 → <span style="color: #c00000;">**36.2**</span> | 52.3 → <span style="color: #c00000;">**79.6**</span> |
| Rotary B / -1.5 | 0.325 → <span style="color: #c00000;">**0.237**</span> | <span style="color: #c00000;">**0.106**</span> → 0.128 | 21.64 → <span style="color: #c00000;">**15.81**</span> | 0.152 → <span style="color: #c00000;">**0.135**</span> | 79.5 → <span style="color: #c00000;">**5.1**</span> | 19.0 → <span style="color: #c00000;">**84.9**</span> |
| Transverse A / +1.5 | <span style="color: #c00000;">**0.134**</span> → 0.135 | <span style="color: #c00000;">**0.049**</span> → 0.090 | <span style="color: #c00000;">**8.92**</span> → 9.00 | <span style="color: #c00000;">**0.101**</span> → 0.145 | <span style="color: #c00000;">**5.6**</span> → 56.9 | <span style="color: #c00000;">**93.7**</span> → 48.4 |
| Transverse A / -1.5 | 0.290 → <span style="color: #c00000;">**0.225**</span> | 0.143 → <span style="color: #c00000;">**0.089**</span> | 19.30 → <span style="color: #c00000;">**15.00**</span> | <span style="color: #c00000;">**0.094**</span> → 0.102 | <span style="color: #c00000;">**10.7**</span> → 25.9 | <span style="color: #c00000;">**80.2**</span> → 74.5 |
| Transverse B / +1.5 | <span style="color: #c00000;">**0.148**</span> → 0.177 | 0.055 → <span style="color: #c00000;">**0.035**</span> | <span style="color: #c00000;">**9.87**</span> → 11.80 | <span style="color: #c00000;">**0.093**</span> → 0.106 | <span style="color: #c00000;">**50.3**</span> → 57.7 | <span style="color: #c00000;">**61.9**</span> → 49.0 |
| Transverse B / -1.5 | 0.261 → <span style="color: #c00000;">**0.227**</span> | 0.163 → <span style="color: #c00000;">**0.078**</span> | 17.40 → <span style="color: #c00000;">**15.10**</span> | 0.130 → <span style="color: #c00000;">**0.092**</span> | 107.4 → <span style="color: #c00000;">**57.2**</span> | -3.3 → <span style="color: #c00000;">**50.1**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 95.3 → <span style="color: #c00000;">**95.8**</span> | 15.6 → <span style="color: #c00000;">**10.0**</span> | <span style="color: #c00000;">**52.40**</span> → 51.85 | <span style="color: #c00000;">**4.44**</span> → 4.99 | 2.50 → <span style="color: #c00000;">**0.00**</span> | <span style="color: #c00000;">**6.25**</span> → 7.27 | 1.00 → 1.00 |
| Trot / -1.5 | 90.6 → <span style="color: #c00000;">**93.2**</span> | 19.9 → <span style="color: #c00000;">**2.5**</span> | 49.04 → <span style="color: #c00000;">**50.94**</span> | 6.96 → <span style="color: #c00000;">**5.06**</span> | 2.50 → <span style="color: #c00000;">**0.00**</span> | 14.29 → <span style="color: #c00000;">**11.65**</span> | 1.00 → 1.00 |
| Bound / +1.5 | 82.2 → <span style="color: #c00000;">**83.9**</span> | 38.9 → <span style="color: #c00000;">**19.9**</span> | <span style="color: #c00000;">**55.87**</span> → 58.17 | 6.74 → <span style="color: #c00000;">**6.37**</span> | <span style="color: #c00000;">**19.34**</span> → 22.50 | 16.64 → <span style="color: #c00000;">**11.61**</span> | <span style="color: #c00000;">**1.00**</span> → 2.00 |
| Bound / -1.5 | <span style="color: #c00000;">**90.6**</span> → 81.9 | 24.8 → <span style="color: #c00000;">**9.9**</span> | 53.82 → <span style="color: #c00000;">**54.20**</span> | <span style="color: #c00000;">**3.87**</span> → 5.28 | <span style="color: #c00000;">**7.50**</span> → 15.92 | <span style="color: #c00000;">**10.76**</span> → 19.64 | <span style="color: #c00000;">**1.00**</span> → 1.97 |
| Half-bound A / +1.5 | <span style="color: #c00000;">**92.8**</span> → 91.7 | 32.5 → <span style="color: #c00000;">**6.2**</span> | <span style="color: #c00000;">**54.48**</span> → 53.86 | 6.06 → <span style="color: #c00000;">**4.80**</span> | 6.80 → <span style="color: #c00000;">**4.88**</span> | <span style="color: #c00000;">**7.56**</span> → 10.85 | <span style="color: #c00000;">**1.05**</span> → 2.00 |
| Half-bound A / -1.5 | <span style="color: #c00000;">**95.5**</span> → 94.8 | 14.6 → <span style="color: #c00000;">**10.0**</span> | 53.52 → <span style="color: #c00000;">**55.74**</span> | 3.52 → <span style="color: #c00000;">**1.95**</span> | <span style="color: #c00000;">**2.44**</span> → 4.88 | 6.08 → <span style="color: #c00000;">**5.50**</span> | 1.34 → <span style="color: #c00000;">**1.00**</span> |
| Half-bound B / +1.5 | 87.0 → <span style="color: #c00000;">**94.8**</span> | 42.8 → <span style="color: #c00000;">**31.8**</span> | <span style="color: #c00000;">**54.76**</span> → 52.88 | <span style="color: #c00000;">**5.71**</span> → 7.81 | 12.99 → <span style="color: #c00000;">**4.88**</span> | 13.01 → <span style="color: #c00000;">**5.45**</span> | 1.01 → <span style="color: #c00000;">**1.00**</span> |
| Half-bound B / -1.5 | <span style="color: #c00000;">**90.5**</span> → 89.6 | 20.0 → <span style="color: #c00000;">**8.8**</span> | 49.59 → <span style="color: #c00000;">**51.92**</span> | 9.44 → <span style="color: #c00000;">**6.85**</span> | <span style="color: #c00000;">**2.78**</span> → 4.88 | 14.55 → 14.55 | 2.00 → 2.00 |
| Rotary A / +1.5 | 85.9 → <span style="color: #c00000;">**90.1**</span> | <span style="color: #c00000;">**12.2**</span> → 17.4 | <span style="color: #c00000;">**56.73**</span> → 54.33 | <span style="color: #c00000;">**6.29**</span> → 6.37 | 15.48 → <span style="color: #c00000;">**7.14**</span> | 12.96 → <span style="color: #c00000;">**12.04**</span> | 2.00 → 2.00 |
| Rotary A / -1.5 | <span style="color: #c00000;">**89.6**</span> → 87.5 | 23.8 → <span style="color: #c00000;">**22.6**</span> | <span style="color: #c00000;">**54.81**</span> → 50.00 | <span style="color: #c00000;">**5.88**</span> → 6.85 | 9.52 → <span style="color: #c00000;">**7.14**</span> | <span style="color: #c00000;">**11.11**</span> → 16.67 | 2.00 → 2.00 |
| Rotary B / +1.5 | 83.7 → <span style="color: #c00000;">**89.6**</span> | 18.8 → <span style="color: #c00000;">**13.2**</span> | 52.01 → <span style="color: #c00000;">**55.77**</span> | 4.10 → <span style="color: #c00000;">**3.85**</span> | 12.81 → <span style="color: #c00000;">**11.90**</span> | 19.06 → <span style="color: #c00000;">**9.26**</span> | 2.00 → 2.00 |
| Rotary B / -1.5 | 87.4 → <span style="color: #c00000;">**94.8**</span> | 34.1 → <span style="color: #c00000;">**19.7**</span> | <span style="color: #c00000;">**54.46**</span> → 53.11 | 8.11 → <span style="color: #c00000;">**3.73**</span> | 11.55 → <span style="color: #c00000;">**4.76**</span> | 13.34 → <span style="color: #c00000;">**5.56**</span> | <span style="color: #c00000;">**1.01**</span> → 2.00 |
| Transverse A / +1.5 | <span style="color: #c00000;">**93.2**</span> → 90.6 | 36.2 → <span style="color: #c00000;">**32.4**</span> | <span style="color: #c00000;">**53.76**</span> → 51.92 | 6.97 → <span style="color: #c00000;">**6.85**</span> | <span style="color: #c00000;">**7.13**</span> → 7.14 | <span style="color: #c00000;">**6.51**</span> → 11.11 | 1.23 → <span style="color: #c00000;">**1.00**</span> |
| Transverse A / -1.5 | <span style="color: #c00000;">**94.3**</span> → 92.7 | <span style="color: #c00000;">**16.2**</span> → 18.3 | 50.48 → <span style="color: #c00000;">**52.45**</span> | 5.52 → <span style="color: #c00000;">**3.92**</span> | <span style="color: #c00000;">**3.57**</span> → 5.44 | <span style="color: #c00000;">**7.41**</span> → 8.76 | 1.50 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / +1.5 | 90.6 → <span style="color: #c00000;">**93.6**</span> | <span style="color: #c00000;">**22.4**</span> → 30.9 | <span style="color: #c00000;">**55.73**</span> → 53.24 | <span style="color: #c00000;">**3.83**</span> → 7.43 | 9.52 → <span style="color: #c00000;">**7.14**</span> | 9.31 → <span style="color: #c00000;">**5.78**</span> | <span style="color: #c00000;">**1.00**</span> → 1.20 |
| Transverse B / -1.5 | <span style="color: #c00000;">**92.0**</span> → 89.6 | <span style="color: #c00000;">**15.6**</span> → 22.4 | 49.01 → <span style="color: #c00000;">**52.88**</span> | 7.83 → <span style="color: #c00000;">**5.88**</span> | <span style="color: #c00000;">**2.38**</span> → 9.52 | 12.31 → <span style="color: #c00000;">**11.11**</span> | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 110.9 → <span style="color: #c00000;">**109.6**</span> | 0.435 → <span style="color: #c00000;">**0.383**</span> | 0.32 → <span style="color: #c00000;">**0.18**</span> | <span style="color: #c00000;">**9.04**</span> → 10.37 | <span style="color: #c00000;">**10.31**</span> → 16.37 | <span style="color: #c00000;">**14.49**</span> → 22.23 |
| Trot / -1.5 | 103.6 → <span style="color: #c00000;">**101.8**</span> | 0.470 → <span style="color: #c00000;">**0.406**</span> | 0.32 → 0.32 | <span style="color: #c00000;">**19.43**</span> → 24.41 | 36.02 → <span style="color: #c00000;">**26.45**</span> | 55.77 → <span style="color: #c00000;">**47.96**</span> |
| Bound / +1.5 | <span style="color: #c00000;">**184.9**</span> → 190.8 | 0.597 → <span style="color: #c00000;">**0.559**</span> | 2.68 → <span style="color: #c00000;">**1.44**</span> | 10.89 → <span style="color: #c00000;">**5.52**</span> | <span style="color: #c00000;">**18.12**</span> → 22.05 | <span style="color: #c00000;">**12.94**</span> → 22.95 |
| Bound / -1.5 | <span style="color: #c00000;">**159.4**</span> → 273.0 | <span style="color: #c00000;">**0.565**</span> → 0.708 | <span style="color: #c00000;">**1.28**</span> → 2.07 | <span style="color: #c00000;">**7.83**</span> → 34.33 | <span style="color: #c00000;">**34.26**</span> → 37.52 | 36.62 → <span style="color: #c00000;">**22.62**</span> |
| Half-bound A / +1.5 | <span style="color: #c00000;">**168.2**</span> → 224.7 | <span style="color: #c00000;">**0.588**</span> → 0.590 | 2.30 → <span style="color: #c00000;">**1.26**</span> | 12.39 → <span style="color: #c00000;">**11.37**</span> | <span style="color: #c00000;">**20.68**</span> → 22.22 | <span style="color: #c00000;">**20.10**</span> → 22.15 |
| Half-bound A / -1.5 | 165.4 → <span style="color: #c00000;">**151.2**</span> | 0.533 → <span style="color: #c00000;">**0.525**</span> | 1.60 → <span style="color: #c00000;">**1.28**</span> | 14.32 → <span style="color: #c00000;">**9.72**</span> | <span style="color: #c00000;">**37.48**</span> → 42.11 | 53.84 → <span style="color: #c00000;">**48.71**</span> |
| Half-bound B / +1.5 | <span style="color: #c00000;">**132.6**</span> → 159.3 | <span style="color: #c00000;">**0.539**</span> → 0.574 | 2.55 → <span style="color: #c00000;">**1.60**</span> | <span style="color: #c00000;">**11.69**</span> → 14.38 | <span style="color: #c00000;">**18.65**</span> → 21.83 | 31.44 → <span style="color: #c00000;">**29.55**</span> |
| Half-bound B / -1.5 | <span style="color: #c00000;">**145.9**</span> → 263.8 | <span style="color: #c00000;">**0.506**</span> → 0.639 | 2.24 → 2.24 | <span style="color: #c00000;">**20.32**</span> → 32.23 | 41.86 → <span style="color: #c00000;">**38.36**</span> | 58.95 → <span style="color: #c00000;">**32.29**</span> |
| Rotary A / +1.5 | <span style="color: #c00000;">**148.9**</span> → 155.7 | <span style="color: #c00000;">**0.507**</span> → 0.524 | 3.13 → <span style="color: #c00000;">**0.96**</span> | 27.39 → <span style="color: #c00000;">**10.29**</span> | 21.25 → <span style="color: #c00000;">**21.22**</span> | 43.93 → <span style="color: #c00000;">**22.64**</span> |
| Rotary A / -1.5 | <span style="color: #c00000;">**134.5**</span> → 211.6 | <span style="color: #c00000;">**0.507**</span> → 0.599 | <span style="color: #c00000;">**1.92**</span> → 2.88 | 30.36 → <span style="color: #c00000;">**21.08**</span> | <span style="color: #c00000;">**32.49**</span> → 35.44 | 59.60 → <span style="color: #c00000;">**43.03**</span> |
| Rotary B / +1.5 | <span style="color: #c00000;">**170.2**</span> → 185.9 | 0.512 → <span style="color: #c00000;">**0.508**</span> | 1.38 → <span style="color: #c00000;">**0.00**</span> | 13.40 → <span style="color: #c00000;">**8.57**</span> | <span style="color: #c00000;">**15.26**</span> → 21.18 | <span style="color: #c00000;">**32.13**</span> → 34.66 |
| Rotary B / -1.5 | <span style="color: #c00000;">**162.6**</span> → 171.0 | 0.566 → <span style="color: #c00000;">**0.531**</span> | 1.60 → <span style="color: #c00000;">**0.64**</span> | 10.90 → <span style="color: #c00000;">**3.99**</span> | <span style="color: #c00000;">**33.87**</span> → 42.76 | <span style="color: #c00000;">**42.31**</span> → 56.62 |
| Transverse A / +1.5 | <span style="color: #c00000;">**132.5**</span> → 137.5 | 0.500 → <span style="color: #c00000;">**0.462**</span> | <span style="color: #c00000;">**1.75**</span> → 2.24 | <span style="color: #c00000;">**0.47**</span> → 1.99 | <span style="color: #c00000;">**19.67**</span> → 20.78 | <span style="color: #c00000;">**19.74**</span> → 26.82 |
| Transverse A / -1.5 | 147.2 → <span style="color: #c00000;">**137.4**</span> | 0.535 → <span style="color: #c00000;">**0.483**</span> | <span style="color: #c00000;">**1.60**</span> → 2.23 | 15.09 → <span style="color: #c00000;">**12.66**</span> | 37.26 → <span style="color: #c00000;">**31.37**</span> | 57.46 → <span style="color: #c00000;">**49.55**</span> |
| Transverse B / +1.5 | 136.1 → <span style="color: #c00000;">**132.1**</span> | <span style="color: #c00000;">**0.513**</span> → 0.548 | 2.88 → <span style="color: #c00000;">**1.66**</span> | <span style="color: #c00000;">**0.68**</span> → 24.47 | <span style="color: #c00000;">**15.01**</span> → 24.83 | <span style="color: #c00000;">**19.03**</span> → 35.52 |
| Transverse B / -1.5 | 131.6 → <span style="color: #c00000;">**116.4**</span> | 0.483 → <span style="color: #c00000;">**0.477**</span> | 2.17 → <span style="color: #c00000;">**1.28**</span> | <span style="color: #c00000;">**12.69**</span> → 21.38 | <span style="color: #c00000;">**34.61**</span> → 35.37 | <span style="color: #c00000;">**51.75**</span> → 57.02 |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | <span style="color: #c00000;">**50.0**</span> → 0.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | <span style="color: #c00000;">**56.0**</span> → 0.0 | <span style="color: #c00000;">**56.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**56.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 3.0 → <span style="color: #c00000;">**100.0**</span> | 100.0 → 100.0 |

</details>

<a id="detail-go2-2"></a>

#### Go2, vx = ±2 m/s

<details>
<summary>Show all metrics for Go2 at ±2 m/s (16 gait/direction cases)</summary>

**Tracking and direction control**

| Gait / vx command [m/s] | [vx RMSE](#metric-velocity-rmse) [m/s] ↓ | [vy RMSE](#metric-velocity-rmse) [m/s] ↓ | [Relative vx RMSE](#metric-relative-velocity-rmse) [%] ↓ | [Yaw RMSE](#metric-yaw-rate-rmse) [rad/s] ↓ | [Heading RMSE](#metric-heading-rmse) [degrees] ↓ | [Progress](#metric-directed-progress) [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | <span style="color: #c00000;">**0.439**</span> → 0.479 | 0.080 → <span style="color: #c00000;">**0.049**</span> | <span style="color: #c00000;">**21.94**</span> → 23.96 | 0.133 → <span style="color: #c00000;">**0.100**</span> | 71.8 → <span style="color: #c00000;">**50.5**</span> | 25.4 → <span style="color: #c00000;">**47.9**</span> |
| Trot / -2 | N/A → 0.503 | N/A → 0.146 | N/A → 25.15 | N/A → 0.122 | N/A → 52.6 | N/A → 44.2 |
| Bound / +2 | 0.459 → <span style="color: #c00000;">**0.435**</span> | <span style="color: #c00000;">**0.048**</span> → 0.083 | 22.93 → <span style="color: #c00000;">**21.75**</span> | <span style="color: #c00000;">**0.115**</span> → 0.117 | 30.4 → <span style="color: #c00000;">**7.5**</span> | 67.4 → <span style="color: #c00000;">**79.6**</span> |
| Bound / -2 | 0.759 → <span style="color: #c00000;">**0.439**</span> | <span style="color: #c00000;">**0.168**</span> → 0.197 | 37.94 → <span style="color: #c00000;">**21.94**</span> | <span style="color: #c00000;">**0.098**</span> → 0.224 | <span style="color: #c00000;">**35.3**</span> → 127.9 | <span style="color: #c00000;">**53.9**</span> → -28.4 |
| Half-bound A / +2 | 0.453 → <span style="color: #c00000;">**0.420**</span> | <span style="color: #c00000;">**0.071**</span> → 0.095 | 22.65 → <span style="color: #c00000;">**21.02**</span> | <span style="color: #c00000;">**0.127**</span> → 0.145 | 80.2 → <span style="color: #c00000;">**43.3**</span> | 15.7 → <span style="color: #c00000;">**59.7**</span> |
| Half-bound A / -2 | 0.717 → <span style="color: #c00000;">**0.643**</span> | <span style="color: #c00000;">**0.113**</span> → 0.172 | 35.85 → <span style="color: #c00000;">**32.17**</span> | <span style="color: #c00000;">**0.133**</span> → 0.174 | <span style="color: #c00000;">**48.4**</span> → 121.9 | <span style="color: #c00000;">**39.9**</span> → -15.2 |
| Half-bound B / +2 | 0.551 → <span style="color: #c00000;">**0.490**</span> | 0.066 → <span style="color: #c00000;">**0.065**</span> | 27.57 → <span style="color: #c00000;">**24.51**</span> | 0.140 → <span style="color: #c00000;">**0.128**</span> | 54.6 → <span style="color: #c00000;">**25.5**</span> | 45.0 → <span style="color: #c00000;">**70.3**</span> |
| Half-bound B / -2 | N/A → 0.392 | N/A → 0.274 | N/A → 19.62 | N/A → 0.305 | N/A → 117.8 | N/A → -22.8 |
| Rotary A / +2 | 0.546 → <span style="color: #c00000;">**0.514**</span> | 0.129 → <span style="color: #c00000;">**0.094**</span> | 27.29 → <span style="color: #c00000;">**25.71**</span> | <span style="color: #c00000;">**0.101**</span> → 0.138 | <span style="color: #c00000;">**43.6**</span> → 61.5 | <span style="color: #c00000;">**56.6**</span> → 37.1 |
| Rotary A / -2 | 0.696 → <span style="color: #c00000;">**0.363**</span> | <span style="color: #c00000;">**0.138**</span> → 0.155 | 34.79 → <span style="color: #c00000;">**18.16**</span> | <span style="color: #c00000;">**0.151**</span> → 0.196 | <span style="color: #c00000;">**128.5**</span> → 130.3 | <span style="color: #c00000;">**-26.5**</span> → -37.0 |
| Rotary B / +2 | 0.432 → <span style="color: #c00000;">**0.384**</span> | 0.145 → <span style="color: #c00000;">**0.108**</span> | 21.62 → <span style="color: #c00000;">**19.18**</span> | 0.136 → <span style="color: #c00000;">**0.094**</span> | 94.2 → <span style="color: #c00000;">**39.7**</span> | 8.2 → <span style="color: #c00000;">**64.4**</span> |
| Rotary B / -2 | 0.699 → <span style="color: #c00000;">**0.656**</span> | <span style="color: #c00000;">**0.109**</span> → 0.143 | 34.94 → <span style="color: #c00000;">**32.80**</span> | <span style="color: #c00000;">**0.110**</span> → 0.205 | <span style="color: #c00000;">**5.6**</span> → 131.0 | <span style="color: #c00000;">**63.8**</span> → -31.7 |
| Transverse A / +2 | <span style="color: #c00000;">**0.397**</span> → 0.430 | <span style="color: #c00000;">**0.086**</span> → 0.176 | <span style="color: #c00000;">**19.87**</span> → 21.48 | 0.152 → <span style="color: #c00000;">**0.132**</span> | <span style="color: #c00000;">**10.3**</span> → 85.2 | <span style="color: #c00000;">**79.0**</span> → 5.6 |
| Transverse A / -2 | 0.707 → <span style="color: #c00000;">**0.573**</span> | <span style="color: #c00000;">**0.173**</span> → 0.210 | 35.37 → <span style="color: #c00000;">**28.66**</span> | 0.138 → <span style="color: #c00000;">**0.122**</span> | <span style="color: #c00000;">**30.4**</span> → 61.2 | <span style="color: #c00000;">**52.2**</span> → 43.7 |
| Transverse B / +2 | <span style="color: #c00000;">**0.494**</span> → 0.520 | 0.096 → <span style="color: #c00000;">**0.053**</span> | <span style="color: #c00000;">**24.69**</span> → 25.98 | 0.161 → <span style="color: #c00000;">**0.134**</span> | 87.9 → <span style="color: #c00000;">**10.9**</span> | 12.5 → <span style="color: #c00000;">**73.6**</span> |
| Transverse B / -2 | <span style="color: #c00000;">**0.655**</span> → 0.656 | <span style="color: #c00000;">**0.176**</span> → 0.190 | <span style="color: #c00000;">**32.75**</span> → 32.80 | 0.302 → <span style="color: #c00000;">**0.109**</span> | 79.0 → <span style="color: #c00000;">**46.6**</span> | 21.6 → <span style="color: #c00000;">**52.7**</span> |

**Contact timing and execution**

| Gait / vx command [m/s] | [Agreement](#metric-contact-agreement) [%] ↑ | [Touchdown error](#metric-touchdown-error) [ms] ↓ | [Measured duty](#metric-measured-duty) [%], target 56 | [Duty error](#metric-duty-error) [pp] ↓ | [Swing contact](#metric-swing-contact) [%] ↓ | [Missed stance](#metric-missed-stance) [%] ↓ | [Max-foot TD/cycle](#metric-touchdowns-per-cycle) ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 93.6 → <span style="color: #c00000;">**95.6**</span> | 15.0 → <span style="color: #c00000;">**11.8**</span> | 51.75 → <span style="color: #c00000;">**52.15**</span> | 5.20 → <span style="color: #c00000;">**3.85**</span> | 2.50 → <span style="color: #c00000;">**0.84**</span> | 9.16 → <span style="color: #c00000;">**6.86**</span> | 1.74 → <span style="color: #c00000;">**1.04**</span> |
| Trot / -2 | N/A → 90.6 | N/A → 12.4 | N/A → 48.48 | N/A → 7.52 | N/A → 1.24 | N/A → 15.17 | N/A → 1.10 |
| Bound / +2 | 79.8 → <span style="color: #c00000;">**80.1**</span> | 38.6 → <span style="color: #c00000;">**24.8**</span> | 54.66 → <span style="color: #c00000;">**55.04**</span> | <span style="color: #c00000;">**6.28**</span> → 6.73 | <span style="color: #c00000;">**20.26**</span> → 22.19 | 20.11 → <span style="color: #c00000;">**18.22**</span> | <span style="color: #c00000;">**1.00**</span> → 1.97 |
| Bound / -2 | <span style="color: #c00000;">**86.9**</span> → 79.2 | 35.6 → <span style="color: #c00000;">**20.0**</span> | 49.11 → <span style="color: #c00000;">**50.00**</span> | <span style="color: #c00000;">**6.89**</span> → 10.69 | <span style="color: #c00000;">**8.22**</span> → 17.50 | <span style="color: #c00000;">**16.54**</span> → 23.21 | <span style="color: #c00000;">**1.16**</span> → 2.00 |
| Half-bound A / +2 | <span style="color: #c00000;">**90.0**</span> → 89.3 | 37.0 → <span style="color: #c00000;">**13.8**</span> | 52.86 → <span style="color: #c00000;">**53.44**</span> | 5.51 → <span style="color: #c00000;">**2.93**</span> | <span style="color: #c00000;">**7.57**</span> → 7.79 | <span style="color: #c00000;">**11.77**</span> → 12.84 | <span style="color: #c00000;">**1.55**</span> → 2.00 |
| Half-bound A / -2 | 88.6 → <span style="color: #c00000;">**91.6**</span> | <span style="color: #c00000;">**19.9**</span> → 20.0 | <span style="color: #c00000;">**50.99**</span> → 50.97 | 6.71 → <span style="color: #c00000;">**5.87**</span> | 7.32 → <span style="color: #c00000;">**4.97**</span> | 14.50 → <span style="color: #c00000;">**10.96**</span> | 2.00 → <span style="color: #c00000;">**1.04**</span> |
| Half-bound B / +2 | 83.1 → <span style="color: #c00000;">**90.8**</span> | 48.5 → <span style="color: #c00000;">**33.0**</span> | <span style="color: #c00000;">**53.76**</span> → 52.80 | <span style="color: #c00000;">**6.17**</span> → 7.06 | 16.22 → <span style="color: #c00000;">**7.89**</span> | 17.34 → <span style="color: #c00000;">**10.12**</span> | <span style="color: #c00000;">**1.28**</span> → 1.45 |
| Half-bound B / -2 | N/A → 86.0 | N/A → 15.6 | N/A → 51.19 | N/A → 6.36 | N/A → 8.19 | N/A → 18.41 | N/A → 2.00 |
| Rotary A / +2 | 83.7 → <span style="color: #c00000;">**88.6**</span> | 19.9 → <span style="color: #c00000;">**18.2**</span> | <span style="color: #c00000;">**54.81**</span> → 53.03 | 5.88 → <span style="color: #c00000;">**5.80**</span> | 17.48 → <span style="color: #c00000;">**9.62**</span> | 15.45 → <span style="color: #c00000;">**12.76**</span> | 2.00 → 2.00 |
| Rotary A / -2 | <span style="color: #c00000;">**89.1**</span> → 82.3 | <span style="color: #c00000;">**26.3**</span> → 32.6 | <span style="color: #c00000;">**52.40**</span> → 47.76 | <span style="color: #c00000;">**4.44**</span> → 9.09 | <span style="color: #c00000;">**8.33**</span> → 10.36 | <span style="color: #c00000;">**12.96**</span> → 23.49 | 2.00 → 2.00 |
| Rotary B / +2 | 79.0 → <span style="color: #c00000;">**87.9**</span> | 31.3 → <span style="color: #c00000;">**19.5**</span> | 53.72 → <span style="color: #c00000;">**55.73**</span> | 3.07 → <span style="color: #c00000;">**2.97**</span> | 20.24 → <span style="color: #c00000;">**13.82**</span> | 21.58 → <span style="color: #c00000;">**10.82**</span> | 2.36 → <span style="color: #c00000;">**2.00**</span> |
| Rotary B / -2 | 81.9 → <span style="color: #c00000;">**90.3**</span> | 37.4 → <span style="color: #c00000;">**33.0**</span> | <span style="color: #c00000;">**56.89**</span> → 48.72 | <span style="color: #c00000;">**6.35**</span> → 7.28 | 21.03 → <span style="color: #c00000;">**4.76**</span> | 15.84 → <span style="color: #c00000;">**13.58**</span> | 2.00 → 2.00 |
| Transverse A / +2 | <span style="color: #c00000;">**92.1**</span> → 86.3 | <span style="color: #c00000;">**34.4**</span> → 45.8 | <span style="color: #c00000;">**52.52**</span> → 52.01 | 6.88 → <span style="color: #c00000;">**5.37**</span> | <span style="color: #c00000;">**6.74**</span> → 11.82 | <span style="color: #c00000;">**8.87**</span> → 15.12 | <span style="color: #c00000;">**1.19**</span> → 2.01 |
| Transverse A / -2 | 91.7 → <span style="color: #c00000;">**92.6**</span> | <span style="color: #c00000;">**15.0**</span> → 20.0 | <span style="color: #c00000;">**51.92**</span> → 50.86 | <span style="color: #c00000;">**4.92**</span> → 5.78 | 7.14 → <span style="color: #c00000;">**4.76**</span> | <span style="color: #c00000;">**9.26**</span> → 9.45 | 2.00 → <span style="color: #c00000;">**1.00**</span> |
| Transverse B / +2 | 87.1 → <span style="color: #c00000;">**92.3**</span> | <span style="color: #c00000;">**26.5**</span> → 28.2 | 53.15 → <span style="color: #c00000;">**53.19**</span> | <span style="color: #c00000;">**5.56**</span> → 6.54 | 11.50 → <span style="color: #c00000;">**7.72**</span> | 14.01 → <span style="color: #c00000;">**7.69**</span> | <span style="color: #c00000;">**1.03**</span> → 1.20 |
| Transverse B / -2 | 87.4 → <span style="color: #c00000;">**92.7**</span> | 23.0 → <span style="color: #c00000;">**16.2**</span> | 47.20 → <span style="color: #c00000;">**49.04**</span> | 8.80 → <span style="color: #c00000;">**6.96**</span> | 5.24 → <span style="color: #c00000;">**2.38**</span> | 18.32 → <span style="color: #c00000;">**11.11**</span> | 1.12 → <span style="color: #c00000;">**1.00**</span> |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | [Power](#metric-mechanical-power) [W] ↓* | [Normalized torque cost](#metric-normalized-torque-cost) ↓* | [Clipping](#metric-action-clipping) [%] ↓ | [Work imbalance](#metric-front-rear-imbalance) [%] ↓† | [Force imbalance](#metric-front-rear-imbalance) [%] ↓† | [Torque imbalance](#metric-front-rear-imbalance) [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | <span style="color: #c00000;">**143.3**</span> → 159.6 | 0.522 → <span style="color: #c00000;">**0.463**</span> | <span style="color: #c00000;">**2.28**</span> → 2.61 | 13.96 → <span style="color: #c00000;">**6.76**</span> | <span style="color: #c00000;">**10.01**</span> → 11.06 | 20.72 → <span style="color: #c00000;">**20.37**</span> |
| Trot / -2 | N/A → 144.3 | N/A → 0.524 | N/A → 2.74 | N/A → 20.86 | N/A → 31.50 | N/A → 54.24 |
| Bound / +2 | <span style="color: #c00000;">**208.7**</span> → 242.6 | <span style="color: #c00000;">**0.631**</span> → 0.668 | <span style="color: #c00000;">**3.60**</span> → 4.26 | <span style="color: #c00000;">**5.53**</span> → 6.78 | <span style="color: #c00000;">**17.08**</span> → 17.88 | <span style="color: #c00000;">**10.73**</span> → 18.72 |
| Bound / -2 | <span style="color: #c00000;">**223.2**</span> → 336.9 | <span style="color: #c00000;">**0.673**</span> → 0.817 | <span style="color: #c00000;">**2.24**</span> → 5.13 | <span style="color: #c00000;">**4.48**</span> → 35.20 | 39.22 → <span style="color: #c00000;">**38.91**</span> | 39.23 → <span style="color: #c00000;">**30.51**</span> |
| Half-bound A / +2 | <span style="color: #c00000;">**214.0**</span> → 302.6 | <span style="color: #c00000;">**0.674**</span> → 0.758 | <span style="color: #c00000;">**3.23**</span> → 3.43 | <span style="color: #c00000;">**8.33**</span> → 10.27 | 21.01 → <span style="color: #c00000;">**18.09**</span> | <span style="color: #c00000;">**15.88**</span> → 21.46 |
| Half-bound A / -2 | <span style="color: #c00000;">**194.5**</span> → 290.2 | <span style="color: #c00000;">**0.614**</span> → 0.640 | 3.21 → <span style="color: #c00000;">**1.28**</span> | <span style="color: #c00000;">**9.41**</span> → 36.35 | <span style="color: #c00000;">**44.63**</span> → 49.15 | 55.55 → <span style="color: #c00000;">**33.79**</span> |
| Half-bound B / +2 | <span style="color: #c00000;">**153.1**</span> → 217.0 | <span style="color: #c00000;">**0.589**</span> → 0.747 | <span style="color: #c00000;">**1.82**</span> → 2.29 | 14.30 → <span style="color: #c00000;">**0.89**</span> | <span style="color: #c00000;">**17.13**</span> → 22.48 | 25.86 → <span style="color: #c00000;">**23.37**</span> |
| Half-bound B / -2 | N/A → 321.0 | N/A → 0.760 | N/A → 5.13 | N/A → 32.50 | N/A → 40.59 | N/A → 31.24 |
| Rotary A / +2 | <span style="color: #c00000;">**180.0**</span> → 206.3 | <span style="color: #c00000;">**0.537**</span> → 0.672 | 3.85 → <span style="color: #c00000;">**3.57**</span> | 23.95 → <span style="color: #c00000;">**0.59**</span> | <span style="color: #c00000;">**15.72**</span> → 19.70 | 40.39 → <span style="color: #c00000;">**18.58**</span> |
| Rotary A / -2 | <span style="color: #c00000;">**150.0**</span> → 309.3 | <span style="color: #c00000;">**0.643**</span> → 0.773 | <span style="color: #c00000;">**1.92**</span> → 4.18 | 30.54 → <span style="color: #c00000;">**29.46**</span> | 37.34 → <span style="color: #c00000;">**37.28**</span> | 64.16 → <span style="color: #c00000;">**39.71**</span> |
| Rotary B / +2 | <span style="color: #c00000;">**233.8**</span> → 247.3 | <span style="color: #c00000;">**0.622**</span> → 0.645 | 4.16 → <span style="color: #c00000;">**2.42**</span> | 22.81 → <span style="color: #c00000;">**1.41**</span> | <span style="color: #c00000;">**8.99**</span> → 22.22 | 31.46 → <span style="color: #c00000;">**27.04**</span> |
| Rotary B / -2 | <span style="color: #c00000;">**197.4**</span> → 232.7 | 0.659 → <span style="color: #c00000;">**0.595**</span> | 2.57 → <span style="color: #c00000;">**2.29**</span> | <span style="color: #c00000;">**0.78**</span> → 25.95 | <span style="color: #c00000;">**43.92**</span> → 50.31 | <span style="color: #c00000;">**44.62**</span> → 53.38 |
| Transverse A / +2 | 207.6 → <span style="color: #c00000;">**170.8**</span> | 0.592 → <span style="color: #c00000;">**0.528**</span> | <span style="color: #c00000;">**2.11**</span> → 3.57 | 7.05 → <span style="color: #c00000;">**2.44**</span> | 20.23 → <span style="color: #c00000;">**19.03**</span> | <span style="color: #c00000;">**17.25**</span> → 21.79 |
| Transverse A / -2 | 179.9 → <span style="color: #c00000;">**166.2**</span> | 0.601 → <span style="color: #c00000;">**0.570**</span> | <span style="color: #c00000;">**1.60**</span> → 2.56 | <span style="color: #c00000;">**11.39**</span> → 13.84 | <span style="color: #c00000;">**39.08**</span> → 40.29 | <span style="color: #c00000;">**55.35**</span> → 55.75 |
| Transverse B / +2 | <span style="color: #c00000;">**158.1**</span> → 194.9 | <span style="color: #c00000;">**0.590**</span> → 0.667 | 3.22 → <span style="color: #c00000;">**2.79**</span> | <span style="color: #c00000;">**1.23**</span> → 9.33 | <span style="color: #c00000;">**14.74**</span> → 27.25 | <span style="color: #c00000;">**14.18**</span> → 29.40 |
| Transverse B / -2 | 167.9 → <span style="color: #c00000;">**158.4**</span> | 0.632 → <span style="color: #c00000;">**0.543**</span> | 2.24 → 2.24 | 24.34 → <span style="color: #c00000;">**5.58**</span> | <span style="color: #c00000;">**32.26**</span> → 40.96 | 59.82 → <span style="color: #c00000;">**58.50**</span> |

**Classification, success, and completion**

| Gait / vx command [m/s] | [Exact template](#metric-gait-classification) [%] ↑ | [Correct family](#metric-gait-classification) [%] ↑ | [Velocity success](#metric-velocity-success) [%] ↑ | [Gait success](#metric-gait-success) [%] ↑ | [Combined success](#metric-combined-success) [%] ↑ | [Positive progress](#metric-positive-progress) [%] ↑ | [Completion](#metric-completion) [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> |
| Bound / +2 | 0.0 → <span style="color: #c00000;">**97.0**</span> | 0.0 → <span style="color: #c00000;">**97.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**97.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound A / +2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Half-bound B / +2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> |
| Rotary A / +2 | <span style="color: #c00000;">**100.0**</span> → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | <span style="color: #c00000;">**100.0**</span> → 35.0 | <span style="color: #c00000;">**100.0**</span> → 35.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 35.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 0.0 | 100.0 → 100.0 |
| Transverse A / +2 | <span style="color: #c00000;">**88.0**</span> → 0.0 | <span style="color: #c00000;">**88.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**88.0**</span> → 0.0 | 0.0 → 0.0 | <span style="color: #c00000;">**95.0**</span> → 92.0 | 95.0 → <span style="color: #c00000;">**100.0**</span> |
| Transverse A / -2 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 0.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +2 | <span style="color: #c00000;">**100.0**</span> → 99.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 | 0.0 → 0.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 | <span style="color: #c00000;">**100.0**</span> → 99.0 |
| Transverse B / -2 | 10.0 → <span style="color: #c00000;">**100.0**</span> | 10.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 10.0 → <span style="color: #c00000;">**100.0**</span> | 0.0 → 0.0 | 10.0 → <span style="color: #c00000;">**100.0**</span> | 10.0 → <span style="color: #c00000;">**100.0**</span> |

</details>

## Source manifests and audit provenance

<details>
<summary>Show source manifests, checkpoint hashes, and audit provenance</summary>

The ±1 studies contribute 1,600 trials per policy; the new six-speed studies contribute 4,800 per policy. Checkpoint hashes match between the two studies for each policy. Both manifest links are supplied so this report does not depend on temporary `/tmp` analysis files.

### Study manifests

#### X1

| Method | Commands [m/s] | Manifest | Manifest SHA-256 |
| ---: | ---: | ---: | ---: |
| No TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `7e940d4610351978d818463e5c95f40afdf1c9931bd57b6625f5fba2778d46ca` |
| No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `8919400a11e9fb46de58eb67cc6cc2f3a53c45456199c92413147f4697b948f6` |
| TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `27043c1e350b97c80991f0c504d7ca62f2150e9518a5785c071ae67f99d3ebac` |
| TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `6394355593d51063ed1f396c670a0aa052e5adcfc284657bfabf80bcc9d4dbd2` |

#### X1_go2bend

| Method | Commands [m/s] | Manifest | Manifest SHA-256 |
| ---: | ---: | ---: | ---: |
| No TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `1f2f2f6163433b930151d25120b56878990b171e3de87233c53f443b8073ce5d` |
| No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `8ec8954286b92e02dab499ef3d8265581d8088eb7f7193f4a2f18df54d4cfa8c` |
| TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `bc5a07fe81ef46243ef4a3e2d779f57853dc4452910c587bf354fc580e604abd` |
| TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `e4a111a42211923d515591550e88b7cc19409f05640789ce215dcb4738daa2b0` |

#### Go2

| Method | Commands [m/s] | Manifest | Manifest SHA-256 |
| ---: | ---: | ---: | ---: |
| No TRS | ±1 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `1fcda0349bb01b5d88fe36ba93abf6c7992e4ad8b10c179e1b5560629d7d66b7` |
| No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `a379f3b626354d46421308a0b5e68748f6a441a5288572217a9ed1545c75483b` |
| TRS | ±1 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `5f2ff12adb02e5809fc138d8472d51f3e68826e7c0465519c33171d89b26d8fa` |
| TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `1a4f70eec0fa5880fa464119e03e3e4a6536b3035e48bae7b3059e3f53b55b15` |

### Checkpoint hashes

#### X1

| Method | Checkpoint SHA-256 |
| ---: | ---: |
| No TRS | `127eb75070436c60ef8a610022c4fb12032b1b92f96f0445bc5e9be8bf4bd796` |
| TRS | `b9f9b4e942e03f8866737f943a733e6062b26d4a693e34622616f6d0463ab6d2` |

#### X1_go2bend

| Method | Checkpoint SHA-256 |
| ---: | ---: |
| No TRS | `6cede8964978146ba403d4fa4169189501e96a09607422edb3aca3e04de3d703` |
| TRS | `d23818ac5c17b3fbfad88a8195a58c77c604abbe19fac88f3294d15e04a82da8` |

#### Go2

| Method | Checkpoint SHA-256 |
| ---: | ---: |
| No TRS | `0430dde06881765effd0a1ffaf171dc9e117b3277cbd04fd7a7e652370346f71` |
| TRS | `1c437c00cf0ca34b8bdac5d83b3da9489306bd99abb604d0f05e31cfaef63eb9` |

Source CSV locations inside each linked study: `metrics/cell_metrics.csv`; each trial's archive and runtime status live at its manifest `relative_output_dir`. The report was generated from audited comparison CSVs containing 12 speed-pair rows, 24 direction rows, 96 gait-pair rows, and 192 gait/direction rows. Matching of time-reversed speed pairs uses the study's saved time-reversal partner indices.

</details>
