<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# TRS versus no TRS: X1, X1_go2bend, and Go2 velocity-sweep evaluation

Saved: **2026-10-22**. Checkpoint: **model_9999.pt** for all six policies.

## Benefits of TRS

The strongest measured benefits are better moderate/high-speed tracking, fewer early terminations at the highest tested speed, and closer agreement with commanded stance/swing timing. The benefits below refer to averages across gaits and directions; they do not imply improvement for every individual gait.

| Benefit | Evidence: No TRS → TRS | Scope |
| --- | --- | --- |
| **Better speed tracking** | At ±1.5 m/s, vx RMSE falls **0.108 → 0.080 m/s** for X1, **0.100 → 0.086 m/s** for X1_go2bend, and **0.219 → 0.171 m/s** for Go2. | All three robot configurations improve at this speed, with every rollout completing. |
| **More completed high-speed rollouts** | At ±2 m/s, completion rises **94.31% → 99.94%** for X1 and **81.56% → 99.94%** for Go2. | Fewer environment-triggered early terminations under these nominal evaluations; both X1_go2bend policies already complete 100%. |
| **Better forward/backward speed consistency** | At ±1.5 m/s, matched TR-partner speed bias falls **0.119 → 0.047 m/s** for X1 and **0.185 → 0.049 m/s** for Go2. | Both improve at ±1, ±1.5, and ±2; this measures speed consistency, not full trajectory symmetry. |
| **Closer contact-schedule tracking** | Contact agreement improves in **all 12 robot/speed-pair comparisons**. Go2 touchdown timing error improves at all four speeds, including **24.3 → 17.3 ms** at ±1.5. | Pooled contact diagnostics; touchdown timing does not directly establish synchronization between legs or correct gait identity. |
| **Fewer action-constraint interventions and lower X1 torque cost** | Action clipping decreases at all four speeds for both X1 configurations. At ±1, X1 clipping falls **0.63% → 0.06%** and X1_go2bend clipping falls **3.39% → 2.19%**. X1 normalized torque cost decreases at all four speeds. | Less clipping and lower normalized effort in these conditions; this alone does not establish smoother motion or electrical efficiency. |
| **Better heading retention for X1_go2bend** | At ±1.5, heading RMSE falls **46.8° → 20.9°**, and directed world progress rises **62.2% → 88.5%** of the requested distance. | Heading and progress improve at all four speeds for X1_go2bend; this benefit does not extend to high-speed Go2. |

**TRS improves several tracking, contact, and completion outcomes, but the benefit depends on the robot, command speed, and gait. The data do not support universal improvements in gait fidelity, mechanical efficiency, or world-frame direction control.**

These are comparisons of one trained policy per method and robot, with nominal evaluation replicas. They do not establish statistical significance across training seeds. The reported training-reward and convergence-speed advantages are separate observations and are not quantified by this evaluation report.

This report combines the six-speed sweep from `evaluation_batch.sh` with the existing ±1 m/s statistical evaluations. It saves the main comparison, all additional metric groups, termination details, matched-completion checks, and every trained gait in both directions. It supersedes the earlier ±1-only comparison for this wider command grid; it does not include lateral or yaw command tests.

## Contents

- [Benefits of TRS](#benefits-of-trs)
- [Conclusions](#conclusions)
- [Runs and evaluation protocol](#runs-and-evaluation-protocol)
- [How to read the tables](#how-to-read-the-tables)
- [Tracking, gait fidelity, and completion](#tracking-gait-fidelity-and-completion)
- [Additional paired metrics](#additional-paired-metrics)
- [Forward/backward speed consistency](#forwardbackward-speed-consistency)
- [Early terminations and matched-completion checks](#early-terminations-and-matched-completion-checks)
- [Separate positive and negative commands](#separate-positive-and-negative-commands)
- [Metric definitions](#metric-definitions)
- [Corrections and limitations](#corrections-and-limitations)
- [Appendix A: each gait, pooled over both directions](#appendix-a-each-gait-pooled-over-both-directions)
- [Appendix B: all metrics for every gait and direction](#appendix-b-all-metrics-for-every-gait-and-direction)
- [Source manifests and audit provenance](#source-manifests-and-audit-provenance)

## Conclusions

- **X1:** TRS reduces pooled speed RMSE at ±1, ±1.5, and ±2 m/s, and improves completion at ±2 from 94.31% to 99.94%. Action clipping and normalized torque cost decrease at all four speeds. Touchdown timing and exact gait-template fidelity do not consistently improve. Heading/progress improve at the two lower speeds and worsen at the two higher speeds.
- **X1_go2bend:** both methods complete every trial. TRS improves contact agreement, heading retention, and directed world progress at all four speeds, with the strongest heading/progress benefit at ±1.5. Speed tracking is mixed at lower speeds and improves when pooled at ±1.5 and ±2. Mechanical power usually increases, and exact template matching falls at the two higher speeds.
- **Go2:** TRS improves pooled speed RMSE from ±1 onward and raises ±2 completion from 81.56% to 99.94%. Touchdown timing and contact agreement improve at every speed. At ±1.5 and ±2, yaw-rate error, accumulated heading drift, and mechanical power increase; directed world progress decreases. Improved completion therefore coexists with poorer world-frame path control.
- **Extra contacts remain an issue:** at ±1.5, the mean per-trial maximum touchdown count per foot per cycle changes from 1.24 to 1.04 for X1, 1.29 to 1.41 for X1_go2bend, and 1.32 to 1.51 for Go2. Better agreement with commanded stance does not guarantee fewer repeated contacts.
- **Forward/backward speed symmetry:** X1 and Go2 improve from ±1 onward. X1_go2bend improves only at ±2 under the time-reversed-partner speed-bias metric. This metric is narrower than full trajectory or action symmetry.

A defensible interpretation is that actor TRS improves several locomotion outcomes in these trained policies, particularly moderate/high-speed tracking and high-speed rollout completion. Gait identity, timing, heading, and effort must still be assessed separately. Training reward or convergence advantages are not quantified by this evaluation report.

## Runs and evaluation protocol

| Robot | Method | Training run |
| --- | ---: | ---: |
| X1 | No TRS | [2026-10-19_05-22-27_x1_no_trs](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs) |
| X1 | Actor TRS 0.1 | [2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |
| X1_go2bend | No TRS | [2026-10-21_08-45-56_x1_go2bend_no_trs](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs) |
| X1_go2bend | Actor TRS 0.1 | [2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |
| Go2 | No TRS | [2026-10-19_09-33-24_go2_no_trs](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs) |
| Go2 | Actor TRS 0.1 | [2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1) |

- Commands: **vx = −2, −1.5, −1, −0.5, +0.5, +1, +1.5, +2 m/s**. All have **vy = 0** and **yaw-rate command = 0**.
- Eight trained templates: trot, bound, half-bound A/B, rotary A/B, transverse A/B. A/B identify command templates; they do not assert that the observed motion has that gait identity.
- **100 parallel nominal replicas per gait/command**, giving 6,400 trials per policy, 12,800 per robot, and **38,400 total**. There are 1,600 planned trials per method per ±speed pair and 100 per individual gait/direction.
- Control step: **0.02 s**. Settling: **250 steps / 5 s**. Measured interval: **1,000 steps / 20 s**. Full rollout: **1,250 steps / 25 s**.
- Complete-cycle metric windows: **960 samples / 19.2 s / 32 cycles** for both X1 configurations; **988 samples / 19.76 s / 38 cycles** for Go2.
- Fixed cycle duration: **0.60 s** for X1 and X1_go2bend; **0.52 s** for Go2. Commanded stance duty factor: **0.56** for all robots.
- TRS uses actor coefficient **0.1**, value coefficient **0**, and warmup **0**. The run-name field `mirror0p1` refers to this actor TRS coefficient in this implementation.
- The earlier ±1 study and the new six-speed study use the same checkpoint, saved configuration provenance, runtime overrides, gait library, nominal profile, and timing within each policy. Recorded source changes between those studies are confined to evaluation folder-layout handling.
- One training seed per method; evaluation replicas are nominal initializations, not independent training seeds or randomized robustness trials.

## How to read the tables

Every paired entry is **No TRS → TRS**. **Bold** marks the preferred displayed value; ties at the displayed precision are unbolded. `N/A` means that no full rollout was available for that method in the displayed condition, so its continuous metric cannot be compared.

- **↓ / ↑:** lower / higher is preferred for that diagnostic.
- **↓\*:** lower actuator effort is preferable only when tracking, gait execution, completion, and displacement are comparable.
- **↓†:** lower front/rear imbalance means more even loading; it is not universally optimal for every gait.
- **≈100% / ≈1:** closer to 100% directed progress / one touchdown per foot per cycle is preferred.
- Continuous metrics are arithmetic means of **per-trial metrics from full rollouts only**, rather than RMSE calculated after pooling all time samples. Failed/partial rollouts are excluded from these means and retained in completion and success denominators.
- Pair tables pool all completed trials across eight templates and both signs. When every trial completes, templates and directions have equal weight. At ±2, survivor populations differ, so their continuous means are conditional on survival; see the matched-completion table.
- Classifier/success percentages use **all planned trials** as their denominator. An early termination counts as unsuccessful. These rates are not conditioned on survival.

## Tracking, gait fidelity, and completion

Forward and backward vx errors are shown separately; yaw, contact agreement, classification, and completion pool both directions. Yaw-rate RMSE has units rad/s, and contact/classifier/completion values are percentages.

### X1

| Command pair [m/s] | +vx RMSE [m/s] ↓ | −vx RMSE [m/s] ↓ | Yaw-rate RMSE ↓ | Contact agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | **0.050** → 0.052 | 0.047 → **0.046** | 0.072 → **0.056** | 94.6 → **95.7** | **81.1** → 75.0 | **81.1** → 75.0 | 100.00 → 100.00 |
| ±1 | **0.070** → 0.085 | 0.101 → **0.064** | 0.079 → **0.064** | 92.4 → **93.2** | 75.0 → 75.0 | 75.0 → 75.0 | 100.00 → 100.00 |
| ±1.5 | 0.094 → **0.073** | 0.123 → **0.087** | 0.100 → **0.092** | 87.5 → **87.6** | 50.0 → 50.0 | 52.7 → **62.2** | 100.00 → 100.00 |
| ±2 | 0.180 → **0.163** | 0.181 → **0.158** | 0.157 → **0.144** | 80.2 → **81.2** | **40.4** → 30.9 | 45.0 → **55.6** | 94.31 → **99.94** |

### X1_go2bend

| Command pair [m/s] | +vx RMSE [m/s] ↓ | −vx RMSE [m/s] ↓ | Yaw-rate RMSE ↓ | Contact agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | 0.060 → **0.052** | **0.057** → 0.083 | 0.089 → **0.066** | 91.9 → **94.3** | 50.0 → **68.8** | 50.0 → **68.8** | 100.00 → 100.00 |
| ±1 | **0.076** → 0.090 | **0.074** → 0.089 | 0.094 → **0.090** | 89.5 → **91.6** | 43.8 → **50.0** | **56.2** → 50.0 | 100.00 → 100.00 |
| ±1.5 | **0.074** → 0.077 | 0.126 → **0.095** | 0.110 → **0.084** | 84.9 → **86.2** | **37.5** → 31.2 | **50.0** → 37.5 | 100.00 → 100.00 |
| ±2 | 0.100 → **0.094** | 0.287 → **0.216** | 0.134 → **0.105** | 78.5 → **79.4** | **43.6** → 37.6 | **56.1** → 50.1 | 100.00 → 100.00 |

### Go2

| Command pair [m/s] | +vx RMSE [m/s] ↓ | −vx RMSE [m/s] ↓ | Yaw-rate RMSE ↓ | Contact agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ±0.5 | **0.043** → 0.049 | **0.045** → 0.047 | **0.064** → 0.065 | 95.8 → **96.5** | 93.8 → 93.8 | 93.8 → 93.8 | 100.00 → 100.00 |
| ±1 | 0.071 → **0.066** | 0.095 → **0.061** | **0.073** → 0.083 | 93.4 → **94.1** | 68.8 → **75.0** | 68.8 → **75.0** | 100.00 → 100.00 |
| ±1.5 | **0.144** → 0.156 | 0.294 → **0.187** | **0.102** → 0.123 | 90.1 → **90.9** | 53.5 → **68.8** | 53.5 → **68.8** | 100.00 → 100.00 |
| ±2 | 0.472 → **0.459** | 0.714 → **0.528** | **0.132** → 0.153 | 86.6 → **88.5** | 31.1 → **64.4** | 31.1 → **64.4** | 81.56 → **99.94** |

## Additional paired metrics

### Pooled velocity errors

| Robot | Command pair [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ |
| --- | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | 0.049 → 0.049 | 0.041 → **0.035** | **9.73** → 9.75 |
| X1 | ±1 | 0.085 → **0.075** | 0.041 → **0.040** | 8.54 → **7.48** |
| X1 | ±1.5 | 0.108 → **0.080** | 0.057 → 0.057 | 7.23 → **5.31** |
| X1 | ±2 | 0.181 → **0.160** | **0.086** → 0.087 | 9.03 → **8.00** |
| X1_go2bend | ±0.5 | **0.059** → 0.067 | **0.044** → 0.045 | **11.76** → 13.43 |
| X1_go2bend | ±1 | **0.075** → 0.089 | **0.045** → 0.056 | **7.48** → 8.91 |
| X1_go2bend | ±1.5 | 0.100 → **0.086** | 0.069 → **0.067** | 6.67 → **5.73** |
| X1_go2bend | ±2 | 0.194 → **0.155** | 0.101 → **0.079** | 9.68 → **7.74** |
| Go2 | ±0.5 | **0.044** → 0.048 | **0.064** → 0.065 | **8.84** → 9.62 |
| Go2 | ±1 | 0.083 → **0.064** | 0.072 → **0.070** | 8.30 → **6.37** |
| Go2 | ±1.5 | 0.219 → **0.171** | 0.103 → **0.093** | 14.60 → **11.41** |
| Go2 | ±2 | 0.567 → **0.494** | **0.110** → 0.138 | 28.33 → **24.68** |

### Contact timing and duty tracking

Touchdown timing measures alignment with the commanded schedule, not the timing difference between two legs. Duty error is measured in percentage points (pp); measured duty is shown as a percentage with target 56%.

| Robot | Command pair [m/s] | Touchdown error [ms] ↓ | Duty error [pp] ↓ | Measured duty [%], target 56 |
| --- | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | **16.8** → 18.6 | 2.91 → **2.65** | 53.77 → **54.55** |
| X1 | ±1 | **21.2** → 23.2 | 4.98 → **4.49** | 51.54 → **51.65** |
| X1 | ±1.5 | 29.7 → **29.6** | **7.99** → 8.40 | **48.03** → 47.67 |
| X1 | ±2 | **38.3** → 41.0 | 12.58 → **11.73** | 43.43 → **44.29** |
| X1_go2bend | ±0.5 | 24.5 → **23.5** | 4.30 → **3.36** | 54.07 → **55.14** |
| X1_go2bend | ±1 | 32.8 → **28.8** | 5.47 → **4.50** | 51.70 → **52.44** |
| X1_go2bend | ±1.5 | **38.8** → 41.7 | **7.81** → 7.86 | **48.37** → 48.24 |
| X1_go2bend | ±2 | **43.1** → 46.8 | 12.53 → **12.47** | 43.49 → **43.53** |
| Go2 | ±0.5 | 13.8 → **13.0** | 4.90 → **4.49** | **57.46** → 57.96 |
| Go2 | ±1 | 16.8 → **15.7** | 5.48 → **4.49** | 55.50 → **55.90** |
| Go2 | ±1.5 | 24.3 → **17.3** | 5.96 → **5.50** | 53.15 → **53.33** |
| Go2 | ±2 | 29.6 → **22.8** | **5.70** → 6.30 | **52.92** → 51.52 |

### Contact errors and repeated contacts

The last column is the average, across trials, of the largest per-foot touchdown count divided by complete cycles within each trial. Approximately one is desired; it is not the maximum over the entire study.

| Robot | Command pair [m/s] | Contact during commanded swing [%] ↓ | Missed commanded stance [%] ↓ | Max-foot touchdowns/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | **3.72** → 3.78 | 6.71 → **4.62** | 1.00 → 1.00 |
| X1 | ±1 | 3.76 → **3.67** | 10.42 → **9.20** | 1.00 → 1.00 |
| X1 | ±1.5 | 5.63 → **5.23** | **17.57** → 17.71 | 1.24 → **1.04** |
| X1 | ±2 | 8.37 → **8.10** | 28.45 → **26.79** | 1.71 → **1.30** |
| X1_go2bend | ±0.5 | 6.98 → **5.21** | 8.88 → **6.02** | 1.00 → 1.00 |
| X1_go2bend | ±1 | 7.12 → **5.89** | 12.97 → **10.27** | **1.06** → 1.09 |
| X1_go2bend | ±1.5 | 8.73 → **7.51** | 19.87 → **18.48** | **1.29** → 1.41 |
| X1_go2bend | ±2 | 10.02 → **9.50** | 30.15 → **28.87** | 1.71 → **1.57** |
| Go2 | ±0.5 | 5.21 → **5.07** | 3.37 → **2.25** | 1.00 → 1.00 |
| Go2 | ±1 | 6.37 → **6.22** | 6.78 → **5.59** | **1.28** → 1.31 |
| Go2 | ±1.5 | 8.05 → **7.38** | 11.32 → **10.42** | **1.32** → 1.51 |
| Go2 | ±2 | 11.86 → **8.49** | 14.47 → **13.74** | 1.64 → **1.61** |

### Actuator effort and action clipping

Mechanical power is summed absolute joint mechanical power, not electrical power. Normalized torque cost is dimensionless. Lower clipping indicates fewer target-constraint interventions; it does not establish smooth motion.

| Robot | Command pair [m/s] | Mechanical power [W] ↓* | Normalized torque cost ↓* | Action clipping [%] ↓ |
| --- | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | **80.6** → 81.4 | 0.598 → **0.580** | 0.07 → **0.00** |
| X1 | ±1 | 134.3 → **124.2** | 0.819 → **0.758** | 0.63 → **0.06** |
| X1 | ±1.5 | 231.4 → **217.0** | 1.232 → **1.138** | 2.09 → **0.73** |
| X1 | ±2 | 312.4 → **300.0** | 1.516 → **1.471** | 3.84 → **2.22** |
| X1_go2bend | ±0.5 | **90.7** → 97.2 | 0.722 → **0.659** | 2.85 → **1.86** |
| X1_go2bend | ±1 | **142.8** → 150.8 | 0.900 → **0.852** | 3.39 → **2.19** |
| X1_go2bend | ±1.5 | **211.8** → 215.9 | **1.090** → 1.096 | 4.17 → **2.72** |
| X1_go2bend | ±2 | 293.6 → **288.9** | 1.281 → **1.264** | 4.60 → **2.91** |
| Go2 | ±0.5 | 75.5 → **66.2** | **0.277** → 0.279 | 1.22 → **0.00** |
| Go2 | ±1 | 103.0 → **100.1** | 0.382 → **0.372** | 1.35 → **0.30** |
| Go2 | ±1.5 | **145.9** → 170.1 | **0.522** → 0.532 | 1.86 → **1.39** |
| Go2 | ±2 | **187.7** → 231.3 | **0.611** → 0.648 | **2.75** → 3.16 |

### Front/rear load balance

Each imbalance is the absolute front/rear difference divided by their sum, expressed as a percentage. Values are averaged after taking each trial's absolute imbalance; signed differences are not allowed to cancel across directions.

| Robot | Command pair [m/s] | Work imbalance [%] ↓† | Vertical-force impulse imbalance [%] ↓† | Normalized torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | 5.43 → **4.08** | **3.13** → 3.62 | 11.83 → **7.53** |
| X1 | ±1 | 7.48 → **3.80** | 5.43 → **2.91** | 14.24 → **8.12** |
| X1 | ±1.5 | **7.90** → 10.76 | **7.52** → 7.63 | **8.46** → 10.21 |
| X1 | ±2 | **6.40** → 9.80 | **6.61** → 7.13 | **6.92** → 12.47 |
| X1_go2bend | ±0.5 | 24.63 → **21.45** | **8.79** → 12.57 | **7.02** → 9.33 |
| X1_go2bend | ±1 | 16.28 → **16.26** | **7.01** → 9.03 | **7.11** → 8.05 |
| X1_go2bend | ±1.5 | **9.53** → 9.79 | **5.40** → 7.41 | 9.13 → **6.00** |
| X1_go2bend | ±2 | **6.93** → 8.30 | 4.29 → **4.26** | 7.88 → **6.40** |
| Go2 | ±0.5 | 20.05 → **11.72** | 29.92 → **28.13** | 39.88 → **35.05** |
| Go2 | ±1 | 15.38 → **12.64** | 27.27 → **26.46** | 39.57 → **35.40** |
| Go2 | ±1.5 | **13.56** → 15.42 | **26.68** → 28.74 | 38.13 → **35.89** |
| Go2 | ±2 | **11.94** → 14.89 | **25.39** → 30.42 | 33.75 → **33.62** |

### Heading retention and directed world progress

Good body-frame velocity tracking can coexist with poor world progress if the robot turns away from its initial heading. Progress is displacement along the requested initial world direction, divided by requested distance.

| Robot | Command pair [m/s] | Heading RMSE [degrees] ↓ | Directed progress [%] ≈100 |
| --- | ---: | ---: | ---: |
| X1 | ±0.5 | 51.9 → **23.1** | 53.5 → **82.1** |
| X1 | ±1 | 43.6 → **39.7** | 61.4 → **66.2** |
| X1 | ±1.5 | **30.9** → 33.1 | **77.4** → 76.7 |
| X1 | ±2 | **43.8** → 46.5 | **60.2** → 58.2 |
| X1_go2bend | ±0.5 | 47.6 → **43.7** | 63.4 → **64.7** |
| X1_go2bend | ±1 | 43.2 → **42.1** | 64.3 → **66.4** |
| X1_go2bend | ±1.5 | 46.8 → **20.9** | 62.2 → **88.5** |
| X1_go2bend | ±2 | 41.9 → **33.2** | 63.0 → **72.7** |
| Go2 | ±0.5 | 43.3 → **22.6** | 64.7 → **84.1** |
| Go2 | ±1 | 40.1 → **38.6** | 68.3 → **72.0** |
| Go2 | ±1.5 | **47.5** → 58.8 | **54.4** → 43.4 |
| Go2 | ±2 | **55.8** → 69.6 | **37.6** → 27.7 |

### Success rates

Velocity-only success requires relative vx RMSE ≤0.20, yaw-rate RMSE ≤0.05 rad/s, positive directed progress, and no termination. Gait-only success requires no termination, contact agreement ≥0.80, correct gait family, at least two complete cycles, and at least two touchdown and liftoff events per foot. Combined success requires both. Positive-progress percentage is a separate diagnostic and only requires positive net displacement along the requested world direction in a completed trial.

| Robot | Command pair [m/s] | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive-progress trials [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| X1 | ±0.5 | 25.00 → **50.00** | **81.12** → 75.00 | 25.00 → **50.00** | 86.94 → **100.00** |
| X1 | ±1 | 12.50 → 12.50 | 75.00 → 75.00 | 12.50 → 12.50 | 81.25 → **93.75** |
| X1 | ±1.5 | 0.00 → 0.00 | 50.00 → 50.00 | 0.00 → 0.00 | **100.00** → 93.75 |
| X1 | ±2 | 0.00 → 0.00 | **38.88** → 30.88 | 0.00 → 0.00 | 87.19 → **94.25** |
| X1_go2bend | ±0.5 | 12.50 → **25.00** | 50.00 → **68.75** | 12.50 → **18.75** | 81.25 → **93.75** |
| X1_go2bend | ±1 | **6.25** → 0.00 | 43.75 → **50.00** | 0.00 → 0.00 | 87.50 → 87.50 |
| X1_go2bend | ±1.5 | 0.00 → 0.00 | **37.50** → 31.25 | 0.00 → 0.00 | 93.75 → **100.00** |
| X1_go2bend | ±2 | 0.00 → 0.00 | **37.31** → 31.31 | 0.00 → 0.00 | 93.75 → **96.81** |
| Go2 | ±0.5 | **37.50** → 6.25 | 93.75 → 93.75 | **31.25** → 6.25 | 93.75 → **100.00** |
| Go2 | ±1 | **12.50** → 6.25 | 68.75 → **75.00** | 6.25 → 6.25 | **100.00** → 93.75 |
| Go2 | ±1.5 | 0.00 → 0.00 | 53.50 → **68.75** | 0.00 → 0.00 | **84.56** → 81.25 |
| Go2 | ±2 | 0.00 → 0.00 | 31.12 → **64.44** | 0.00 → 0.00 | **75.31** → 68.19 |

## Forward/backward speed consistency

This comparison pairs the forward rollout of template g with the backward rollout of its **time-reversed template TR(g)**, rather than necessarily the same named template. The statistic is the mean of `abs(mean_vx(g,+v) + mean_vx(TR(g),−v))` in m/s.

Only pairs for which all four relevant rollouts complete in both methods are included, so each row compares the same cohort. Lower bias indicates more consistent speed under reversal; it does not establish full action or trajectory symmetry.

| Robot | Command pair [m/s] | Matched completed directional pairs | TR-partner speed bias [m/s] ↓ |
| --- | ---: | ---: | ---: |
| X1 | ±0.5 | 800 | **0.028** → 0.032 |
| X1 | ±1 | 800 | 0.083 → **0.050** |
| X1 | ±1.5 | 800 | 0.119 → **0.047** |
| X1 | ±2 | 708 | 0.108 → **0.089** |
| X1_go2bend | ±0.5 | 800 | **0.045** → 0.072 |
| X1_go2bend | ±1 | 800 | **0.044** → 0.100 |
| X1_go2bend | ±1.5 | 800 | **0.089** → 0.091 |
| X1_go2bend | ±2 | 800 | 0.289 → **0.221** |
| Go2 | ±0.5 | 800 | **0.032** → 0.046 |
| Go2 | ±1 | 800 | 0.087 → **0.041** |
| Go2 | ±1.5 | 800 | 0.185 → **0.049** |
| Go2 | ±2 | 509 | 0.222 → **0.087** |

## Early terminations and matched-completion checks

All trials at ±0.5, ±1, and ±1.5 complete. There are **388 early terminations**, all at ±2: 91 X1 no-TRS, 1 X1 TRS, 295 Go2 no-TRS, and 1 Go2 TRS. Both X1_go2bend methods complete every trial.

Termination labels below identify the environment condition that caused reset; `calf_height` does not by itself establish that the robot fell. Counts include failures before the measured interval, which can be missing from the original metrics CSV's `terminated` field.

| Robot | Method | Requested gait | vx [m/s] | Terminated / planned | Termination condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| X1 | TRS | Half-bound A | +2 | 1/100 | base_height: 1 |
| X1 | No TRS | Half-bound A | -2 | 91/100 | base_height: 91 |
| Go2 | No TRS | Trot | -2 | 100/100 | calf_height: 100 |
| Go2 | No TRS | Half-bound B | -2 | 100/100 | calf_height: 100 |
| Go2 | No TRS | Transverse A | +2 | 5/100 | calf_height: 5 |
| Go2 | TRS | Transverse B | +2 | 1/100 | calf_height: 1 |
| Go2 | No TRS | Transverse B | -2 | 90/100 | calf_height: 90 |

The next table restricts ±2 continuous metrics to identical trial identities that complete in both methods. It reduces differences in cohort composition, but still excludes conditions that fail under either method and therefore is not an unconditional performance estimate.

| Robot | Matched full trials / 1,600 | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| X1 | 1508 | 0.181 → **0.161** | 0.157 → **0.147** | 80.1 → **81.3** | **43.8** → 48.1 | **60.1** → 56.4 |
| X1_go2bend | 1600 | 0.194 → **0.155** | 0.134 → **0.105** | 78.5 → **79.4** | 41.9 → **33.2** | 63.0 → **72.7** |
| Go2 | 1304 | 0.567 → **0.490** | **0.132** → 0.147 | 86.6 → **88.3** | **55.8** → 68.8 | **37.7** → 28.7 |

## Separate positive and negative commands

Each row pools eight templates at one signed velocity; 800 trials are planned per method. Full per-gait metrics are in Appendix B.

### X1: direction-level results

| vx command [m/s] | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | **0.050** → 0.052 | 0.078 → **0.054** | 94.7 → **95.7** | 59.4 → **21.3** | 46.2 → **81.3** | 100.00 → 100.00 |
| -0.5 | 0.047 → **0.046** | 0.067 → **0.057** | 94.4 → **95.8** | 44.4 → **24.8** | 60.8 → **82.9** | 100.00 → 100.00 |
| +1 | **0.070** → 0.085 | 0.088 → **0.066** | 91.9 → **93.2** | 49.7 → **29.9** | 58.6 → **74.1** | 100.00 → 100.00 |
| -1 | 0.101 → **0.064** | 0.071 → **0.062** | 93.0 → **93.1** | **37.5** → 49.4 | **64.2** → 58.3 | 100.00 → 100.00 |
| +1.5 | 0.094 → **0.073** | 0.109 → **0.089** | **87.6** → 87.3 | 34.3 → **33.6** | 76.5 → **77.8** | 100.00 → 100.00 |
| -1.5 | 0.123 → **0.087** | **0.091** → 0.096 | 87.5 → **88.0** | **27.5** → 32.5 | **78.2** → 75.6 | 100.00 → 100.00 |
| +2 | 0.180 → **0.163** | **0.158** → 0.159 | **81.6** → 81.5 | **31.8** → 59.9 | **72.0** → 43.2 | **100.00** → 99.88 |
| -2 | 0.181 → **0.158** | 0.155 → **0.129** | 78.5 → **81.0** | 57.3 → **33.1** | 46.8 → **73.2** | 88.62 → **100.00** |

### X1_go2bend: direction-level results

| vx command [m/s] | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | 0.060 → **0.052** | 0.089 → **0.072** | 92.2 → **94.9** | **36.9** → 55.6 | **78.2** → 56.2 | 100.00 → 100.00 |
| -0.5 | **0.057** → 0.083 | 0.089 → **0.061** | 91.7 → **93.8** | 58.2 → **31.8** | 48.7 → **73.2** | 100.00 → 100.00 |
| +1 | **0.076** → 0.090 | 0.096 → 0.096 | 89.7 → **92.2** | **32.0** → 50.7 | **75.0** → 58.4 | 100.00 → 100.00 |
| -1 | **0.074** → 0.089 | 0.092 → **0.084** | 89.4 → **91.1** | 54.4 → **33.6** | 53.6 → **74.4** | 100.00 → 100.00 |
| +1.5 | **0.074** → 0.077 | 0.100 → **0.083** | 84.3 → **87.5** | 31.3 → **18.1** | 78.3 → **92.4** | 100.00 → 100.00 |
| -1.5 | 0.126 → **0.095** | 0.121 → **0.085** | **85.5** → 84.9 | 62.3 → **23.8** | 46.0 → **84.5** | 100.00 → 100.00 |
| +2 | 0.100 → **0.094** | 0.123 → **0.105** | 78.2 → **79.8** | 33.0 → **29.0** | 75.7 → **81.1** | 100.00 → 100.00 |
| -2 | 0.287 → **0.216** | 0.145 → **0.106** | 78.8 → **79.1** | 50.8 → **37.3** | 50.2 → **64.4** | 100.00 → 100.00 |

### Go2: direction-level results

| vx command [m/s] | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.5 | **0.043** → 0.049 | **0.060** → 0.070 | 95.6 → **96.4** | 33.0 → **26.4** | 77.5 → **81.0** | 100.00 → 100.00 |
| -0.5 | **0.045** → 0.047 | 0.067 → **0.061** | 96.0 → **96.7** | 53.6 → **18.8** | 51.9 → **87.2** | 100.00 → 100.00 |
| +1 | 0.071 → **0.066** | **0.076** → 0.085 | 92.9 → **94.5** | **33.6** → 36.6 | **78.4** → 77.1 | 100.00 → 100.00 |
| -1 | 0.095 → **0.061** | **0.070** → 0.082 | **93.9** → 93.7 | 46.5 → **40.6** | 58.2 → **66.9** | 100.00 → 100.00 |
| +1.5 | **0.144** → 0.156 | **0.099** → 0.106 | 88.8 → **91.3** | **28.2** → 42.8 | **79.4** → 63.2 | 100.00 → 100.00 |
| -1.5 | 0.294 → **0.187** | **0.106** → 0.140 | **91.3** → 90.5 | **66.8** → 74.9 | **29.4** → 23.6 | 100.00 → 100.00 |
| +2 | 0.472 → **0.459** | 0.133 → **0.124** | 86.0 → **88.9** | 59.4 → **40.6** | 38.5 → **54.7** | 99.38 → **99.88** |
| -2 | 0.714 → **0.528** | **0.129** → 0.182 | 87.6 → **88.1** | **50.2** → 98.7 | **36.4** → 0.7 | 63.75 → **100.00** |

## Metric definitions

| Metric | Definition and preferred direction |
| --- | ---: |
| vx / vy RMSE [m/s] ↓ | Square root of the temporal mean squared body-frame velocity error. Each trial is evaluated first; trial RMSE values are then averaged. vy command is zero. |
| Relative vx RMSE [%] ↓ | 100 × vx RMSE / abs(vx command). |
| Yaw-rate RMSE [rad/s] ↓ | Square root of the temporal mean squared yaw-rate error; the command is zero. This differs from accumulated heading drift. |
| Heading RMSE [degrees] ↓ | RMSE of heading error wrapped to [−π, π] relative to nominal reset heading zero; converted to degrees. |
| Directed progress [%] ≈100 | 100 × requested-sign world-x displacement / commanded distance over the complete-cycle measurement window. Negative values mean net displacement opposite the requested world direction. |
| Contact agreement [%] ↑ | Contact agreement with commanded stance, using recorded signed per-foot phases and excluding samples within 0.02 cycles of a phase boundary. |
| Contact during swing [%] ↓ | Percentage of eligible commanded-swing samples with force-detected contact. |
| Missed stance [%] ↓ | Percentage of eligible commanded-stance samples without force-detected contact. |
| Touchdown timing error [ms] ↓ | Mean absolute matched touchdown timing error relative to the commanded gait clock; not inter-foot synchronization. Extra contacts are evaluated separately. |
| Measured duty [%], target 56 | Mean measured contact fraction across feet. Closer to 56% is preferred; duty error below is the stronger diagnostic because averaging can cancel opposing errors. |
| Duty error [pp] ↓ | 100 × mean absolute difference between each foot's measured contact fraction and commanded duty 0.56. |
| Max-foot touchdowns/cycle ≈1 | For each trial, maximum touchdown count among its four feet divided by complete cycles; then averaged across trials. Small deviations below one can reflect event-window boundaries. |
| Mechanical power [W] ↓* | Temporal mean of the sum over joints of abs(torque × joint velocity). This measures absolute mechanical effort, not electrical consumption. |
| Normalized torque cost ↓* | Temporal mean of the sum over joints of (torque / configured joint effort limit)^2; dimensionless. |
| Action clipping [%] ↓ | Fraction of joint-time action commands clipped by target constraints. Lower does not prove smoothness. |
| Work imbalance [%] ↓† | 100 × abs(front absolute mechanical work − rear absolute mechanical work) / their sum. |
| Vertical-force impulse imbalance [%] ↓† | The same imbalance formula using integrated positive vertical ground forces. |
| Normalized torque imbalance [%] ↓† | The same imbalance formula using integrated squared torque normalized by joint effort limits. |
| Exact template / correct family [%] ↑ | Fraction of all planned trials that complete and whose contact-pattern classifier matches the requested exact template / broader family. Classifier maximum error is 0.12 cycles and minimum winning margin 0.01 cycles. |
| Completion [%] ↑ | Fraction of planned trials completing all 250 settling + 1,000 measured steps without episode termination. |
| Velocity / gait / combined success [%] ↑ | Fraction of all planned trials satisfying the separate predicates stated above. Early terminations count as unsuccessful. |
| Positive-progress trials [%] ↑ | Fraction of all planned trials that complete and have positive requested-direction displacement. |
| TR-partner speed bias [m/s] ↓ | Mean absolute sum of forward mean vx and backward mean vx for its time-reversed template partner, restricted to matched completed pairs. |

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

Each row has 200 planned trials per method. Continuous metrics average completed trials from both signs; percentages retain all 200 trials in their denominator. A/B labels denote the saved command templates. The naming map is `gallop_a/c = Rotary A/B`, `gallop_b/d = Transverse A/B`, and `half_bound_front_a/b = Half-bound A/B`.

### X1: gait-level paired results

**vx = ±0.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | **0.048** → 0.054 | **0.046** → 0.064 | 96.4 → **98.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.058 → **0.037** | 0.094 → **0.046** | 96.7 → **99.0** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A | **0.037** → 0.048 | 0.077 → **0.044** | 96.0 → **96.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | 0.046 → 0.046 | 0.081 → **0.046** | 95.1 → **97.1** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | **0.033** → 0.046 | 0.094 → **0.080** | 86.6 → **88.8** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | **0.043** → 0.054 | 0.078 → **0.068** | **89.9** → 88.8 | **49.0** → 0.0 | **49.0** → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.066 → **0.053** | 0.051 → **0.046** | **99.1** → 98.7 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.057 → **0.053** | 0.058 → **0.051** | 96.9 → **98.7** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.071 → **0.067** | **0.051** → 0.072 | 95.1 → **96.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.092 → **0.081** | 0.094 → **0.050** | 95.5 → **96.2** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A | **0.064** → 0.069 | 0.076 → **0.063** | **95.1** → 93.2 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | 0.122 → **0.086** | 0.096 → **0.056** | 94.0 → **96.0** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | **0.077** → 0.080 | 0.092 → **0.074** | 81.0 → **86.0** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | 0.091 → **0.074** | 0.104 → **0.069** | **86.0** → 84.4 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.079 → **0.071** | **0.053** → 0.070 | **97.7** → 96.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.086 → **0.070** | 0.069 → **0.062** | 95.1 → **97.1** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.069 → **0.063** | 0.079 → **0.070** | 92.0 → **92.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.136 → **0.098** | **0.101** → 0.106 | 88.2 → **89.4** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Half-bound A | **0.088** → 0.102 | 0.105 → **0.095** | **90.7** → 87.8 | **100.0** → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Half-bound B | 0.195 → **0.084** | 0.096 → 0.096 | 87.6 → **91.5** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Rotary A | **0.069** → 0.075 | 0.101 → **0.073** | 77.1 → **80.9** | 0.0 → 0.0 | **21.5** → 0.0 | 100.0 → 100.0 |
| Rotary B | 0.114 → **0.077** | **0.120** → 0.128 | **81.2** → 74.2 | 0.0 → 0.0 | 0.0 → **98.0** | 100.0 → 100.0 |
| Transverse A | **0.067** → 0.070 | **0.092** → 0.093 | **93.5** → 90.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.130 → **0.068** | 0.105 → **0.078** | 90.2 → **94.1** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | **0.130** → 0.170 | **0.129** → 0.188 | 88.0 → **89.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.234 → **0.220** | **0.140** → 0.163 | **81.0** → 80.6 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.201 → **0.133** | 0.144 → **0.122** | **87.2** → 78.6 | **32.0** → 0.0 | **32.0** → 0.0 | 54.5 → **99.5** |
| Half-bound B | 0.181 → **0.150** | 0.264 → **0.123** | 72.6 → **84.2** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary A | 0.219 → **0.146** | 0.164 → **0.124** | 70.6 → **73.6** | 0.0 → 0.0 | 36.5 → **100.0** | 100.0 → 100.0 |
| Rotary B | **0.132** → 0.157 | **0.153** → 0.213 | **75.0** → 69.5 | 0.0 → 0.0 | 0.0 → **98.0** | 100.0 → 100.0 |
| Transverse A | 0.219 → **0.104** | 0.115 → **0.105** | **89.2** → 85.2 | **92.5** → 50.0 | **92.5** → 50.0 | 100.0 → 100.0 |
| Transverse B | **0.139** → 0.201 | 0.139 → **0.112** | 80.8 → **88.6** | **99.0** → 97.0 | **99.0** → 97.0 | 100.0 → 100.0 |

### X1_go2bend: gait-level paired results

**vx = ±0.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | **0.027** → 0.035 | **0.042** → 0.062 | 96.9 → **99.1** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.100 → **0.081** | 0.109 → **0.060** | 91.4 → **91.7** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Half-bound A | **0.071** → 0.088 | 0.100 → **0.073** | 95.8 → **97.3** | 50.0 → **100.0** | 50.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B | 0.061 → **0.048** | 0.092 → **0.049** | 91.5 → **96.9** | 50.0 → **100.0** | 50.0 → **100.0** | 100.0 → 100.0 |
| Rotary A | 0.057 → **0.048** | **0.083** → 0.095 | 78.1 → **86.8** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B | **0.052** → 0.103 | 0.108 → **0.088** | **87.5** → 86.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | **0.052** → 0.078 | 0.094 → **0.058** | 96.9 → **98.7** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | **0.051** → 0.056 | 0.085 → **0.045** | 97.3 → **98.2** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.059 → **0.053** | **0.061** → 0.063 | 96.4 → **96.9** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.125 → **0.109** | 0.146 → **0.090** | 88.4 → **89.0** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | **0.076** → 0.093 | **0.071** → 0.107 | 93.8 → **95.5** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Half-bound B | **0.074** → 0.084 | 0.112 → **0.078** | 89.9 → **94.2** | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary A | **0.069** → 0.079 | **0.100** → 0.115 | 74.1 → **81.2** | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary B | **0.075** → 0.131 | **0.087** → 0.134 | 83.3 → **84.6** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | **0.057** → 0.091 | 0.094 → **0.057** | 95.5 → **96.9** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | **0.062** → 0.073 | 0.079 → **0.077** | **94.8** → 94.6 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.116 → **0.079** | 0.075 → **0.054** | 93.3 → **95.1** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.118 → **0.072** | 0.124 → **0.108** | **82.9** → 81.9 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.092 → **0.077** | 0.129 → **0.090** | **89.6** → 88.2 | **50.0** → 0.0 | **50.0** → 0.0 | 100.0 → 100.0 |
| Half-bound B | **0.083** → 0.101 | 0.129 → **0.084** | 83.7 → **89.1** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Rotary A | 0.090 → **0.079** | 0.113 → **0.086** | 69.8 → **75.0** | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.108 → **0.096** | 0.128 → **0.096** | **78.8** → 77.0 | 0.0 → 0.0 | **50.0** → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.095 → **0.090** | 0.084 → **0.061** | 91.0 → **92.4** | **100.0** → 50.0 | **100.0** → 50.0 | 100.0 → 100.0 |
| Transverse B | 0.098 → **0.092** | 0.102 → **0.092** | 90.2 → **91.1** | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.223 → **0.161** | 0.093 → **0.071** | 89.5 → **91.8** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.226 → **0.155** | 0.143 → **0.138** | **75.7** → 74.6 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | **0.164** → 0.169 | 0.124 → **0.104** | **82.2** → 80.7 | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound B | 0.209 → **0.171** | 0.151 → **0.096** | 75.2 → **80.3** | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary A | 0.175 → **0.130** | 0.148 → **0.090** | 63.0 → **68.1** | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.208 → **0.156** | 0.140 → **0.124** | **72.0** → 68.4 | 0.0 → 0.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Transverse A | **0.137** → 0.149 | 0.137 → **0.118** | 85.7 → **86.3** | **89.0** → 53.5 | **89.0** → 53.5 | 100.0 → 100.0 |
| Transverse B | 0.206 → **0.146** | 0.136 → **0.101** | 84.6 → **85.3** | **59.5** → 47.0 | **59.5** → 47.0 | 100.0 → 100.0 |

### Go2: gait-level paired results

**vx = ±0.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.048 → **0.031** | **0.060** → 0.070 | 97.6 → **99.5** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | **0.041** → 0.051 | **0.055** → 0.073 | 93.2 → **94.6** | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound A | **0.036** → 0.058 | **0.058** → 0.060 | 96.4 → **96.9** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B | **0.040** → 0.051 | 0.070 → **0.050** | 94.3 → **95.8** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A | **0.047** → 0.057 | 0.067 → **0.063** | 96.9 → **97.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B | 0.048 → 0.048 | 0.112 → **0.063** | 96.4 → 96.4 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A | **0.051** → 0.066 | **0.037** → 0.075 | 96.9 → 96.9 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.041 → **0.023** | **0.050** → 0.068 | **95.1** → 94.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | 0.039 → **0.037** | 0.082 → **0.066** | 96.4 → **97.4** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.119 → **0.097** | **0.065** → 0.084 | **89.5** → 88.0 | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Half-bound A | 0.096 → **0.065** | 0.074 → **0.067** | 94.7 → **96.3** | 50.0 → **100.0** | 50.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B | 0.081 → **0.076** | 0.094 → **0.075** | 91.9 → **94.3** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Rotary A | 0.091 → **0.078** | **0.049** → 0.104 | 92.2 → **94.3** | **100.0** → 50.0 | **100.0** → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.106 → **0.055** | **0.082** → 0.096 | 93.4 → **94.3** | 50.0 → 50.0 | 50.0 → 50.0 | 100.0 → 100.0 |
| Transverse A | 0.067 → **0.059** | **0.067** → 0.100 | **96.3** → 94.8 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B | 0.066 → **0.044** | **0.072** → 0.076 | 92.7 → **93.8** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±1.5 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | **0.130** → 0.142 | **0.099** → 0.107 | 93.0 → **94.5** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound | 0.281 → **0.136** | **0.068** → 0.125 | **86.4** → 82.9 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Half-bound A | 0.247 → **0.190** | 0.096 → **0.085** | **94.1** → 93.2 | 50.0 → **100.0** | 50.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B | 0.228 → **0.181** | **0.119** → 0.184 | 88.7 → **92.2** | 0.0 → **100.0** | 0.0 → **100.0** | 100.0 → 100.0 |
| Rotary A | 0.230 → **0.162** | **0.100** → 0.158 | 87.8 → **88.8** | **100.0** → 50.0 | **100.0** → 50.0 | 100.0 → 100.0 |
| Rotary B | 0.220 → **0.176** | 0.127 → **0.104** | 85.6 → **92.2** | 0.0 → **50.0** | 0.0 → **50.0** | 100.0 → 100.0 |
| Transverse A | 0.212 → **0.180** | **0.098** → 0.124 | **93.7** → 91.7 | **78.0** → 50.0 | **78.0** → 50.0 | 100.0 → 100.0 |
| Transverse B | 0.204 → **0.202** | 0.112 → **0.099** | 91.3 → **91.6** | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

**vx = ±2 m/s**

| Gait | vx RMSE [m/s] ↓ | Yaw-rate RMSE [rad/s] ↓ | Agreement [%] ↑ | Exact template [%] ↑ | Correct family [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot | **0.439** → 0.491 | 0.133 → **0.111** | **93.6** → 93.1 | 50.0 → **100.0** | 50.0 → **100.0** | 50.0 → **100.0** |
| Bound | 0.609 → **0.437** | **0.107** → 0.171 | **83.4** → 79.6 | 0.0 → **48.5** | 0.0 → **48.5** | 100.0 → 100.0 |
| Half-bound A | 0.585 → **0.532** | **0.130** → 0.159 | 89.3 → **90.5** | 0.0 → **100.0** | 0.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B | 0.551 → **0.441** | **0.140** → 0.217 | 83.1 → **88.4** | 0.0 → **100.0** | 0.0 → **100.0** | 50.0 → **100.0** |
| Rotary A | 0.621 → **0.439** | **0.126** → 0.167 | **86.4** → 85.4 | **100.0** → 17.5 | **100.0** → 17.5 | 100.0 → 100.0 |
| Rotary B | 0.566 → **0.520** | **0.123** → 0.149 | 80.4 → **89.1** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Transverse A | 0.556 → **0.501** | 0.145 → **0.127** | **91.9** → 89.5 | 44.0 → **50.0** | 44.0 → **50.0** | 97.5 → **100.0** |
| Transverse B | **0.508** → 0.588 | 0.173 → **0.122** | 87.1 → **92.5** | 55.0 → **99.5** | 55.0 → **99.5** | 55.0 → **99.5** |

## Appendix B: all metrics for every gait and direction

Every condition below contains **100 planned trials per method**. All entries remain No TRS → TRS. Tables separate metric groups to keep each comparison readable. `N/A` is retained for continuous metrics when no full trial completes; success/completion still use all 100 planned trials.

### X1: full gait/direction details

#### X1, vx = ±0.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | **0.051** → 0.068 | **0.025** → 0.033 | **10.18** → 13.60 | **0.035** → 0.069 | **19.1** → 42.2 | **87.3** → 62.6 |
| Trot / -0.5 | 0.046 → **0.040** | 0.021 → 0.021 | 9.22 → **8.02** | 0.058 → 0.058 | 44.4 → **9.8** | 66.8 → **93.0** |
| Bound / +0.5 | 0.045 → **0.033** | 0.039 → **0.009** | 8.90 → **6.55** | 0.139 → **0.043** | 126.1 → **5.1** | -33.1 → **91.2** |
| Bound / -0.5 | 0.072 → **0.041** | 0.036 → **0.012** | 14.36 → **8.14** | **0.049** → 0.050 | 66.7 → **15.2** | 41.2 → **85.3** |
| Half-bound A / +0.5 | **0.047** → 0.054 | 0.043 → **0.025** | **9.47** → 10.83 | 0.074 → **0.041** | 46.7 → **43.5** | 66.9 → **68.4** |
| Half-bound A / -0.5 | **0.027** → 0.041 | 0.035 → **0.032** | **5.41** → 8.22 | 0.079 → **0.048** | 50.3 → **26.9** | 64.8 → **89.6** |
| Half-bound B / +0.5 | **0.028** → 0.054 | **0.025** → 0.031 | **5.63** → 10.79 | 0.070 → **0.048** | 68.4 → **7.5** | 41.7 → **86.1** |
| Half-bound B / -0.5 | 0.063 → **0.038** | 0.046 → 0.046 | 12.60 → **7.52** | 0.092 → **0.044** | 73.2 → **18.4** | 33.9 → **86.6** |
| Rotary A / +0.5 | **0.024** → 0.059 | **0.044** → 0.051 | **4.76** → 11.86 | 0.101 → **0.089** | 104.6 → **21.1** | -9.1 → **80.3** |
| Rotary A / -0.5 | 0.042 → **0.033** | **0.051** → 0.059 | 8.50 → **6.65** | 0.086 → **0.072** | 97.1 → **6.1** | 1.2 → **91.3** |
| Rotary B / +0.5 | **0.044** → 0.048 | 0.081 → **0.067** | **8.71** → 9.68 | 0.091 → **0.055** | 12.8 → **3.4** | **96.3** → 92.2 |
| Rotary B / -0.5 | **0.043** → 0.059 | **0.056** → 0.060 | **8.54** → 11.76 | **0.065** → 0.081 | **4.7** → 51.8 | **92.4** → 66.2 |
| Transverse A / +0.5 | 0.080 → **0.064** | **0.015** → 0.016 | 15.99 → **12.82** | 0.045 → **0.029** | 34.1 → **10.2** | 71.1 → **90.3** |
| Transverse A / -0.5 | 0.053 → **0.042** | **0.029** → 0.034 | 10.60 → **8.44** | **0.057** → 0.063 | **17.3** → 49.6 | **88.2** → 71.1 |
| Transverse B / +0.5 | 0.082 → **0.034** | 0.067 → **0.037** | 16.41 → **6.80** | 0.071 → **0.063** | 63.4 → **37.3** | 48.5 → **79.6** |
| Transverse B / -0.5 | **0.032** → 0.071 | 0.049 → **0.021** | **6.37** → 14.24 | 0.046 → **0.040** | **1.4** → 20.9 | **98.1** → 79.6 |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 96.4 → **100.0** | **4.0** → 16.0 | 53.33 → 53.33 | 2.67 → 2.67 | 0.00 → 0.00 | 6.25 → **0.00** | 1.00 → 1.00 |
| Trot / -0.5 | 96.4 → **96.9** | **5.1** → 7.8 | 52.50 → 52.50 | 3.50 → 3.50 | 0.00 → 0.00 | 6.25 → **5.47** | 1.00 → 1.00 |
| Bound / +0.5 | 96.9 → **97.9** | 16.0 → **14.0** | 50.42 → **51.95** | 5.58 → **4.05** | 0.00 → 0.00 | 5.47 → **3.61** | 1.00 → 1.00 |
| Bound / -0.5 | 96.4 → **100.0** | **10.4** → 20.0 | 51.67 → **53.33** | 4.33 → **2.67** | 0.00 → 0.00 | 6.25 → **0.00** | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.5 → **96.4** | 15.5 → **7.9** | 52.92 → **53.79** | **3.42** → 3.79 | 2.08 → **0.00** | 6.25 → 6.25 | 1.00 → 1.00 |
| Half-bound A / -0.5 | 96.4 → 96.4 | **14.0** → 19.0 | 54.17 → 54.17 | 2.50 → **2.17** | 2.08 → 2.08 | 4.69 → 4.69 | 1.00 → 1.00 |
| Half-bound B / +0.5 | 95.5 → **96.9** | **14.0** → 20.8 | 53.33 → **54.58** | 3.00 → **2.08** | 2.08 → 2.08 | 6.25 → **3.91** | 1.00 → 1.00 |
| Half-bound B / -0.5 | 94.6 → **97.3** | 24.0 → **19.0** | **55.83** → 54.17 | 2.50 → 2.50 | 6.25 → **2.08** | 4.69 → **3.12** | 1.00 → 1.00 |
| Rotary A / +0.5 | 87.1 → **88.4** | 42.9 → **38.0** | **55.41** → 57.50 | **1.25** → 1.50 | **13.54** → 14.58 | 12.50 → **9.38** | 1.00 → 1.00 |
| Rotary A / -0.5 | 86.1 → **89.3** | 35.5 → **33.0** | **56.18** → 56.67 | **2.64** → 3.67 | 14.58 → **12.50** | 13.42 → **9.37** | 1.00 → 1.00 |
| Rotary B / +0.5 | **90.1** → 88.4 | **27.3** → 37.8 | 54.23 → **57.50** | **2.57** → 2.83 | **8.49** → 14.58 | 10.94 → **9.38** | 1.00 → 1.00 |
| Rotary B / -0.5 | **89.6** → 89.3 | **31.5** → 33.0 | 55.31 → **56.67** | 2.31 → **2.00** | **10.42** → 12.50 | 10.35 → **9.38** | 1.00 → 1.00 |
| Transverse A / +0.5 | 99.1 → 99.1 | 8.0 → **6.0** | 54.17 → **55.00** | 2.17 → **1.67** | 0.00 → 0.00 | 1.56 → 1.56 | 1.00 → 1.00 |
| Transverse A / -0.5 | **99.1** → 98.2 | 11.0 → **10.0** | **54.17** → 53.33 | **2.50** → 3.00 | 0.00 → 0.00 | **1.56** → 3.12 | 1.00 → 1.00 |
| Transverse B / +0.5 | 97.3 → **98.2** | 8.0 → **6.0** | 53.33 → **54.17** | 2.67 → **2.17** | 0.00 → 0.00 | 4.69 → **3.12** | 1.00 → 1.00 |
| Transverse B / -0.5 | 96.4 → **99.1** | **2.0** → 10.0 | 53.33 → **54.17** | 3.00 → **2.17** | 0.00 → 0.00 | 6.25 → **1.56** | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 38.6 → **30.7** | 0.582 → **0.575** | 0.00 → 0.00 | 7.15 → **6.79** | 0.88 → **0.09** | 10.71 → **3.93** |
| Trot / -0.5 | 37.9 → **35.5** | 0.625 → **0.614** | 0.00 → 0.00 | **3.23** → 14.93 | **0.18** → 6.25 | **0.23** → 7.38 |
| Bound / +0.5 | **97.8** → 103.4 | 0.594 → **0.534** | 0.00 → 0.00 | 7.01 → **1.35** | **0.08** → 0.51 | **1.07** → 8.54 |
| Bound / -0.5 | **99.7** → 100.8 | 0.572 → **0.546** | 0.00 → 0.00 | 7.27 → **0.30** | **0.74** → 4.11 | 21.50 → **13.25** |
| Half-bound A / +0.5 | **78.0** → 86.5 | **0.613** → 0.616 | 0.00 → 0.00 | **4.98** → 5.41 | 5.47 → **2.14** | 14.58 → **1.12** |
| Half-bound A / -0.5 | 93.1 → **92.5** | **0.555** → 0.587 | 0.00 → 0.00 | 4.10 → **0.23** | 4.71 → **2.85** | 14.33 → **1.62** |
| Half-bound B / +0.5 | 97.2 → **87.2** | **0.582** → 0.608 | 0.00 → 0.00 | **4.77** → 4.97 | **8.15** → 8.33 | **0.80** → 2.37 |
| Half-bound B / -0.5 | **87.3** → 96.6 | 0.636 → **0.559** | 0.00 → 0.00 | 6.18 → **0.39** | 4.45 → **3.16** | 17.08 → **14.26** |
| Rotary A / +0.5 | 99.8 → **91.0** | **0.545** → 0.621 | 0.00 → 0.00 | **0.94** → 2.95 | **0.63** → 3.70 | 19.02 → **12.70** |
| Rotary A / -0.5 | **90.0** → 92.4 | 0.586 → **0.516** | 0.00 → 0.00 | 10.17 → **1.42** | 5.13 → **5.07** | 25.84 → **4.49** |
| Rotary B / +0.5 | **83.6** → 87.5 | 0.596 → **0.533** | 0.56 → **0.00** | **0.93** → 2.65 | **5.07** → 6.87 | 22.14 → **16.80** |
| Rotary B / -0.5 | 92.7 → **89.3** | 0.557 → **0.535** | 0.00 → 0.00 | 4.97 → **2.06** | **4.26** → 5.64 | 6.15 → **4.73** |
| Transverse A / +0.5 | **66.1** → 75.1 | 0.629 → **0.560** | 0.00 → 0.00 | 8.32 → **6.26** | **0.35** → 0.48 | 7.13 → **3.93** |
| Transverse A / -0.5 | **73.9** → 78.7 | 0.604 → **0.599** | 0.00 → 0.00 | 5.36 → **5.23** | **1.66** → 3.01 | 8.98 → **2.12** |
| Transverse B / +0.5 | 82.1 → **75.3** | **0.592** → 0.618 | 0.56 → **0.00** | **2.73** → 3.82 | **4.19** → 5.15 | 8.89 → **8.80** |
| Transverse B / -0.5 | **71.4** → 79.5 | 0.708 → **0.656** | 0.00 → 0.00 | 8.72 → **6.48** | 4.17 → **0.63** | **10.79** → 14.37 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 0.0 → **100.0** | 100.0 → 100.0 |
| Bound / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 8.0 → **100.0** | 100.0 → 100.0 |
| Rotary A / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 83.0 → **100.0** | 100.0 → 100.0 |
| Rotary B / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | **98.0** → 0.0 | **98.0** → 0.0 | 0.0 → 0.0 | **98.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1, vx = ±1 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | **0.056** → 0.074 | **0.020** → 0.049 | **5.61** → 7.44 | **0.060** → 0.102 | **26.5** → 77.9 | **90.5** → 28.6 |
| Trot / -1 | 0.086 → **0.060** | 0.027 → **0.021** | 8.63 → **5.98** | **0.041** → 0.043 | 10.4 → **2.2** | 90.4 → **96.4** |
| Bound / +1 | **0.069** → 0.101 | 0.021 → **0.019** | **6.91** → 10.06 | 0.079 → **0.044** | 44.9 → **42.2** | **69.9** → 63.9 |
| Bound / -1 | 0.116 → **0.060** | 0.049 → **0.027** | 11.56 → **6.05** | 0.109 → **0.055** | 124.2 → **47.9** | -27.2 → **63.3** |
| Half-bound A / +1 | **0.054** → 0.081 | 0.032 → **0.030** | **5.38** → 8.11 | **0.059** → 0.073 | **10.4** → 25.9 | **95.8** → 82.1 |
| Half-bound A / -1 | 0.074 → **0.057** | 0.057 → **0.028** | 7.44 → **5.72** | 0.093 → **0.054** | 72.2 → **23.7** | 34.9 → **89.5** |
| Half-bound B / +1 | **0.056** → 0.101 | 0.044 → **0.033** | **5.59** → 10.14 | 0.132 → **0.054** | 101.0 → **4.4** | -2.0 → **87.4** |
| Half-bound B / -1 | 0.189 → **0.071** | **0.042** → 0.057 | 18.85 → **7.06** | 0.060 → **0.058** | **3.8** → 64.6 | **77.6** → 41.1 |
| Rotary A / +1 | **0.076** → 0.102 | **0.041** → 0.061 | **7.60** → 10.22 | 0.105 → **0.074** | 125.8 → **10.1** | -32.7 → **86.6** |
| Rotary A / -1 | 0.078 → **0.058** | **0.034** → 0.066 | 7.80 → **5.77** | 0.079 → **0.073** | **33.3** → 113.2 | **78.5** → -19.6 |
| Rotary B / +1 | 0.087 → **0.075** | 0.072 → **0.060** | 8.73 → **7.49** | 0.122 → **0.062** | 23.4 → **6.6** | **97.1** → 91.7 |
| Rotary B / -1 | 0.095 → **0.073** | 0.079 → **0.043** | 9.54 → **7.35** | 0.086 → **0.075** | **14.3** → 38.6 | **86.9** → 74.5 |
| Transverse A / +1 | 0.083 → **0.075** | **0.030** → 0.033 | 8.29 → **7.53** | **0.051** → 0.058 | **12.8** → 32.6 | **89.5** → 78.8 |
| Transverse A / -1 | 0.075 → **0.067** | **0.034** → 0.045 | 7.48 → **6.71** | **0.054** → 0.082 | **19.6** → 62.9 | **88.4** → 51.7 |
| Transverse B / +1 | 0.075 → **0.073** | **0.032** → 0.043 | 7.49 → **7.25** | 0.092 → **0.064** | 52.8 → **40.0** | 60.9 → **73.7** |
| Transverse B / -1 | 0.097 → **0.068** | 0.039 → **0.025** | 9.74 → **6.82** | **0.047** → 0.059 | **22.0** → 42.3 | **83.9** → 69.3 |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 94.6 → **97.3** | **7.0** → 20.3 | 50.83 → 50.83 | 5.17 → 5.17 | 0.00 → 0.00 | 9.38 → **4.69** | 1.00 → 1.00 |
| Trot / -1 | 95.5 → 95.5 | 5.5 → **5.1** | 51.67 → 51.67 | 4.33 → 4.33 | 0.00 → 0.00 | 7.81 → 7.81 | 1.00 → 1.00 |
| Bound / +1 | 93.8 → **95.8** | **15.0** → 18.0 | 49.17 → **50.29** | 6.83 → **5.71** | 0.00 → 0.00 | 10.94 → **7.28** | 1.00 → 1.00 |
| Bound / -1 | **97.3** → 96.6 | **10.4** → 23.3 | **52.50** → 50.16 | **3.50** → 5.84 | 0.00 → 0.00 | **4.69** → 5.96 | 1.00 → 1.00 |
| Half-bound A / +1 | **94.6** → 92.9 | **9.0** → 19.0 | 50.78 → **52.60** | 5.55 → **3.73** | **0.00** → 4.17 | 9.47 → **9.38** | 1.00 → 1.00 |
| Half-bound A / -1 | **95.5** → 93.5 | **19.0** → 21.3 | **53.33** → 51.09 | **3.33** → 4.91 | **2.08** → 2.73 | **6.25** → 9.38 | 1.00 → 1.00 |
| Half-bound B / +1 | 93.8 → **95.5** | **12.9** → 18.2 | 50.00 → **51.67** | 6.33 → **4.33** | 0.00 → 0.00 | 10.94 → **7.81** | 1.00 → 1.00 |
| Half-bound B / -1 | 94.2 → **96.4** | 29.0 → **24.0** | **54.56** → 52.53 | 4.22 → **3.81** | 6.25 → **2.08** | 5.52 → **4.69** | 1.00 → 1.00 |
| Rotary A / +1 | 80.8 → **85.4** | 55.7 → **44.7** | 49.69 → **51.38** | 6.31 → **4.62** | 14.58 → **10.42** | 22.66 → **17.72** | 1.00 → 1.00 |
| Rotary A / -1 | 81.2 → **86.6** | 40.5 → **38.0** | 52.50 → **53.33** | 4.17 → **3.33** | 16.67 → **12.50** | 20.31 → **14.06** | 1.00 → 1.00 |
| Rotary B / +1 | **86.7** → 84.9 | **35.9** → 47.8 | 50.91 → **52.42** | 5.79 → **3.91** | **8.14** → 12.30 | 17.24 → **17.19** | 1.00 → 1.00 |
| Rotary B / -1 | **85.4** → 83.9 | 48.0 → **41.3** | 53.02 → **53.33** | 4.69 → **3.33** | **12.50** → 14.58 | **16.21** → 17.19 | 1.00 → 1.00 |
| Transverse A / +1 | **97.3** → 96.4 | 12.0 → **11.5** | **51.67** → 51.35 | 4.67 → **4.65** | 0.00 → 0.00 | **4.69** → 6.25 | 1.00 → 1.00 |
| Transverse A / -1 | **98.0** → 95.5 | **11.0** → 15.0 | **53.12** → 50.00 | **3.21** → 6.00 | 0.00 → 0.00 | **3.52** → 7.81 | 1.00 → 1.00 |
| Transverse B / +1 | 93.8 → **97.3** | 17.0 → **14.0** | 49.17 → **51.67** | 6.83 → **4.33** | 0.00 → 0.00 | 10.94 → **4.69** | 1.00 → 1.00 |
| Transverse B / -1 | 96.4 → **96.9** | 11.0 → **10.0** | 51.67 → **52.14** | 4.67 → **3.86** | 0.00 → 0.00 | 6.25 → **5.37** | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 89.2 → **74.0** | **0.722** → 0.742 | 3.06 → **0.00** | **5.07** → 9.62 | 4.30 → **0.81** | 12.13 → **2.58** |
| Trot / -1 | 79.3 → **72.7** | **0.750** → 0.768 | 0.00 → 0.00 | **10.52** → 11.10 | **2.49** → 4.11 | 2.74 → **1.31** |
| Bound / +1 | **143.8** → 145.6 | 0.849 → **0.705** | 0.00 → 0.00 | 6.22 → **0.24** | 4.57 → **4.38** | 19.46 → **16.64** |
| Bound / -1 | 152.2 → **144.2** | 0.802 → **0.713** | 0.00 → 0.00 | **4.26** → 4.31 | **0.13** → 3.94 | 17.58 → **1.36** |
| Half-bound A / +1 | **127.8** → 130.8 | 0.830 → **0.701** | 0.83 → **0.00** | **5.59** → 5.93 | 5.87 → **2.72** | 18.04 → **0.80** |
| Half-bound A / -1 | 148.8 → **146.0** | **0.773** → 0.825 | **0.00** → 0.19 | **0.90** → 2.17 | 2.40 → **1.77** | 7.03 → **0.33** |
| Half-bound B / +1 | 158.0 → **125.0** | 0.842 → **0.770** | 0.83 → **0.00** | **2.79** → 3.09 | 11.73 → **7.17** | **5.64** → 14.94 |
| Half-bound B / -1 | **134.5** → 141.4 | 0.795 → **0.715** | 0.00 → 0.00 | 6.99 → **0.37** | 5.02 → **2.33** | 14.28 → **1.12** |
| Rotary A / +1 | 157.5 → **125.5** | 0.880 → **0.807** | 0.00 → 0.00 | 4.77 → **3.42** | 8.64 → **7.13** | 24.72 → **21.72** |
| Rotary A / -1 | 156.2 → **139.8** | 0.791 → **0.676** | 0.28 → **0.00** | 10.79 → **0.63** | 6.89 → **1.67** | 19.79 → **11.90** |
| Rotary B / +1 | 183.1 → **133.2** | 0.976 → **0.745** | 0.83 → **0.00** | 26.00 → **1.01** | 14.91 → **3.27** | 30.69 → **27.93** |
| Rotary B / -1 | 144.8 → **127.5** | 0.758 → **0.711** | 0.00 → 0.00 | 4.77 → **0.12** | 4.25 → **0.96** | **5.21** → 6.33 |
| Transverse A / +1 | **108.0** → 117.6 | 0.780 → **0.756** | 1.11 → **0.00** | 11.36 → **3.82** | 3.59 → **1.57** | 14.07 → **4.59** |
| Transverse A / -1 | **120.7** → 121.9 | 0.809 → **0.788** | **0.28** → 0.83 | **7.93** → 9.77 | 4.23 → **0.36** | 14.34 → **1.79** |
| Transverse B / +1 | 130.3 → **115.4** | 0.923 → **0.823** | 2.78 → **0.00** | 4.27 → **1.03** | 4.66 → **3.12** | 13.71 → **9.28** |
| Transverse B / -1 | **113.8** → 126.5 | **0.819** → 0.880 | 0.00 → 0.00 | 7.48 → **4.20** | 3.25 → **1.30** | 8.45 → **7.34** |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Half-bound A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Rotary A / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1, vx = ±1.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 0.083 → **0.060** | 0.056 → **0.040** | 5.55 → **3.98** | 0.082 → **0.081** | **3.9** → 47.1 | **101.3** → 69.3 |
| Trot / -1.5 | **0.054** → 0.066 | **0.036** → 0.042 | **3.61** → 4.37 | 0.077 → **0.060** | 31.9 → **7.4** | 84.2 → **98.7** |
| Bound / +1.5 | 0.088 → **0.067** | 0.060 → **0.059** | 5.89 → **4.44** | 0.121 → **0.107** | 34.5 → **6.1** | 80.2 → **98.0** |
| Bound / -1.5 | 0.183 → **0.130** | **0.046** → 0.097 | 12.19 → **8.67** | **0.081** → 0.105 | 79.1 → **21.2** | 23.3 → **82.1** |
| Half-bound A / +1.5 | **0.084** → 0.118 | 0.056 → **0.038** | **5.57** → 7.88 | 0.127 → **0.114** | **42.6** → 42.8 | **73.3** → 66.9 |
| Half-bound A / -1.5 | 0.093 → **0.085** | 0.061 → **0.052** | 6.19 → **5.70** | 0.084 → **0.077** | 17.6 → **5.4** | 90.1 → **95.5** |
| Half-bound B / +1.5 | 0.112 → **0.080** | **0.039** → 0.056 | 7.47 → **5.33** | 0.100 → **0.060** | **19.9** → 23.0 | 87.5 → **87.8** |
| Half-bound B / -1.5 | 0.278 → **0.089** | **0.049** → 0.057 | 18.51 → **5.92** | **0.092** → 0.132 | **19.0** → 38.4 | 75.8 → **76.9** |
| Rotary A / +1.5 | 0.062 → **0.058** | 0.083 → **0.074** | 4.15 → **3.90** | 0.096 → **0.082** | 74.5 → **43.6** | 26.7 → **72.8** |
| Rotary A / -1.5 | **0.077** → 0.092 | **0.054** → 0.065 | **5.11** → 6.11 | 0.106 → **0.063** | **16.2** → 103.0 | **91.8** → -6.9 |
| Rotary B / +1.5 | 0.116 → **0.071** | **0.060** → 0.071 | 7.73 → **4.73** | 0.121 → **0.111** | **11.9** → 43.3 | **101.5** → 74.3 |
| Rotary B / -1.5 | 0.112 → **0.083** | **0.067** → 0.085 | 7.46 → **5.52** | **0.118** → 0.145 | **30.4** → 40.4 | **80.4** → 78.8 |
| Transverse A / +1.5 | 0.061 → **0.053** | 0.085 → **0.043** | 4.09 → **3.52** | **0.086** → 0.089 | **36.3** → 54.8 | **76.2** → 58.6 |
| Transverse A / -1.5 | **0.073** → 0.087 | **0.057** → 0.070 | **4.87** → 5.80 | 0.098 → **0.096** | **1.7** → 6.6 | 95.9 → **101.4** |
| Transverse B / +1.5 | 0.142 → **0.073** | 0.060 → **0.034** | 9.46 → **4.89** | 0.140 → **0.070** | 51.0 → **8.1** | 65.4 → **94.7** |
| Transverse B / -1.5 | 0.117 → **0.063** | 0.039 → **0.025** | 7.82 → **4.18** | **0.070** → 0.086 | **24.1** → 37.6 | **84.0** → 78.7 |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 92.9 → **93.8** | **16.0** → 22.7 | 46.67 → **47.81** | 9.33 → **8.19** | 0.00 → 0.00 | 12.50 → **10.89** | 1.00 → 1.00 |
| Trot / -1.5 | 91.1 → 91.1 | 10.4 → 10.4 | 46.67 → 46.67 | 9.33 → 9.33 | 0.00 → 0.00 | 15.62 → 15.62 | 1.00 → 1.00 |
| Bound / +1.5 | 86.9 → **87.7** | **35.1** → 41.6 | **47.85** → 46.76 | **8.15** → 9.24 | 6.33 → **3.95** | **18.16** → 18.54 | **1.04** → 1.06 |
| Bound / -1.5 | 89.6 → **91.1** | 35.4 → **30.9** | **48.30** → 45.89 | **7.70** → 10.11 | 3.71 → **0.00** | **15.49** → 15.53 | 1.00 → 1.00 |
| Half-bound A / +1.5 | **92.0** → 87.7 | **12.0** → 30.9 | **50.00** → 47.84 | **6.33** → 8.16 | **2.08** → 4.38 | **12.50** → 18.28 | 1.00 → 1.00 |
| Half-bound A / -1.5 | **89.4** → 88.0 | **29.1** → 36.5 | **48.40** → 47.08 | **7.60** → 8.92 | **4.16** → 4.17 | **15.50** → 17.94 | 2.00 → **1.03** |
| Half-bound B / +1.5 | 86.5 → **92.9** | 43.5 → **21.0** | 47.96 → **48.33** | 8.04 → **8.00** | 6.85 → **0.00** | 18.47 → **12.50** | 1.32 → **1.00** |
| Half-bound B / -1.5 | 88.7 → **90.2** | 30.0 → 30.0 | **50.30** → 49.17 | **5.70** → 6.83 | 6.32 → **4.17** | 15.12 → **14.06** | 1.36 → **1.00** |
| Rotary A / +1.5 | 76.7 → **81.1** | **23.6** → 52.1 | 48.13 → **48.20** | **7.87** → 8.13 | 16.80 → **12.50** | 28.13 → **23.68** | 2.00 → **1.00** |
| Rotary A / -1.5 | 77.5 → **80.8** | **46.9** → 55.6 | 48.05 → **50.06** | 7.95 → **5.94** | 16.56 → **15.21** | 27.01 → **22.25** | 1.14 → **1.00** |
| Rotary B / +1.5 | **82.8** → 71.4 | 44.7 → **18.0** | **48.28** → 47.50 | **7.72** → 8.50 | **9.73** → 22.92 | **22.73** → 32.81 | 1.05 → **1.00** |
| Rotary B / -1.5 | **79.5** → 76.9 | 59.8 → **34.3** | **48.94** → 46.48 | **7.06** → 9.77 | 15.46 → **14.28** | **24.26** → 29.73 | 1.21 → **1.00** |
| Transverse A / +1.5 | **93.5** → 90.5 | **19.0** → 22.4 | **48.07** → 45.26 | **7.93** → 10.74 | 0.00 → 0.00 | **11.43** → 16.70 | 1.00 → 1.00 |
| Transverse A / -1.5 | **93.5** → 91.1 | 24.3 → **22.0** | **47.44** → 46.67 | **8.56** → 9.33 | 0.00 → 0.00 | **11.42** → 15.62 | 1.42 → **1.00** |
| Transverse B / +1.5 | 89.6 → **93.5** | 31.4 → **29.3** | 46.98 → **49.75** | 9.02 → **6.34** | **2.05** → 2.08 | 16.65 → **9.84** | 1.05 → **1.00** |
| Transverse B / -1.5 | 90.8 → **94.6** | **14.6** → 16.0 | 46.39 → **49.17** | 9.61 → **6.83** | 0.00 → 0.00 | 16.14 → **9.38** | **1.19** → 1.50 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 172.2 → **155.1** | 1.080 → **1.069** | 3.33 → **0.00** | **10.01** → 14.94 | 0.88 → **0.39** | 11.56 → **9.07** |
| Trot / -1.5 | 195.0 → **163.8** | 1.183 → **1.094** | **0.00** → 0.83 | **9.85** → 11.52 | 4.59 → **3.06** | 10.76 → **2.94** |
| Bound / +1.5 | 263.7 → **229.4** | 1.275 → **1.104** | 1.62 → **0.29** | 25.70 → **17.62** | **4.02** → 5.63 | **1.08** → 2.81 |
| Bound / -1.5 | **238.8** → 250.1 | 1.124 → **0.998** | 1.06 → **0.00** | **0.72** → 17.28 | **9.92** → 15.08 | 5.35 → **2.20** |
| Half-bound A / +1.5 | 245.4 → **233.3** | 1.246 → **1.204** | 2.50 → **0.83** | **7.11** → 12.88 | 9.68 → **8.38** | **11.29** → 14.58 |
| Half-bound A / -1.5 | 260.5 → **253.4** | **1.133** → 1.297 | **1.38** → 1.84 | **5.51** → 18.31 | **4.17** → 5.50 | **2.24** → 5.69 |
| Half-bound B / +1.5 | 235.0 → **198.8** | 1.389 → **1.055** | 4.94 → **1.39** | **0.27** → 8.05 | **11.54** → 14.16 | **11.85** → 18.25 |
| Half-bound B / -1.5 | **197.9** → 223.3 | 1.119 → **1.043** | 0.28 → **0.00** | **9.60** → 10.05 | **3.12** → 14.88 | 10.14 → **6.97** |
| Rotary A / +1.5 | 241.0 → **218.1** | 1.356 → **1.097** | 3.59 → **1.39** | 7.02 → **3.27** | **1.09** → 8.57 | **15.30** → 23.87 |
| Rotary A / -1.5 | 305.6 → **244.5** | 1.377 → **1.027** | 2.78 → **0.07** | **5.51** → 9.91 | 16.24 → **5.24** | **7.79** → 17.23 |
| Rotary B / +1.5 | 287.1 → **224.2** | 1.313 → **1.175** | **2.42** → 2.68 | 14.81 → **3.35** | 13.09 → **0.93** | **12.94** → 20.13 |
| Rotary B / -1.5 | 249.1 → **240.0** | 1.227 → **1.147** | 1.27 → **0.77** | 4.05 → **1.54** | **10.29** → 22.86 | **0.18** → 5.79 |
| Transverse A / +1.5 | **180.3** → 191.8 | 1.193 → 1.193 | 2.05 → **0.00** | 10.00 → **1.76** | 11.17 → **1.89** | 15.95 → **13.99** |
| Transverse A / -1.5 | 221.4 → **215.2** | **1.161** → 1.256 | **0.59** → 0.83 | **9.87** → 21.41 | 10.62 → **1.80** | 3.36 → **1.14** |
| Transverse B / +1.5 | 223.5 → **191.6** | 1.326 → **1.220** | 5.01 → **0.00** | **0.79** → 9.48 | 9.34 → **7.15** | **13.34** → 16.31 |
| Transverse B / -1.5 | **185.0** → 239.3 | **1.204** → 1.238 | **0.56** → 0.83 | **5.68** → 10.74 | **0.62** → 6.58 | **2.26** → 2.37 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 0.0 → 0.0 | **43.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | 0.0 → **96.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1, vx = ±2 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | **0.113** → 0.231 | **0.095** → 0.165 | **5.64** → 11.56 | **0.128** → 0.296 | **9.3** → 119.8 | **95.1** → -27.1 |
| Trot / -2 | 0.147 → **0.109** | 0.080 → **0.043** | 7.37 → **5.44** | 0.131 → **0.079** | 88.9 → **13.8** | 11.8 → **92.3** |
| Bound / +2 | 0.198 → **0.124** | **0.093** → 0.106 | 9.89 → **6.20** | 0.191 → **0.149** | 89.6 → **69.9** | 8.1 → **34.2** |
| Bound / -2 | **0.271** → 0.315 | **0.059** → 0.086 | **13.56** → 15.76 | **0.089** → 0.177 | **58.7** → 81.0 | **45.6** → 20.7 |
| Half-bound A / +2 | 0.205 → **0.113** | **0.071** → 0.080 | 10.26 → **5.67** | **0.145** → 0.159 | **38.9** → 73.6 | **69.6** → 28.4 |
| Half-bound A / -2 | 0.153 → **0.152** | 0.140 → **0.084** | 7.65 → **7.62** | 0.142 → **0.085** | **7.1** → 19.0 | **92.9** → 87.7 |
| Half-bound B / +2 | **0.114** → 0.128 | **0.068** → 0.129 | **5.72** → 6.39 | 0.191 → **0.106** | **52.7** → 66.1 | **60.7** → 36.9 |
| Half-bound B / -2 | 0.247 → **0.172** | 0.111 → **0.076** | 12.37 → **8.60** | 0.337 → **0.139** | 109.1 → **34.8** | -7.6 → **76.6** |
| Rotary A / +2 | 0.261 → **0.135** | 0.109 → **0.085** | 13.07 → **6.75** | 0.199 → **0.130** | 39.8 → **4.3** | 66.8 → **93.2** |
| Rotary A / -2 | 0.176 → **0.156** | 0.089 → **0.077** | 8.81 → **7.82** | 0.129 → **0.119** | **4.3** → 13.3 | **91.1** → 89.6 |
| Rotary B / +2 | **0.159** → 0.186 | 0.093 → **0.086** | **7.93** → 9.30 | **0.173** → 0.201 | **9.2** → 62.4 | **92.7** → 45.0 |
| Rotary B / -2 | **0.105** → 0.128 | 0.131 → **0.112** | **5.23** → 6.40 | **0.133** → 0.225 | **51.4** → 67.3 | **64.4** → 37.9 |
| Transverse A / +2 | 0.288 → **0.117** | 0.074 → **0.065** | 14.42 → **5.84** | 0.120 → **0.109** | **3.5** → 45.5 | **84.8** → 65.9 |
| Transverse A / -2 | 0.149 → **0.091** | **0.048** → 0.066 | 7.43 → **4.55** | 0.111 → **0.101** | 23.6 → **21.3** | 84.2 → **91.1** |
| Transverse B / +2 | **0.102** → 0.267 | **0.065** → 0.081 | **5.11** → 13.33 | 0.122 → **0.119** | **11.4** → 37.7 | **98.1** → 68.6 |
| Transverse B / -2 | 0.175 → **0.136** | 0.092 → **0.045** | 8.75 → **6.80** | 0.157 → **0.105** | 69.6 → **14.4** | 34.0 → **89.5** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 89.3 → 89.3 | **16.5** → 28.6 | 43.37 → **44.71** | 12.63 → **11.29** | **0.00** → 1.18 | 18.73 → **17.79** | 1.39 → **1.19** |
| Trot / -2 | 86.7 → **89.4** | **14.7** → 18.3 | 41.77 → **44.43** | 14.23 → **11.57** | **0.00** → 0.03 | 23.34 → **18.46** | **1.00** → 1.12 |
| Bound / +2 | **81.6** → 78.4 | **49.9** → 72.5 | **44.54** → 43.98 | **11.46** → 12.02 | **8.35** → 11.39 | **25.95** → 29.27 | 1.43 → **1.04** |
| Bound / -2 | 80.5 → **82.9** | 51.9 → **46.1** | 43.46 → **46.02** | 12.54 → **9.98** | **8.33** → 8.72 | 27.88 → **23.42** | 2.00 → **1.35** |
| Half-bound A / +2 | **87.7** → 77.2 | **21.1** → 61.0 | **47.08** → 41.23 | **8.92** → 14.77 | **3.95** → 8.33 | **18.53** → 33.62 | 1.28 → **1.06** |
| Half-bound A / -2 | **81.9** → 80.0 | **36.9** → 52.0 | **43.06** → 42.51 | **12.94** → 13.49 | **6.05** → 6.63 | **27.09** → 29.97 | 2.00 → **1.05** |
| Half-bound B / +2 | 72.3 → **85.9** | 73.2 → **43.1** | 39.26 → **45.30** | 16.74 → **10.84** | 11.54 → **4.30** | 39.73 → **21.45** | 1.88 → **1.19** |
| Half-bound B / -2 | 72.8 → **82.5** | 67.0 → **36.5** | 42.92 → **44.58** | 13.08 → **11.42** | 16.67 → **6.33** | 35.16 → **25.86** | 2.27 → **1.41** |
| Rotary A / +2 | 73.4 → **75.8** | **23.5** → 69.3 | 44.51 → **47.48** | 11.49 → **8.65** | **16.33** → 17.81 | 34.35 → **29.02** | 2.01 → **1.90** |
| Rotary A / -2 | 67.9 → **71.5** | 48.9 → **44.2** | 43.30 → **44.54** | 12.70 → **11.46** | 21.70 → **18.00** | 39.87 → **36.41** | 2.00 → **1.12** |
| Rotary B / +2 | **78.4** → 70.8 | 56.1 → **11.4** | **45.25** → 41.83 | **10.75** → 14.17 | **11.89** → 15.05 | **28.81** → 39.86 | **1.36** → 2.00 |
| Rotary B / -2 | **71.5** → 68.1 | 45.7 → **42.1** | 40.99 → **44.56** | 15.01 → **11.44** | **14.71** → 22.99 | 38.84 → **38.52** | 2.00 → **1.86** |
| Transverse A / +2 | **90.1** → 86.1 | **8.7** → 31.7 | **49.05** → 43.90 | **7.14** → 12.10 | 2.72 → **2.36** | **15.34** → 22.58 | 1.44 → **1.24** |
| Transverse A / -2 | **88.4** → 84.3 | **18.0** → 40.9 | **44.21** → 43.65 | **11.79** → 12.35 | **0.03** → 4.16 | **20.25** → 24.41 | 2.00 → **1.00** |
| Transverse B / +2 | 80.1 → **88.3** | 43.7 → **37.5** | 38.70 → **45.18** | 17.30 → **10.82** | 2.79 → **2.37** | 32.66 → **18.66** | 1.84 → **1.19** |
| Transverse B / -2 | 81.4 → **88.9** | 35.2 → **21.8** | 43.09 → **44.67** | 12.91 → **11.33** | 6.79 → **0.00** | 27.46 → **19.37** | 1.67 → **1.00** |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 265.2 → **243.6** | **1.398** → 1.429 | 4.49 → **0.29** | 9.86 → **6.30** | 1.91 → **1.43** | **10.53** → 11.61 |
| Trot / -2 | 266.6 → **258.6** | 1.431 → **1.366** | 1.64 → **0.83** | 6.72 → **4.33** | 4.09 → **1.08** | 10.71 → **7.60** |
| Bound / +2 | 365.4 → **277.3** | 1.570 → **1.404** | 4.15 → **2.70** | 8.27 → **3.30** | 3.69 → **2.02** | **0.97** → 7.37 |
| Bound / -2 | 336.9 → **303.8** | 1.471 → **1.211** | 3.24 → **1.37** | **5.94** → 14.10 | 16.37 → **15.17** | **0.76** → 12.60 |
| Half-bound A / +2 | 329.9 → **326.7** | **1.492** → 1.659 | 3.12 → **1.71** | **8.33** → 13.23 | **1.02** → 6.72 | **11.71** → 17.35 |
| Half-bound A / -2 | 328.4 → **303.0** | **1.484** → 1.587 | **2.10** → 3.98 | **4.63** → 27.21 | 9.50 → **6.08** | **5.87** → 7.26 |
| Half-bound B / +2 | 322.8 → **269.6** | 1.558 → **1.407** | 6.39 → **2.73** | 4.17 → **2.50** | 8.88 → **8.52** | **12.21** → 14.93 |
| Half-bound B / -2 | **265.4** → 312.9 | 1.433 → **1.392** | 3.67 → **2.32** | 10.15 → **8.53** | **3.67** → 11.27 | **2.09** → 8.66 |
| Rotary A / +2 | 287.5 → **280.2** | 1.631 → **1.374** | 5.05 → **2.26** | 9.92 → **6.77** | 6.53 → **4.61** | **3.65** → 23.49 |
| Rotary A / -2 | 350.6 → **345.9** | 1.681 → **1.382** | 4.56 → **2.70** | **1.18** → 7.37 | **5.46** → 5.55 | **0.51** → 8.40 |
| Rotary B / +2 | 418.7 → **308.0** | 1.643 → **1.558** | **3.49** → 5.23 | 16.06 → **15.73** | **5.63** → 7.50 | **10.47** → 24.14 |
| Rotary B / -2 | **348.8** → 363.2 | **1.482** → 1.554 | 4.18 → **2.73** | **4.36** → 6.80 | **12.21** → 15.99 | **2.01** → 11.28 |
| Transverse A / +2 | **245.7** → 305.8 | **1.473** → 1.639 | 3.32 → **0.58** | **0.61** → 4.79 | **7.68** → 10.06 | 16.77 → **16.56** |
| Transverse A / -2 | 297.8 → **293.0** | **1.441** → 1.502 | **1.47** → 1.92 | **0.70** → 23.41 | 12.58 → **2.98** | 7.03 → **0.49** |
| Transverse B / +2 | 315.3 → **269.4** | **1.494** → 1.545 | 5.83 → **2.03** | **2.56** → 6.82 | **2.73** → 9.28 | **14.08** → 20.78 |
| Transverse B / -2 | **267.5** → 338.7 | 1.547 → **1.526** | 3.18 → **2.14** | 7.29 → **5.68** | 6.54 → **5.82** | **0.42** → 7.00 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | **100.0** → 10.0 | 100.0 → 100.0 |
| Trot / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 86.0 → **100.0** | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 99.0 | 100.0 → 100.0 |
| Half-bound A / +2 | **55.0** → 0.0 | **55.0** → 0.0 | 0.0 → 0.0 | **55.0** → 0.0 | 0.0 → 0.0 | **100.0** → 99.0 | **100.0** → 99.0 |
| Half-bound A / -2 | **9.0** → 0.0 | **9.0** → 0.0 | 0.0 → 0.0 | **9.0** → 0.0 | 0.0 → 0.0 | 9.0 → **100.0** | 9.0 → **100.0** |
| Half-bound B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Rotary A / +2 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | 0.0 → 0.0 | 73.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 0.0 → **96.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +2 | 87.0 → **100.0** | 87.0 → **100.0** | 0.0 → 0.0 | 87.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -2 | **98.0** → 0.0 | **98.0** → 0.0 | 0.0 → 0.0 | **98.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +2 | **99.0** → 94.0 | **99.0** → 94.0 | 0.0 → 0.0 | 74.0 → **94.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -2 | 99.0 → **100.0** | 99.0 → **100.0** | 0.0 → 0.0 | 99.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

### X1_go2bend: full gait/direction details

#### X1_go2bend, vx = ±0.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 0.027 → **0.025** | 0.045 → **0.030** | 5.30 → **4.93** | **0.034** → 0.063 | **10.0** → 45.2 | **95.0** → 70.9 |
| Trot / -0.5 | **0.027** → 0.045 | **0.033** → 0.048 | **5.33** → 8.99 | **0.049** → 0.061 | **7.7** → 39.1 | **96.3** → 77.3 |
| Bound / +0.5 | 0.131 → **0.080** | **0.032** → 0.037 | 26.20 → **16.07** | 0.085 → **0.045** | **18.9** → 46.2 | **118.7** → 80.1 |
| Bound / -0.5 | **0.069** → 0.082 | 0.042 → **0.037** | **13.71** → 16.43 | 0.133 → **0.075** | 113.0 → **14.1** | -16.5 → **79.5** |
| Half-bound A / +0.5 | 0.068 → **0.045** | **0.043** → 0.052 | 13.65 → **8.94** | 0.126 → **0.083** | **20.2** → 53.3 | **105.5** → 65.2 |
| Half-bound A / -0.5 | **0.074** → 0.131 | 0.051 → **0.046** | **14.70** → 26.18 | 0.075 → **0.064** | 65.2 → **57.0** | 44.8 → **45.3** |
| Half-bound B / +0.5 | 0.072 → **0.047** | 0.065 → **0.035** | 14.39 → **9.36** | 0.094 → **0.054** | **2.4** → 48.5 | **111.0** → 65.9 |
| Half-bound B / -0.5 | 0.051 → **0.050** | **0.033** → 0.052 | 10.19 → **10.02** | 0.089 → **0.044** | **23.1** → 30.6 | **99.1** → 77.9 |
| Rotary A / +0.5 | **0.039** → 0.061 | **0.058** → 0.068 | **7.72** → 12.18 | **0.111** → 0.121 | **82.4** → 122.6 | **25.3** → -28.7 |
| Rotary A / -0.5 | 0.076 → **0.035** | **0.045** → 0.055 | 15.21 → **6.91** | **0.055** → 0.069 | **8.2** → 9.4 | 111.9 → **100.2** |
| Rotary B / +0.5 | 0.039 → **0.036** | 0.082 → **0.077** | 7.81 → **7.17** | 0.117 → **0.100** | 83.5 → **71.6** | 23.6 → **37.8** |
| Rotary B / -0.5 | **0.065** → 0.171 | 0.065 → **0.055** | **12.97** → 34.15 | 0.100 → **0.075** | 94.7 → **48.3** | -2.4 → **48.0** |
| Transverse A / +0.5 | **0.044** → 0.068 | **0.029** → 0.034 | **8.86** → 13.52 | 0.078 → **0.065** | 68.6 → **53.3** | 42.1 → **63.3** |
| Transverse A / -0.5 | **0.059** → 0.088 | **0.022** → 0.037 | **11.75** → 17.56 | 0.111 → **0.051** | 105.6 → **35.3** | -11.6 → **72.9** |
| Transverse B / +0.5 | 0.061 → **0.053** | **0.014** → 0.025 | 12.29 → **10.67** | 0.069 → **0.042** | 9.5 → **3.7** | **104.2** → 94.8 |
| Transverse B / -0.5 | **0.041** → 0.059 | 0.040 → **0.036** | **8.10** → 11.87 | 0.100 → **0.047** | 48.2 → **20.3** | 67.9 → **84.8** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 97.3 → **100.0** | **10.0** → 13.0 | 52.50 → **54.17** | 3.50 → **2.17** | 0.00 → 0.00 | 4.69 → **0.00** | 1.00 → 1.00 |
| Trot / -0.5 | 96.4 → **98.2** | **10.4** → 21.0 | 51.67 → **52.50** | 4.33 → **3.83** | 0.00 → 0.00 | 6.25 → **3.12** | 1.00 → 1.00 |
| Bound / +0.5 | 91.1 → **91.4** | 43.0 → 43.0 | 51.67 → **52.79** | 6.33 → **4.12** | 6.25 → 6.25 | 10.94 → **10.40** | 1.00 → 1.00 |
| Bound / -0.5 | 91.8 → **92.0** | 35.6 → **30.0** | 52.53 → **53.33** | **3.47** → 4.67 | **5.34** → 6.25 | 10.30 → **9.38** | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.5 → **99.1** | 25.0 → **8.3** | 52.50 → **56.41** | 5.50 → **1.93** | 0.00 → 0.00 | 7.81 → **1.56** | 1.00 → 1.00 |
| Half-bound A / -0.5 | **96.0** → 95.5 | 20.0 → 20.0 | 55.00 → **56.67** | **2.17** → 3.33 | **3.12** → 4.17 | 4.69 → 4.69 | 1.00 → 1.00 |
| Half-bound B / +0.5 | 92.9 → **96.4** | 22.0 → **10.8** | **55.31** → 54.17 | **2.31** → 3.83 | 6.25 → **0.00** | 7.81 → **6.25** | 1.00 → 1.00 |
| Half-bound B / -0.5 | 90.2 → **97.3** | 30.0 → **14.9** | 54.17 → **55.01** | 4.17 → **3.35** | 8.33 → **0.00** | 10.94 → **4.69** | 1.00 → 1.00 |
| Rotary A / +0.5 | 77.7 → **88.4** | 33.5 → **31.0** | 61.67 → **60.00** | 8.33 → **4.00** | 31.25 → **16.67** | 15.62 → **7.81** | 1.00 → 1.00 |
| Rotary A / -0.5 | 78.6 → **85.2** | **24.0** → 45.6 | 52.50 → **54.64** | 5.50 → **3.82** | 18.75 → **14.68** | 23.44 → **14.81** | 1.00 → 1.00 |
| Rotary B / +0.5 | **90.2** → 87.5 | **29.4** → 41.0 | 59.17 → **57.50** | 5.83 → **1.50** | 14.58 → 14.58 | **6.25** → 10.94 | 1.00 → 1.00 |
| Rotary B / -0.5 | **84.9** → 84.4 | 45.7 → **45.5** | **55.62** → 55.44 | **3.48** → 4.06 | **15.67** → 16.67 | **14.66** → 14.79 | 1.00 → 1.00 |
| Transverse A / +0.5 | 95.5 → **99.1** | 17.0 → **14.0** | 52.50 → **53.96** | 3.83 → **2.38** | 2.08 → **0.00** | 6.25 → **1.56** | 1.00 → 1.00 |
| Transverse A / -0.5 | 98.2 → 98.2 | 16.0 → 16.0 | 52.50 → **55.71** | **3.50** → 4.05 | **0.00** → 2.08 | 3.12 → **1.56** | 1.00 → 1.00 |
| Transverse B / +0.5 | 97.3 → 97.3 | 19.0 → **15.0** | 52.50 → **54.17** | **3.50** → 3.83 | **0.00** → 2.08 | 4.69 → **3.12** | 1.00 → 1.00 |
| Transverse B / -0.5 | 97.3 → **99.1** | 12.0 → **7.0** | 53.33 → **55.83** | 3.00 → **2.83** | 0.00 → 0.00 | 4.69 → **1.56** | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 34.5 → **31.7** | 0.646 → **0.535** | 2.22 → **0.83** | 24.49 → **15.78** | **1.24** → 9.72 | **5.90** → 14.90 |
| Trot / -0.5 | 38.7 → **30.0** | 0.591 → **0.565** | 2.50 → **0.00** | 20.13 → **11.57** | **2.04** → 4.08 | 3.99 → **2.74** |
| Bound / +0.5 | **110.4** → 140.5 | 0.972 → **0.796** | **2.18** → 2.20 | 46.12 → **32.69** | **13.46** → 21.04 | **1.69** → 2.98 |
| Bound / -0.5 | 141.8 → **132.2** | 0.702 → **0.695** | 3.15 → **1.11** | 32.26 → **20.45** | **12.45** → 14.72 | **0.31** → 15.68 |
| Half-bound A / +0.5 | **93.1** → 103.4 | 0.816 → **0.760** | **2.78** → 3.61 | 44.14 → **32.02** | 13.02 → **8.77** | 5.43 → **2.26** |
| Half-bound A / -0.5 | 118.3 → **94.9** | 0.712 → **0.697** | 2.72 → **2.22** | 36.41 → **27.86** | **10.90** → 13.39 | 8.90 → **7.10** |
| Half-bound B / +0.5 | **93.9** → 121.9 | 0.812 → **0.732** | 5.24 → **1.39** | **22.63** → 32.96 | 14.10 → **12.51** | 8.12 → **0.81** |
| Half-bound B / -0.5 | **96.8** → 117.2 | 0.746 → **0.613** | 3.89 → **1.67** | 28.25 → **17.03** | **8.98** → 16.55 | **0.81** → 5.49 |
| Rotary A / +0.5 | **68.7** → 110.7 | 0.814 → **0.604** | **1.11** → 1.39 | **15.45** → 22.95 | **9.00** → 15.36 | 15.59 → **7.01** |
| Rotary A / -0.5 | **139.4** → 146.1 | 0.748 → **0.673** | **1.94** → 2.50 | **14.01** → 22.68 | **3.18** → 18.48 | 11.65 → **0.70** |
| Rotary B / +0.5 | **88.9** → 110.1 | **0.701** → 0.722 | 3.89 → 3.89 | **31.84** → 37.21 | 11.45 → **5.74** | **11.96** → 15.46 |
| Rotary B / -0.5 | 120.3 → **90.7** | **0.665** → 0.738 | 3.47 → **3.06** | 28.58 → **21.49** | **13.02** → 14.70 | **0.96** → 14.89 |
| Transverse A / +0.5 | **70.6** → 76.1 | 0.731 → **0.659** | **1.39** → 2.78 | 10.48 → **9.45** | **8.97** → 9.43 | 18.56 → **14.20** |
| Transverse A / -0.5 | **83.2** → 87.2 | 0.619 → **0.602** | 2.22 → **0.28** | **10.03** → 18.30 | **8.38** → 11.94 | 11.38 → **6.43** |
| Transverse B / +0.5 | **69.6** → 83.2 | 0.640 → **0.561** | 4.17 → **1.39** | **5.81** → 6.60 | **3.77** → 12.16 | **5.95** → 16.81 |
| Transverse B / -0.5 | 82.6 → **79.3** | 0.645 → **0.601** | 2.78 → **1.39** | 23.47 → **14.19** | **6.70** → 12.59 | **1.19** → 21.77 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -0.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Half-bound A / +0.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary A / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1_go2bend, vx = ±1 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | **0.034** → 0.042 | 0.037 → **0.031** | **3.40** → 4.19 | **0.051** → 0.052 | **3.8** → 23.1 | **100.4** → 94.2 |
| Trot / -1 | 0.084 → **0.063** | 0.048 → **0.028** | 8.45 → **6.31** | **0.071** → 0.073 | 47.7 → **33.4** | 60.8 → **82.0** |
| Bound / +1 | 0.181 → **0.163** | **0.034** → 0.046 | 18.14 → **16.30** | 0.168 → **0.094** | 100.9 → **6.7** | -7.9 → **114.5** |
| Bound / -1 | 0.069 → **0.054** | 0.041 → **0.039** | 6.85 → **5.40** | 0.125 → **0.086** | 95.3 → **12.8** | 3.6 → **94.2** |
| Half-bound A / +1 | **0.063** → 0.082 | **0.031** → 0.057 | **6.26** → 8.15 | **0.094** → 0.129 | **28.2** → 117.3 | **91.3** → -21.7 |
| Half-bound A / -1 | **0.090** → 0.104 | **0.047** → 0.072 | **8.98** → 10.40 | **0.047** → 0.084 | **14.0** → 52.7 | **92.3** → 60.0 |
| Half-bound B / +1 | **0.078** → 0.084 | 0.060 → **0.052** | **7.85** → 8.44 | 0.093 → **0.076** | **1.0** → 5.8 | **95.3** → 105.0 |
| Half-bound B / -1 | **0.070** → 0.084 | **0.045** → 0.066 | **6.96** → 8.44 | 0.131 → **0.079** | **49.5** → 53.4 | **66.3** → 53.4 |
| Rotary A / +1 | **0.064** → 0.078 | **0.051** → 0.070 | **6.43** → 7.75 | 0.094 → **0.088** | **43.2** → 83.2 | **71.4** → 20.4 |
| Rotary A / -1 | **0.075** → 0.080 | **0.043** → 0.044 | **7.45** → 7.98 | **0.107** → 0.141 | 74.7 → **67.7** | 35.5 → **41.5** |
| Rotary B / +1 | **0.083** → 0.116 | **0.064** → 0.087 | **8.27** → 11.56 | **0.118** → 0.191 | **7.9** → 125.8 | **105.2** → -37.8 |
| Rotary B / -1 | **0.068** → 0.147 | **0.052** → 0.081 | **6.76** → 14.71 | **0.055** → 0.078 | **19.5** → 20.8 | **90.1** → 82.1 |
| Transverse A / +1 | **0.040** → 0.075 | **0.032** → 0.060 | **4.02** → 7.51 | 0.077 → **0.059** | 69.3 → **29.6** | 41.2 → **91.1** |
| Transverse A / -1 | **0.075** → 0.107 | 0.052 → **0.039** | **7.47** → 10.73 | 0.111 → **0.055** | 105.5 → **9.7** | -7.5 → **90.2** |
| Transverse B / +1 | **0.064** → 0.078 | **0.027** → 0.057 | **6.42** → 7.78 | **0.071** → 0.078 | **1.6** → 13.9 | 103.1 → **101.2** |
| Transverse B / -1 | **0.060** → 0.068 | **0.055** → 0.063 | **6.03** → 6.84 | 0.088 → **0.077** | 29.1 → **18.6** | 87.8 → **91.6** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 96.4 → 96.4 | 13.0 → **10.0** | 50.83 → **51.67** | 5.17 → **4.33** | 0.00 → 0.00 | 6.25 → 6.25 | 1.00 → 1.00 |
| Trot / -1 | 96.4 → **97.3** | **6.1** → 15.7 | **52.29** → 51.67 | **3.71** → 4.33 | 0.00 → 0.00 | 6.25 → **4.69** | 1.00 → 1.00 |
| Bound / +1 | 88.4 → **89.5** | 58.0 → **46.2** | **50.83** → 50.81 | 8.83 → **6.15** | 8.33 → **6.71** | 14.06 → **13.33** | 1.00 → 1.00 |
| Bound / -1 | 88.4 → 88.4 | 44.9 → **40.0** | 50.83 → **52.50** | 5.17 → **3.83** | **6.25** → 8.33 | 15.62 → **14.06** | 1.00 → 1.00 |
| Half-bound A / +1 | 93.8 → **97.3** | 26.0 → **16.0** | 50.83 → **56.67** | 7.17 → **2.00** | **0.00** → 4.17 | 10.94 → **1.56** | 1.00 → 1.00 |
| Half-bound A / -1 | 93.8 → 93.8 | **25.0** → 30.0 | 51.67 → 51.67 | **4.33** → 4.67 | 2.08 → 2.08 | 9.38 → 9.38 | 1.00 → 1.00 |
| Half-bound B / +1 | 93.2 → **93.8** | 25.5 → **17.0** | **52.81** → 50.00 | **3.85** → 6.33 | 4.17 → **0.00** | **8.79** → 10.94 | 1.00 → 1.00 |
| Half-bound B / -1 | 86.6 → **94.6** | 45.0 → **25.0** | 48.33 → **52.50** | 8.00 → **3.83** | 6.25 → **2.08** | 18.75 → **7.81** | 1.00 → 1.00 |
| Rotary A / +1 | 71.5 → **84.8** | 47.7 → **31.0** | **55.86** → 55.00 | 7.53 → **3.00** | 31.25 → **14.58** | 26.51 → **15.62** | **1.00** → 2.00 |
| Rotary A / -1 | 76.8 → **77.5** | 46.0 → **40.6** | 50.83 → **51.54** | 5.50 → **4.80** | 20.83 → 20.83 | 25.00 → **23.68** | **1.00** → 1.44 |
| Rotary B / +1 | 85.3 → **86.2** | **29.4** → 42.5 | 54.58 → **56.67** | 5.42 → **3.33** | **14.58** → 16.67 | 14.84 → **11.72** | 2.00 → **1.00** |
| Rotary B / -1 | 81.4 → **83.0** | 63.8 → **54.0** | **52.79** → 51.67 | **3.21** → 4.67 | 18.10 → **14.58** | 18.99 → **18.75** | 1.00 → 1.00 |
| Transverse A / +1 | 94.6 → **95.5** | 22.0 → 22.0 | 51.67 → **52.50** | 4.33 → **3.83** | 2.08 → 2.08 | 7.81 → **6.25** | 1.00 → 1.00 |
| Transverse A / -1 | 96.4 → **98.2** | 26.0 → **21.0** | 50.83 → **53.33** | 5.17 → **3.33** | 0.00 → 0.00 | 6.25 → **3.12** | 1.00 → 1.00 |
| Transverse B / +1 | **94.1** → 93.8 | 29.8 → **29.0** | **50.47** → 50.00 | **5.53** → 8.00 | **0.00** → 2.08 | 10.25 → **9.38** | 1.00 → 1.00 |
| Transverse B / -1 | 95.5 → 95.5 | **17.0** → 21.0 | **51.67** → 50.83 | **4.67** → 5.50 | 0.00 → 0.00 | 7.81 → 7.81 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 80.4 → **72.6** | 0.849 → **0.708** | 2.22 → **1.67** | 3.38 → **1.19** | **1.08** → 2.40 | 12.01 → **11.47** |
| Trot / -1 | 74.4 → **69.6** | **0.651** → 0.752 | 2.64 → **0.00** | 17.25 → **9.43** | **1.15** → 3.93 | **0.32** → 4.56 |
| Bound / +1 | **184.1** → 241.7 | 1.198 → **1.113** | **2.78** → 2.85 | **33.72** → 41.85 | **11.55** → 19.02 | **7.88** → 7.90 |
| Bound / -1 | 208.8 → **193.1** | 0.954 → **0.887** | 4.44 → **2.50** | 21.21 → **10.58** | **1.56** → 12.53 | 4.91 → **2.11** |
| Half-bound A / +1 | **152.1** → 162.3 | 1.003 → **0.827** | **2.78** → 4.17 | **29.44** → 32.90 | 10.93 → **4.15** | 8.75 → **1.15** |
| Half-bound A / -1 | 172.7 → **154.7** | 0.899 → **0.851** | 4.17 → **2.22** | 25.47 → **11.29** | **9.06** → 11.57 | 5.26 → **0.78** |
| Half-bound B / +1 | **147.0** → 182.8 | 1.035 → **1.002** | 5.00 → **1.94** | **21.31** → 35.07 | **8.60** → 16.29 | **6.64** → 6.66 |
| Half-bound B / -1 | **125.8** → 175.9 | **0.767** → 0.831 | 3.61 → **3.06** | **10.28** → 15.00 | 7.65 → **6.04** | **0.43** → 8.46 |
| Rotary A / +1 | **142.6** → 160.6 | 1.041 → **0.908** | 2.22 → **1.39** | **13.48** → 27.35 | **11.43** → 14.48 | 9.50 → **2.85** |
| Rotary A / -1 | 200.1 → **183.7** | 0.922 → **0.887** | 5.00 → **1.94** | 14.08 → **3.22** | 9.50 → **2.45** | **2.26** → 6.90 |
| Rotary B / +1 | **161.4** → 179.2 | 0.965 → **0.857** | **2.64** → 3.89 | **21.22** → 40.44 | 10.68 → **3.51** | **15.75** → 18.48 |
| Rotary B / -1 | 198.8 → **154.2** | 0.938 → **0.860** | 4.44 → **3.33** | 20.59 → **7.46** | **3.17** → 12.99 | **0.20** → 8.68 |
| Transverse A / +1 | **94.8** → 104.5 | 0.845 → **0.828** | **1.67** → 2.78 | **2.90** → 8.86 | 6.41 → **5.90** | 17.66 → **14.14** |
| Transverse A / -1 | **124.6** → 130.6 | **0.728** → 0.776 | 3.06 → **0.56** | 12.60 → **6.16** | **9.61** → 10.27 | 6.89 → **1.73** |
| Transverse B / +1 | **101.9** → 129.2 | 0.861 → **0.839** | 4.72 → **1.67** | 4.99 → **0.66** | **0.66** → 11.39 | **14.05** → 19.22 |
| Transverse B / -1 | **115.4** → 118.8 | 0.741 → **0.704** | 2.78 → **1.11** | **8.61** → 8.73 | 9.06 → **7.55** | **1.20** → 13.78 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Bound / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1_go2bend, vx = ±1.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 0.058 → **0.056** | 0.071 → **0.056** | 3.85 → **3.72** | 0.064 → **0.056** | 7.5 → **5.1** | **100.1** → 100.8 |
| Trot / -1.5 | 0.174 → **0.103** | 0.043 → **0.030** | 11.58 → **6.87** | 0.087 → **0.051** | 57.5 → **4.2** | 49.1 → **95.3** |
| Bound / +1.5 | **0.057** → 0.078 | **0.054** → 0.061 | **3.83** → 5.17 | 0.093 → **0.089** | 25.4 → **5.1** | 90.2 → **103.4** |
| Bound / -1.5 | 0.179 → **0.066** | 0.154 → **0.061** | 11.96 → **4.42** | 0.155 → **0.127** | 98.5 → **52.4** | 5.2 → **62.6** |
| Half-bound A / +1.5 | **0.058** → 0.076 | **0.048** → 0.090 | **3.89** → 5.03 | **0.107** → 0.110 | 30.0 → **8.3** | 87.3 → **102.7** |
| Half-bound A / -1.5 | 0.126 → **0.079** | 0.090 → **0.042** | 8.37 → **5.26** | 0.151 → **0.070** | 118.8 → **15.5** | -20.9 → **93.0** |
| Half-bound B / +1.5 | 0.073 → **0.052** | **0.061** → 0.076 | 4.86 → **3.47** | 0.128 → **0.071** | 24.0 → **2.2** | 91.5 → **98.7** |
| Half-bound B / -1.5 | **0.094** → 0.150 | **0.038** → 0.073 | **6.27** → 10.02 | 0.130 → **0.096** | 28.1 → **16.9** | **87.0** → 85.7 |
| Rotary A / +1.5 | 0.080 → **0.076** | **0.060** → 0.094 | 5.33 → **5.04** | 0.120 → **0.092** | 73.9 → **60.9** | 29.7 → **53.5** |
| Rotary A / -1.5 | 0.100 → **0.083** | 0.084 → **0.074** | 6.65 → **5.51** | 0.105 → **0.081** | 61.6 → **28.9** | 50.8 → **87.3** |
| Rotary B / +1.5 | **0.072** → 0.106 | **0.070** → 0.076 | **4.78** → 7.05 | 0.143 → **0.078** | 78.1 → **25.6** | 28.1 → **86.7** |
| Rotary B / -1.5 | 0.145 → **0.086** | **0.062** → 0.084 | 9.66 → **5.74** | **0.114** → 0.115 | 42.8 → **4.4** | 69.9 → **95.3** |
| Transverse A / +1.5 | 0.093 → 0.093 | **0.069** → 0.115 | 6.20 → **6.19** | 0.079 → **0.067** | **2.1** → 20.0 | 95.1 → **95.9** |
| Transverse A / -1.5 | 0.096 → **0.088** | 0.062 → **0.031** | 6.41 → **5.84** | 0.088 → **0.055** | 73.2 → **24.7** | 34.5 → **87.5** |
| Transverse B / +1.5 | 0.105 → **0.079** | 0.080 → **0.064** | 6.98 → **5.27** | **0.066** → 0.099 | **9.0** → 17.4 | 104.3 → **97.6** |
| Transverse B / -1.5 | **0.092** → 0.106 | 0.051 → **0.037** | **6.15** → 7.04 | 0.138 → **0.086** | **18.3** → 43.1 | **92.6** → 69.3 |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 92.0 → **95.6** | 18.0 → **13.0** | 46.67 → **50.03** | 9.33 → **5.97** | 0.00 → 0.00 | 14.06 → **7.76** | **1.00** → 1.03 |
| Trot / -1.5 | 94.6 → 94.6 | **15.7** → 20.0 | **49.14** → 48.33 | **6.86** → 7.67 | 0.00 → 0.00 | 9.42 → **9.38** | 1.00 → 1.00 |
| Bound / +1.5 | 82.8 → **83.8** | 65.3 → **64.2** | 49.01 → 49.01 | **6.99** → 7.27 | 10.55 → **10.35** | 22.27 → **20.65** | **1.06** → 1.12 |
| Bound / -1.5 | **83.0** → 80.1 | **60.0** → 66.8 | **49.17** → 46.38 | **6.83** → 9.62 | 10.42 → 10.42 | **21.88** → 27.10 | 1.00 → 1.00 |
| Half-bound A / +1.5 | **89.1** → 88.0 | **26.3** → 31.0 | 49.14 → **50.44** | 7.19 → **5.89** | **4.36** → 6.25 | **15.82** → 16.31 | 2.22 → **2.00** |
| Half-bound A / -1.5 | **90.2** → 88.4 | **40.0** → 49.4 | **49.17** → 47.50 | **6.83** → 8.50 | 4.17 → 4.17 | **14.06** → 17.19 | 1.00 → 1.00 |
| Half-bound B / +1.5 | 84.3 → **89.2** | 38.3 → **36.0** | **47.84** → 46.57 | **8.16** → 9.77 | 8.33 → **2.08** | 21.24 → **17.38** | **1.97** → 2.07 |
| Half-bound B / -1.5 | 83.0 → **89.1** | **31.5** → 38.1 | 46.12 → **46.89** | 9.88 → **9.11** | 8.33 → **2.21** | 23.44 → **17.38** | 1.00 → 1.00 |
| Rotary A / +1.5 | 66.3 → **80.9** | **34.8** → 37.2 | 50.21 → **51.17** | 7.86 → **5.41** | 31.25 → **16.30** | 35.55 → **21.19** | **1.50** → 2.00 |
| Rotary A / -1.5 | **73.2** → 69.1 | **52.7** → 63.6 | **46.67** → 46.13 | **9.33** → 9.87 | **20.83** → 25.00 | **31.25** → 35.38 | 1.00 → 1.00 |
| Rotary B / +1.5 | **80.4** → 79.0 | **49.8** → 57.0 | **51.63** → 51.25 | 4.77 → **4.75** | **17.18** → 18.75 | **21.45** → 22.66 | 2.09 → **2.00** |
| Rotary B / -1.5 | **77.3** → 75.0 | **46.1** → 49.1 | **50.65** → 48.31 | **5.35** → 7.69 | **20.04** → 20.83 | **24.75** → 28.17 | **1.00** → 1.97 |
| Transverse A / +1.5 | 89.1 → **92.7** | **37.0** → 43.7 | 46.53 → **50.05** | 9.47 → **6.08** | 2.08 → 2.08 | 17.44 → **11.14** | 1.76 → **1.13** |
| Transverse A / -1.5 | **92.9** → 92.0 | **26.0** → 36.0 | **47.50** → 46.67 | **8.50** → 9.33 | 0.00 → 0.00 | **12.50** → 14.06 | 1.00 → 1.00 |
| Transverse B / +1.5 | 90.2 → **91.1** | 47.0 → **31.5** | 46.99 → **47.33** | 9.01 → **8.70** | **0.00** → 1.65 | 17.19 → **14.29** | **1.00** → 2.27 |
| Transverse B / -1.5 | 90.2 → **91.1** | 32.0 → **31.0** | **47.50** → 45.84 | **8.50** → 10.16 | 2.08 → **0.00** | 15.62 → 15.62 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | **151.8** → 157.0 | 1.013 → **0.963** | **2.50** → 3.33 | **4.34** → 8.22 | 4.15 → **3.81** | 16.21 → **9.00** |
| Trot / -1.5 | 136.0 → **135.8** | **0.758** → 0.881 | 3.89 → **0.00** | 5.76 → **4.30** | 7.39 → **5.37** | **3.15** → 10.22 |
| Bound / +1.5 | **283.3** → 327.5 | 1.554 → **1.384** | 4.70 → **4.18** | **20.58** → 25.90 | **12.11** → 17.84 | 9.41 → **6.48** |
| Bound / -1.5 | 245.8 → **230.7** | **1.036** → 1.156 | 6.39 → **3.89** | 8.88 → **1.88** | 5.19 → **3.75** | 8.38 → **2.09** |
| Half-bound A / +1.5 | **221.1** → 231.5 | 1.213 → **1.137** | **3.59** → 3.89 | **3.82** → 17.75 | **6.73** → 9.27 | 22.12 → **1.86** |
| Half-bound A / -1.5 | 224.0 → **199.0** | **1.000** → 1.096 | 5.28 → **2.50** | 15.51 → **1.36** | **5.51** → 8.61 | 5.14 → **2.90** |
| Half-bound B / +1.5 | **214.8** → 253.1 | 1.349 → **1.241** | 5.00 → **3.33** | **10.85** → 12.99 | **7.95** → 13.77 | **9.11** → 11.63 |
| Half-bound B / -1.5 | **177.2** → 252.7 | **0.940** → 1.051 | **3.61** → 3.98 | **2.49** → 11.68 | **11.17** → 13.46 | 7.42 → **2.82** |
| Rotary A / +1.5 | **209.6** → 245.0 | 1.249 → **1.202** | **2.78** → 3.17 | **0.44** → 13.84 | **0.02** → 8.77 | 13.90 → **4.06** |
| Rotary A / -1.5 | 258.4 → **229.0** | **1.103** → 1.191 | 6.39 → **3.06** | 5.01 → **3.86** | 2.17 → **2.04** | 4.80 → **2.58** |
| Rotary B / +1.5 | 339.4 → **222.8** | 1.309 → **1.117** | 4.00 → **3.89** | 35.66 → **27.31** | 6.51 → **0.97** | **0.43** → 13.90 |
| Rotary B / -1.5 | 245.1 → **221.7** | **1.070** → 1.211 | 5.83 → **3.61** | 14.70 → **0.70** | **5.31** → 6.17 | 7.22 → **2.27** |
| Transverse A / +1.5 | **156.3** → 189.2 | **0.995** → 1.066 | 1.99 → **1.90** | **2.21** → 11.09 | 1.24 → **0.40** | 13.62 → **6.45** |
| Transverse A / -1.5 | **177.3** → 177.9 | **0.795** → 0.955 | 3.33 → **0.83** | 7.62 → **0.90** | **6.64** → 8.94 | 1.30 → **1.06** |
| Transverse B / +1.5 | **171.4** → 218.0 | 1.133 → **1.093** | 4.44 → **0.56** | 10.20 → **6.81** | **1.65** → 2.62 | 19.43 → **16.70** |
| Transverse B / -1.5 | 177.0 → **164.2** | 0.930 → **0.799** | 3.06 → **1.39** | **4.47** → 8.09 | **2.62** → 12.70 | 4.47 → **1.93** |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### X1_go2bend, vx = ±2 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 0.088 → **0.075** | 0.065 → **0.058** | 4.40 → **3.76** | 0.082 → **0.074** | 4.7 → **3.5** | 96.8 → **98.6** |
| Trot / -2 | 0.358 → **0.247** | 0.079 → **0.044** | 17.91 → **12.36** | 0.104 → **0.068** | 62.3 → **3.6** | 42.9 → **88.3** |
| Bound / +2 | **0.078** → 0.080 | 0.079 → **0.063** | **3.91** → 4.02 | 0.129 → **0.103** | 11.3 → **3.4** | 96.9 → **100.9** |
| Bound / -2 | 0.375 → **0.229** | 0.167 → **0.116** | 18.73 → **11.44** | **0.157** → 0.172 | **74.0** → 80.7 | **32.8** → 23.6 |
| Half-bound A / +2 | **0.078** → 0.151 | **0.105** → 0.130 | **3.89** → 7.54 | **0.118** → 0.123 | 19.8 → **18.6** | 95.4 → **101.2** |
| Half-bound A / -2 | 0.251 → **0.188** | 0.087 → **0.063** | 12.53 → **9.39** | 0.131 → **0.086** | 74.0 → **9.4** | 31.4 → **90.4** |
| Half-bound B / +2 | 0.152 → **0.080** | 0.094 → **0.077** | 7.62 → **4.00** | 0.153 → **0.084** | **27.9** → 28.6 | **91.7** → 88.4 |
| Half-bound B / -2 | 0.267 → **0.262** | **0.055** → 0.083 | 13.33 → **13.11** | 0.149 → **0.109** | **2.3** → 59.1 | **87.6** → 44.6 |
| Rotary A / +2 | 0.091 → **0.079** | 0.113 → **0.050** | 4.54 → **3.94** | 0.130 → **0.071** | 76.8 → **7.3** | 24.3 → **99.4** |
| Rotary A / -2 | 0.260 → **0.182** | 0.148 → **0.081** | 12.99 → **9.09** | 0.165 → **0.109** | **22.6** → 69.7 | **78.0** → 37.2 |
| Rotary B / +2 | **0.087** → 0.111 | **0.063** → 0.119 | **4.36** → 5.54 | 0.136 → **0.105** | **13.3** → 20.3 | 96.2 → **97.4** |
| Rotary B / -2 | 0.328 → **0.201** | 0.108 → **0.094** | 16.41 → **10.06** | 0.144 → **0.143** | 42.7 → **19.7** | 64.9 → **85.4** |
| Transverse A / +2 | **0.079** → 0.096 | 0.122 → **0.101** | **3.93** → 4.80 | **0.136** → 0.174 | **82.0** → 96.7 | **14.4** → -0.7 |
| Transverse A / -2 | **0.195** → 0.203 | 0.093 → **0.045** | **9.74** → 10.13 | 0.139 → **0.062** | 108.0 → **4.8** | -16.2 → **90.3** |
| Transverse B / +2 | 0.146 → **0.077** | 0.141 → **0.086** | 7.32 → **3.83** | **0.101** → 0.106 | **28.1** → 53.9 | **90.0** → 63.1 |
| Transverse B / -2 | 0.265 → **0.216** | 0.090 → **0.051** | 13.23 → **10.78** | 0.171 → **0.096** | **20.7** → 51.6 | **80.5** → 55.7 |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 90.5 → **92.6** | 23.6 → **16.3** | 44.51 → **46.41** | 11.49 → **9.59** | 0.00 → 0.00 | 16.55 → **12.99** | **1.50** → 1.88 |
| Trot / -2 | 88.5 → **91.1** | 20.7 → **20.0** | 43.46 → **45.42** | 12.54 → **10.58** | 0.00 → 0.00 | 20.07 → **15.62** | 1.00 → 1.00 |
| Bound / +2 | **74.2** → 73.3 | **75.9** → 87.8 | **43.70** → 42.79 | **12.30** → 13.21 | **13.87** → 14.84 | **34.72** → 35.55 | **2.00** → 2.09 |
| Bound / -2 | **77.1** → 75.9 | 75.1 → **62.3** | **45.31** → 42.45 | **10.69** → 13.55 | 12.50 → **10.29** | **30.66** → 34.38 | 2.00 → **1.03** |
| Half-bound A / +2 | **81.8** → 79.7 | **35.3** → 44.6 | 43.20 → **44.52** | 12.80 → **11.48** | **5.21** → 9.58 | **28.03** → 28.40 | 2.00 → 2.00 |
| Half-bound A / -2 | **82.6** → 81.7 | 54.5 → **40.0** | **43.75** → 42.92 | **12.25** → 13.08 | 6.25 → 6.25 | **25.78** → 27.34 | 1.00 → 1.00 |
| Half-bound B / +2 | 75.3 → **81.6** | **50.2** → 57.4 | 42.29 → **43.94** | 13.71 → **12.06** | 11.07 → **6.65** | 34.91 → **27.28** | 2.09 → **2.06** |
| Half-bound B / -2 | 75.0 → **79.1** | **48.0** → 58.1 | **42.50** → 42.03 | **13.50** → 13.97 | 12.50 → **8.23** | 34.38 → **30.49** | 1.00 → 1.00 |
| Rotary A / +2 | 60.5 → **70.7** | **19.9** → 43.8 | 42.44 → **44.95** | 13.56 → **11.05** | 28.14 → **20.54** | 48.03 → **35.90** | **2.01** → 2.08 |
| Rotary A / -2 | **65.5** → 65.4 | **42.8** → 66.4 | 41.12 → **42.78** | 15.21 → **13.22** | **22.92** → 25.00 | 43.21 → **41.71** | 2.00 → **1.29** |
| Rotary B / +2 | **73.2** → 68.5 | **62.0** → 75.8 | **46.17** → 44.73 | **9.83** → 11.27 | **20.26** → 23.82 | **31.76** → 37.24 | 2.00 → 2.00 |
| Rotary B / -2 | **70.8** → 68.2 | **2.0** → 28.8 | **46.08** → 43.39 | **9.92** → 12.61 | **20.83** → 22.43 | **35.48** → 38.79 | **1.00** → 1.24 |
| Transverse A / +2 | 84.8 → 84.8 | **44.5** → 48.9 | **42.49** → 42.39 | **13.51** → 13.61 | **2.08** → 2.32 | 25.04 → **24.89** | **1.85** → 2.47 |
| Transverse A / -2 | 86.6 → **87.8** | 37.0 → **30.3** | **44.17** → 43.62 | **11.83** → 12.38 | 2.08 → **0.00** | 21.88 → **21.34** | 2.00 → **1.00** |
| Transverse B / +2 | 85.3 → **87.3** | 55.7 → **31.0** | 43.01 → **43.12** | 12.99 → **12.88** | 0.48 → **0.00** | 25.33 → **22.27** | **1.84** → 2.00 |
| Transverse B / -2 | **83.9** → 83.2 | 42.0 → **37.7** | **41.67** → 41.02 | **14.33** → 14.98 | 2.08 → 2.08 | **26.56** → 27.77 | 2.00 → **1.00** |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | **235.7** → 266.1 | **1.197** → 1.246 | 3.19 → **3.14** | **3.19** → 15.89 | 7.92 → **7.75** | 11.72 → **4.96** |
| Trot / -2 | 199.3 → **186.3** | **0.843** → 0.933 | 3.89 → **0.00** | **0.29** → 2.65 | **0.67** → 5.58 | **8.52** → 12.36 |
| Bound / +2 | **352.9** → 354.5 | 1.704 → **1.574** | 4.32 → **4.17** | **7.42** → 9.35 | **0.17** → 2.12 | 8.38 → **8.21** |
| Bound / -2 | **281.5** → 296.4 | **1.147** → 1.324 | 6.45 → **4.05** | **8.98** → 13.16 | 6.15 → **1.63** | 10.73 → **2.54** |
| Half-bound A / +2 | **352.0** → 372.7 | 1.520 → **1.505** | 4.24 → **3.23** | **0.46** → 8.89 | **2.23** → 6.92 | 12.88 → **3.41** |
| Half-bound A / -2 | **250.3** → 252.9 | **1.040** → 1.281 | 4.72 → **3.61** | **9.22** → 9.96 | 7.12 → **2.37** | **2.86** → 3.82 |
| Half-bound B / +2 | 328.1 → **301.0** | 1.613 → **1.313** | 5.73 → **2.90** | **0.04** → 4.17 | 5.00 → **2.60** | **4.63** → 15.26 |
| Half-bound B / -2 | **263.5** → 278.7 | 1.095 → **1.043** | **4.17** → 4.55 | 15.77 → **1.07** | **0.67** → 8.01 | 3.49 → **3.13** |
| Rotary A / +2 | 372.2 → **313.7** | 1.591 → **1.339** | 3.46 → **3.41** | 9.15 → **0.34** | 5.81 → **1.24** | **7.95** → 13.24 |
| Rotary A / -2 | 295.2 → **293.9** | 1.297 → **1.249** | 6.91 → **3.52** | **1.77** → 15.88 | 2.26 → **0.93** | 10.57 → **1.65** |
| Rotary B / +2 | 423.1 → **347.0** | 1.649 → **1.516** | 4.73 → **3.61** | 15.06 → **14.83** | **3.31** → 8.88 | **0.69** → 12.25 |
| Rotary B / -2 | 276.8 → **265.7** | **1.129** → 1.369 | 6.39 → **4.60** | **11.42** → 15.71 | 10.00 → **1.15** | 9.67 → **0.31** |
| Transverse A / +2 | **290.0** → 322.0 | **1.247** → 1.311 | 2.80 → **1.92** | 14.73 → **11.79** | **3.07** → 5.87 | 20.60 → **2.29** |
| Transverse A / -2 | 241.4 → **239.3** | **0.942** → 1.080 | 3.61 → **1.67** | 4.57 → **1.41** | 7.27 → **1.14** | **0.16** → 3.88 |
| Transverse B / +2 | **292.2** → 297.9 | 1.431 → **1.260** | 5.39 → **1.12** | 5.08 → **1.40** | **1.55** → 3.60 | **7.40** → 14.62 |
| Transverse B / -2 | 243.1 → **234.4** | 1.051 → **0.882** | 3.61 → **1.14** | **3.77** → 6.29 | **5.38** → 8.39 | 5.74 → **0.41** |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +2 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +2 | **78.0** → 7.0 | **78.0** → 7.0 | 0.0 → 0.0 | **78.0** → 7.0 | 0.0 → 0.0 | **100.0** → 49.0 | 100.0 → 100.0 |
| Transverse A / -2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Transverse B / +2 | 19.0 → **94.0** | 19.0 → **94.0** | 0.0 → 0.0 | 19.0 → **94.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -2 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

### Go2: full gait/direction details

#### Go2, vx = ±0.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | **0.032** → 0.044 | 0.039 → **0.035** | **6.41** → 8.79 | **0.074** → 0.078 | 22.7 → **22.0** | **96.6** → 87.1 |
| Trot / -0.5 | 0.065 → **0.018** | 0.068 → **0.055** | 12.92 → **3.51** | **0.046** → 0.063 | **2.5** → 33.6 | **87.3** → 87.1 |
| Bound / +0.5 | **0.037** → 0.050 | **0.031** → 0.035 | **7.49** → 9.90 | **0.042** → 0.074 | **51.0** → 75.6 | **66.4** → 30.4 |
| Bound / -0.5 | **0.045** → 0.052 | 0.033 → **0.032** | **9.07** → 10.32 | **0.068** → 0.071 | 52.6 → **9.9** | 57.6 → **93.3** |
| Half-bound A / +0.5 | 0.034 → 0.034 | 0.065 → **0.053** | **6.72** → 6.79 | **0.048** → 0.064 | **7.1** → 24.8 | **100.4** → 90.8 |
| Half-bound A / -0.5 | **0.039** → 0.083 | 0.071 → **0.063** | **7.83** → 16.56 | 0.068 → **0.057** | 30.9 → **2.4** | **87.3** → 84.1 |
| Half-bound B / +0.5 | **0.041** → 0.043 | 0.065 → **0.056** | **8.30** → 8.61 | 0.087 → **0.048** | 66.7 → **40.2** | 37.0 → **75.5** |
| Half-bound B / -0.5 | **0.039** → 0.059 | **0.047** → 0.065 | **7.76** → 11.83 | 0.052 → 0.052 | 77.1 → **23.2** | 28.4 → **81.7** |
| Rotary A / +0.5 | **0.051** → 0.059 | **0.119** → 0.120 | **10.29** → 11.83 | **0.061** → 0.072 | 52.1 → **13.9** | 57.6 → **88.1** |
| Rotary A / -0.5 | **0.043** → 0.055 | **0.104** → 0.137 | **8.62** → 11.07 | 0.072 → **0.053** | 98.9 → **8.0** | -8.4 → **89.9** |
| Rotary B / +0.5 | **0.042** → 0.056 | 0.126 → **0.123** | **8.34** → 11.19 | 0.091 → **0.062** | 26.9 → **8.5** | 90.4 → **96.1** |
| Rotary B / -0.5 | 0.054 → **0.040** | **0.115** → 0.123 | 10.82 → **7.99** | 0.132 → **0.064** | 101.1 → **2.7** | 0.8 → **98.8** |
| Transverse A / +0.5 | **0.071** → 0.081 | 0.026 → **0.018** | **14.22** → 16.23 | **0.037** → 0.091 | **7.9** → 22.6 | **87.5** → 79.2 |
| Transverse A / -0.5 | **0.032** → 0.051 | 0.056 → **0.029** | **6.32** → 10.12 | **0.038** → 0.060 | **19.3** → 46.6 | **97.5** → 63.6 |
| Transverse B / +0.5 | 0.035 → **0.023** | 0.028 → **0.018** | 7.01 → **4.66** | **0.042** → 0.070 | 29.9 → **3.8** | 83.7 → **100.7** |
| Transverse B / -0.5 | 0.047 → **0.023** | **0.026** → 0.076 | 9.34 → **4.54** | **0.059** → 0.065 | 46.1 → **24.1** | 64.4 → **98.9** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 99.0 → **100.0** | 10.0 → 10.0 | 54.81 → **55.77** | 2.88 → **1.92** | 0.00 → 0.00 | 1.79 → **0.00** | 1.00 → 1.00 |
| Trot / -0.5 | 96.2 → **99.0** | **3.3** → 5.1 | 53.52 → **55.77** | 2.48 → **1.92** | 0.00 → 0.00 | 6.58 → **1.79** | 1.00 → 1.00 |
| Bound / +0.5 | 92.7 → **92.8** | 30.6 → **27.1** | 56.73 → **56.18** | **6.73** → 8.10 | 10.00 → **7.88** | **5.36** → 6.65 | 1.00 → 1.00 |
| Bound / -0.5 | 93.8 → **96.3** | 10.0 → **4.9** | 53.85 → **58.15** | 5.77 → **4.35** | **2.50** → 3.75 | 8.93 → **3.62** | 1.00 → 1.00 |
| Half-bound A / +0.5 | 95.8 → **96.9** | 19.3 → **14.4** | 58.65 → 58.65 | 4.81 → 4.81 | 4.88 → 4.88 | 3.64 → **1.82** | 1.00 → 1.00 |
| Half-bound A / -0.5 | 96.9 → 96.9 | 13.8 → **8.8** | **56.73** → 59.62 | 4.81 → **4.69** | **2.44** → 4.88 | 3.64 → **1.82** | 1.00 → 1.00 |
| Half-bound B / +0.5 | 92.7 → **96.9** | 24.4 → **14.4** | 59.62 → **57.69** | 7.69 → **3.85** | 9.76 → **4.88** | 5.45 → **1.82** | 1.00 → 1.00 |
| Half-bound B / -0.5 | **95.8** → 94.8 | **5.0** → 15.0 | 60.55 → **59.59** | 5.68 → **4.67** | 7.32 → 7.32 | **1.87** → 3.68 | 1.00 → 1.00 |
| Rotary A / +0.5 | 96.9 → 96.9 | **10.6** → 18.2 | 60.58 → **59.62** | 5.65 → **4.69** | 7.14 → 7.14 | 0.00 → 0.00 | 1.00 → 1.00 |
| Rotary A / -0.5 | 96.9 → **97.9** | 13.8 → **8.8** | 59.62 → **56.73** | 3.62 → **2.88** | 4.76 → **2.38** | 1.85 → 1.85 | 1.00 → 1.00 |
| Rotary B / +0.5 | **96.9** → 95.8 | **15.0** → 18.2 | **59.62** → 61.54 | 5.77 → **5.54** | **7.14** → 9.52 | 0.00 → 0.00 | 1.00 → 1.00 |
| Rotary B / -0.5 | 95.8 → **96.9** | 17.6 → **12.6** | 58.65 → 58.65 | 3.73 → 3.73 | 4.76 → 4.76 | 3.70 → **1.85** | 1.00 → 1.00 |
| Transverse A / +0.5 | 96.9 → 96.9 | 11.8 → 11.8 | 56.73 → 56.73 | 4.81 → 4.81 | 4.76 → 4.76 | 1.85 → 1.85 | 1.00 → 1.00 |
| Transverse A / -0.5 | 96.9 → 96.9 | 11.2 → **9.4** | **55.77** → 58.17 | **3.85** → 5.29 | 4.76 → 4.76 | 1.85 → 1.85 | 1.00 → 1.00 |
| Transverse B / +0.5 | 94.3 → **94.8** | 14.0 → **12.4** | 58.17 → **57.69** | **4.33** → 5.77 | 8.33 → **7.14** | 3.70 → 3.70 | 1.00 → 1.00 |
| Transverse B / -0.5 | **95.8** → 94.8 | **10.0** → 16.2 | **55.77** → 56.73 | 5.77 → **4.81** | **4.76** → 7.14 | 3.70 → 3.70 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 30.6 → **24.8** | 0.300 → **0.278** | 0.00 → 0.00 | 24.55 → **1.80** | 23.90 → **14.95** | 29.55 → **15.79** |
| Trot / -0.5 | **24.4** → 25.9 | 0.298 → **0.284** | 0.00 → 0.00 | 34.03 → **7.12** | 32.22 → **16.28** | 49.85 → **14.20** |
| Bound / +0.5 | 87.3 → **79.7** | **0.307** → 0.330 | 0.96 → **0.00** | 24.18 → **17.60** | **31.89** → 33.99 | **30.11** → 36.26 |
| Bound / -0.5 | 108.7 → **98.8** | **0.272** → 0.288 | 0.64 → **0.00** | 12.62 → **5.68** | 29.43 → **25.35** | 34.48 → **18.18** |
| Half-bound A / +0.5 | 84.8 → **77.6** | 0.295 → **0.272** | 1.60 → **0.00** | 26.93 → **10.26** | 29.76 → **27.08** | 40.90 → **34.47** |
| Half-bound A / -0.5 | 93.0 → **73.1** | **0.259** → 0.272 | 1.60 → **0.00** | 11.48 → **1.25** | 32.76 → **30.58** | 41.53 → **33.17** |
| Half-bound B / +0.5 | 80.0 → **74.0** | **0.284** → 0.303 | 2.24 → **0.00** | **22.01** → 24.70 | 33.91 → **32.75** | 39.56 → **33.47** |
| Half-bound B / -0.5 | 95.6 → **77.3** | **0.251** → 0.269 | 1.60 → **0.00** | 10.28 → **2.62** | 29.56 → **27.50** | 33.28 → **31.87** |
| Rotary A / +0.5 | 74.1 → **67.9** | **0.243** → 0.253 | 1.92 → **0.00** | 27.06 → **26.54** | **31.37** → 34.70 | 55.99 → **49.15** |
| Rotary A / -0.5 | 101.9 → **70.9** | 0.258 → **0.253** | 1.60 → **0.00** | 15.54 → **8.51** | **26.29** → 32.81 | **43.51** → 54.87 |
| Rotary B / +0.5 | 68.3 → **63.5** | **0.263** → 0.271 | 1.28 → **0.00** | 27.15 → **13.31** | 35.41 → **33.81** | **56.35** → 56.67 |
| Rotary B / -0.5 | 91.1 → **73.5** | 0.264 → **0.254** | 1.28 → **0.00** | **2.70** → 8.85 | **29.02** → 29.90 | **39.07** → 49.41 |
| Transverse A / +0.5 | **62.9** → 65.6 | 0.282 → **0.276** | 0.64 → **0.00** | 21.98 → **18.78** | 25.84 → **25.16** | 30.11 → **29.58** |
| Transverse A / -0.5 | 78.3 → **63.3** | 0.293 → **0.282** | 0.96 → **0.00** | 15.49 → **0.09** | 30.35 → **27.97** | 44.18 → **31.67** |
| Transverse B / +0.5 | 62.4 → **59.7** | **0.282** → 0.292 | 1.60 → **0.00** | **30.29** → 32.79 | **26.89** → 28.77 | **34.31** → 39.74 |
| Transverse B / -0.5 | 65.3 → **63.8** | **0.289** → 0.294 | 1.60 → **0.00** | 14.59 → **7.60** | 30.11 → **28.53** | 35.31 → **32.25** |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +0.5 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → **100.0** | 100.0 → 100.0 |
| Rotary B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +0.5 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -0.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### Go2, vx = ±1 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 0.048 → **0.033** | **0.029** → 0.032 | 4.76 → **3.26** | 0.096 → **0.082** | **28.7** → 43.5 | **90.7** → 75.9 |
| Trot / -1 | **0.031** → 0.042 | 0.072 → **0.044** | **3.08** → 4.19 | 0.068 → **0.050** | 43.2 → **16.9** | 74.8 → **92.5** |
| Bound / +1 | **0.078** → 0.081 | **0.040** → 0.041 | **7.79** → 8.11 | 0.078 → **0.070** | 53.2 → **34.1** | 62.3 → **87.9** |
| Bound / -1 | 0.161 → **0.112** | 0.056 → **0.054** | 16.07 → **11.23** | **0.052** → 0.098 | **5.2** → 56.7 | **83.0** → 51.4 |
| Half-bound A / +1 | 0.092 → **0.064** | 0.071 → **0.069** | 9.25 → **6.36** | **0.046** → 0.079 | **4.0** → 60.2 | **104.2** → 50.0 |
| Half-bound A / -1 | 0.099 → **0.067** | 0.086 → **0.074** | 9.90 → **6.71** | 0.101 → **0.055** | 70.4 → **0.9** | 33.1 → **94.3** |
| Half-bound B / +1 | **0.078** → 0.090 | **0.044** → 0.047 | **7.77** → 8.95 | 0.134 → **0.076** | 90.5 → **52.0** | 8.3 → **62.9** |
| Half-bound B / -1 | 0.084 → **0.062** | 0.087 → **0.082** | 8.36 → **6.16** | **0.054** → 0.073 | 63.0 → **41.2** | 47.8 → **74.1** |
| Rotary A / +1 | **0.059** → 0.086 | 0.118 → **0.111** | **5.91** → 8.59 | **0.051** → 0.114 | **4.5** → 39.0 | **101.7** → 77.5 |
| Rotary A / -1 | 0.123 → **0.070** | **0.114** → 0.131 | 12.35 → **7.00** | **0.047** → 0.094 | **76.9** → 114.5 | **21.5** → -21.4 |
| Rotary B / +1 | 0.068 → **0.050** | 0.132 → **0.118** | 6.82 → **5.03** | 0.081 → **0.065** | 28.8 → **1.3** | 90.1 → **100.1** |
| Rotary B / -1 | 0.143 → **0.059** | **0.118** → 0.157 | 14.32 → **5.92** | **0.084** → 0.126 | **15.6** → 17.8 | 84.3 → **98.6** |
| Transverse A / +1 | 0.073 → **0.070** | **0.023** → 0.031 | 7.32 → **7.00** | **0.071** → 0.108 | 23.6 → **23.4** | **89.9** → 87.1 |
| Transverse A / -1 | 0.060 → **0.048** | 0.072 → **0.033** | 6.02 → **4.76** | **0.063** → 0.092 | **23.7** → 58.1 | **86.8** → 52.3 |
| Transverse B / +1 | 0.070 → **0.055** | **0.018** → 0.021 | 7.04 → **5.49** | **0.054** → 0.087 | **35.7** → 39.4 | **79.9** → 75.3 |
| Transverse B / -1 | 0.061 → **0.032** | **0.066** → 0.075 | 6.08 → **3.22** | 0.089 → **0.064** | 74.0 → **18.6** | 34.3 → **93.3** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 97.9 → **100.0** | 10.0 → 10.0 | 53.85 → **55.77** | 3.85 → **1.92** | 0.00 → 0.00 | 3.57 → **0.00** | 1.00 → 1.00 |
| Trot / -1 | 94.8 → 94.8 | 10.1 → **2.1** | 50.96 → **52.49** | 5.04 → **3.51** | 0.00 → 0.00 | 8.93 → 8.93 | 1.00 → 1.00 |
| Bound / +1 | **87.3** → 84.4 | **23.5** → 26.2 | 57.26 → **55.77** | **5.61** → 7.69 | **15.63** → 20.00 | **10.66** → 12.50 | **1.53** → 2.00 |
| Bound / -1 | **91.7** → 91.6 | **9.9** → 14.8 | **54.81** → 57.67 | **4.81** → 5.79 | **5.00** → 10.00 | 10.71 → **7.19** | 1.00 → 1.00 |
| Half-bound A / +1 | 94.6 → **96.9** | 26.2 → **11.2** | **57.44** → 58.65 | 5.24 → **3.73** | 5.24 → **4.88** | 5.45 → **1.82** | **1.00** → 2.00 |
| Half-bound A / -1 | 94.8 → **95.8** | 13.8 → **10.0** | **56.73** → 57.67 | 6.73 → **2.74** | 4.88 → 4.88 | 5.45 → **3.68** | 1.00 → 1.00 |
| Half-bound B / +1 | 90.0 → **94.8** | 36.2 → **26.2** | 56.35 → **55.77** | 8.18 → **6.85** | 11.16 → **4.88** | 9.09 → **5.45** | 1.00 → 1.00 |
| Half-bound B / -1 | 93.8 → 93.8 | **10.0** → 15.0 | **54.81** → 57.69 | 6.73 → **3.85** | **4.88** → 7.32 | 7.27 → **5.45** | 2.00 → **1.00** |
| Rotary A / +1 | 91.7 → **94.8** | **5.6** → 15.6 | 57.69 → **56.73** | **3.85** → 4.81 | 9.52 → **4.76** | 7.41 → **5.56** | 2.00 → 2.00 |
| Rotary A / -1 | 92.7 → **93.8** | 18.8 → 18.8 | **55.77** → 54.81 | 5.77 → **3.96** | 7.14 → **4.76** | 7.41 → 7.41 | 2.00 → **1.00** |
| Rotary B / +1 | 93.1 → **93.8** | 10.6 → **10.0** | 58.10 → **56.73** | 3.48 → **2.88** | 9.52 → **7.14** | **4.77** → 5.56 | 2.00 → 2.00 |
| Rotary B / -1 | 93.7 → **94.8** | 22.6 → **12.9** | **56.73** → 54.43 | 4.81 → **4.80** | 5.96 → **4.76** | 6.48 → **5.56** | **1.00** → 2.00 |
| Transverse A / +1 | **96.9** → 94.8 | 26.8 → **22.4** | 54.81 → **55.77** | 5.88 → **3.85** | **4.76** → 7.14 | **1.85** → 3.70 | 1.00 → 1.00 |
| Transverse A / -1 | **95.6** → 94.8 | 16.2 → **11.2** | 53.06 → **53.85** | 5.10 → **4.92** | **4.01** → 4.76 | **4.63** → 5.56 | 1.00 → 1.00 |
| Transverse B / +1 | 91.7 → **96.9** | **17.4** → 21.8 | 56.73 → 56.73 | 6.73 → **4.81** | 9.52 → **4.76** | 7.41 → **1.85** | 1.00 → 1.00 |
| Transverse B / -1 | **93.8** → 90.6 | **11.2** → 22.4 | 52.88 → **53.85** | 5.88 → **5.77** | **4.76** → 9.52 | **7.41** → 9.26 | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 66.4 → **62.0** | 0.364 → **0.322** | 0.00 → 0.00 | **7.33** → 15.98 | **14.07** → 16.72 | **15.74** → 24.41 |
| Trot / -1 | **58.0** → 59.0 | 0.358 → **0.325** | 0.00 → 0.00 | 27.23 → **23.78** | 30.20 → **19.55** | 52.19 → **30.60** |
| Bound / +1 | 122.9 → **114.2** | 0.436 → **0.397** | 1.07 → **0.00** | 9.00 → **3.79** | **23.10** → 24.17 | **20.44** → 24.93 |
| Bound / -1 | 134.6 → **131.9** | **0.385** → 0.422 | 0.96 → **0.64** | 9.22 → **1.16** | 29.44 → **25.49** | 35.39 → **22.28** |
| Half-bound A / +1 | 120.0 → **113.2** | 0.450 → **0.358** | 1.92 → **0.00** | 23.43 → **3.81** | 25.03 → **24.19** | 36.19 → **29.65** |
| Half-bound A / -1 | 126.3 → **116.7** | 0.383 → **0.375** | 1.60 → **0.00** | 13.61 → **3.22** | 34.67 → **30.95** | 50.37 → **39.25** |
| Half-bound B / +1 | **103.8** → 106.2 | **0.408** → 0.429 | 2.25 → **0.32** | **12.37** → 26.17 | **27.66** → 31.25 | **38.35** → 39.30 |
| Half-bound B / -1 | **119.6** → 122.5 | **0.347** → 0.387 | 1.92 → **0.96** | 15.46 → **0.88** | 32.90 → **29.80** | 46.06 → **29.36** |
| Rotary A / +1 | **90.5** → 99.0 | **0.358** → 0.373 | 1.60 → **0.32** | **19.92** → 24.27 | **25.29** → 26.18 | 50.73 → **33.93** |
| Rotary A / -1 | 118.7 → **118.4** | 0.375 → **0.367** | 1.92 → **0.64** | 22.93 → **1.27** | **30.14** → 30.88 | 49.14 → **48.34** |
| Rotary B / +1 | **91.8** → 107.7 | 0.355 → **0.338** | 0.96 → **0.00** | 16.57 → **14.56** | 25.44 → **23.43** | 42.38 → **40.53** |
| Rotary B / -1 | 123.4 → **106.9** | 0.381 → **0.370** | 1.28 → **0.00** | **9.58** → 15.08 | 33.40 → **32.57** | **48.68** → 55.07 |
| Transverse A / +1 | 83.3 → **79.3** | 0.377 → **0.356** | 1.28 → **0.96** | 20.17 → **17.82** | **21.12** → 21.41 | 28.87 → **28.30** |
| Transverse A / -1 | 114.1 → **97.8** | 0.414 → **0.372** | 1.28 → **0.64** | 12.81 → **7.22** | 33.08 → **26.88** | 53.54 → **37.65** |
| Transverse B / +1 | 80.0 → **79.3** | **0.372** → 0.384 | 1.92 → **0.32** | **17.32** → 29.41 | **22.69** → 26.76 | **25.70** → 38.27 |
| Transverse B / -1 | 95.1 → **87.4** | **0.353** → 0.381 | 1.60 → **0.00** | **9.08** → 13.89 | **28.08** → 33.11 | **39.27** → 44.57 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 0.0 → **100.0** | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / +1 | 0.0 → **100.0** | 0.0 → **100.0** | **100.0** → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / +1 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | **100.0** → 0.0 | 100.0 → 100.0 | **100.0** → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Rotary B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |

#### Go2, vx = ±1.5 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | **0.095** → 0.111 | 0.031 → **0.023** | **6.35** → 7.42 | **0.092** → 0.117 | **20.9** → 49.4 | **87.5** → 60.8 |
| Trot / -1.5 | **0.165** → 0.172 | 0.181 → **0.098** | **10.99** → 11.46 | 0.106 → **0.097** | 97.0 → **75.8** | 7.9 → **30.9** |
| Bound / +1.5 | 0.172 → **0.149** | 0.065 → **0.048** | 11.50 → **9.90** | **0.072** → 0.073 | **1.4** → 5.0 | 92.3 → **94.1** |
| Bound / -1.5 | 0.389 → **0.124** | **0.089** → 0.119 | 25.94 → **8.23** | **0.063** → 0.177 | **20.2** → 128.9 | **67.3** → -37.5 |
| Half-bound A / +1.5 | 0.170 → **0.127** | **0.065** → 0.087 | 11.31 → **8.49** | 0.097 → **0.072** | 33.8 → **12.0** | 79.8 → **94.3** |
| Half-bound A / -1.5 | 0.324 → **0.253** | 0.120 → **0.094** | 21.60 → **16.84** | **0.095** → 0.098 | **9.4** → 64.4 | **75.2** → 40.3 |
| Half-bound B / +1.5 | **0.170** → 0.221 | **0.049** → 0.051 | **11.37** → 14.73 | 0.115 → **0.096** | **31.8** → 36.1 | **79.0** → 70.6 |
| Half-bound B / -1.5 | 0.286 → **0.142** | **0.148** → 0.218 | 19.06 → **9.44** | **0.123** → 0.272 | **98.7** → 112.1 | **3.3** → -15.0 |
| Rotary A / +1.5 | **0.145** → 0.209 | 0.110 → **0.097** | **9.69** → 13.94 | **0.118** → 0.166 | **17.8** → 89.3 | **89.0** → 8.4 |
| Rotary A / -1.5 | 0.315 → **0.115** | **0.124** → 0.133 | 20.99 → **7.68** | **0.082** → 0.150 | **111.5** → 129.4 | **-14.1** → -39.3 |
| Rotary B / +1.5 | 0.115 → 0.115 | 0.143 → **0.109** | 7.67 → 7.67 | 0.102 → **0.073** | 63.8 → **36.2** | 52.3 → **79.6** |
| Rotary B / -1.5 | 0.325 → **0.237** | **0.106** → 0.128 | 21.64 → **15.81** | 0.152 → **0.135** | 79.5 → **5.1** | 19.0 → **84.9** |
| Transverse A / +1.5 | **0.134** → 0.135 | **0.049** → 0.090 | **8.92** → 9.00 | **0.101** → 0.145 | **5.6** → 56.9 | **93.7** → 48.4 |
| Transverse A / -1.5 | 0.290 → **0.225** | 0.143 → **0.089** | 19.30 → **15.00** | **0.094** → 0.102 | **10.7** → 25.9 | **80.2** → 74.5 |
| Transverse B / +1.5 | **0.148** → 0.177 | 0.055 → **0.035** | **9.87** → 11.80 | **0.093** → 0.106 | **50.3** → 57.7 | **61.9** → 49.0 |
| Transverse B / -1.5 | 0.261 → **0.227** | 0.163 → **0.078** | 17.40 → **15.10** | 0.130 → **0.092** | 107.4 → **57.2** | -3.3 → **50.1** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 95.3 → **95.8** | 15.6 → **10.0** | **52.40** → 51.85 | **4.44** → 4.99 | 2.50 → **0.00** | **6.25** → 7.27 | 1.00 → 1.00 |
| Trot / -1.5 | 90.6 → **93.2** | 19.9 → **2.5** | 49.04 → **50.94** | 6.96 → **5.06** | 2.50 → **0.00** | 14.29 → **11.65** | 1.00 → 1.00 |
| Bound / +1.5 | 82.2 → **83.9** | 38.9 → **19.9** | **55.87** → 58.17 | 6.74 → **6.37** | **19.34** → 22.50 | 16.64 → **11.61** | **1.00** → 2.00 |
| Bound / -1.5 | **90.6** → 81.9 | 24.8 → **9.9** | 53.82 → **54.20** | **3.87** → 5.28 | **7.50** → 15.92 | **10.76** → 19.64 | **1.00** → 1.97 |
| Half-bound A / +1.5 | **92.8** → 91.7 | 32.5 → **6.2** | **54.48** → 53.86 | 6.06 → **4.80** | 6.80 → **4.88** | **7.56** → 10.85 | **1.05** → 2.00 |
| Half-bound A / -1.5 | **95.5** → 94.8 | 14.6 → **10.0** | 53.52 → **55.74** | 3.52 → **1.95** | **2.44** → 4.88 | 6.08 → **5.50** | 1.34 → **1.00** |
| Half-bound B / +1.5 | 87.0 → **94.8** | 42.8 → **31.8** | **54.76** → 52.88 | **5.71** → 7.81 | 12.99 → **4.88** | 13.01 → **5.45** | 1.01 → **1.00** |
| Half-bound B / -1.5 | **90.5** → 89.6 | 20.0 → **8.8** | 49.59 → **51.92** | 9.44 → **6.85** | **2.78** → 4.88 | 14.55 → 14.55 | 2.00 → 2.00 |
| Rotary A / +1.5 | 85.9 → **90.1** | **12.2** → 17.4 | **56.73** → 54.33 | **6.29** → 6.37 | 15.48 → **7.14** | 12.96 → **12.04** | 2.00 → 2.00 |
| Rotary A / -1.5 | **89.6** → 87.5 | 23.8 → **22.6** | **54.81** → 50.00 | **5.88** → 6.85 | 9.52 → **7.14** | **11.11** → 16.67 | 2.00 → 2.00 |
| Rotary B / +1.5 | 83.7 → **89.6** | 18.8 → **13.2** | 52.01 → **55.77** | 4.10 → **3.85** | 12.81 → **11.90** | 19.06 → **9.26** | 2.00 → 2.00 |
| Rotary B / -1.5 | 87.4 → **94.8** | 34.1 → **19.7** | **54.46** → 53.11 | 8.11 → **3.73** | 11.55 → **4.76** | 13.34 → **5.56** | **1.01** → 2.00 |
| Transverse A / +1.5 | **93.2** → 90.6 | 36.2 → **32.4** | **53.76** → 51.92 | 6.97 → **6.85** | **7.13** → 7.14 | **6.51** → 11.11 | 1.23 → **1.00** |
| Transverse A / -1.5 | **94.3** → 92.7 | **16.2** → 18.3 | 50.48 → **52.45** | 5.52 → **3.92** | **3.57** → 5.44 | **7.41** → 8.76 | 1.50 → **1.00** |
| Transverse B / +1.5 | 90.6 → **93.6** | **22.4** → 30.9 | **55.73** → 53.24 | **3.83** → 7.43 | 9.52 → **7.14** | 9.31 → **5.78** | **1.00** → 1.20 |
| Transverse B / -1.5 | **92.0** → 89.6 | **15.6** → 22.4 | 49.01 → **52.88** | 7.83 → **5.88** | **2.38** → 9.52 | 12.31 → **11.11** | 1.00 → 1.00 |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 110.9 → **109.6** | 0.435 → **0.383** | 0.32 → **0.18** | **9.04** → 10.37 | **10.31** → 16.37 | **14.49** → 22.23 |
| Trot / -1.5 | 103.6 → **101.8** | 0.470 → **0.406** | 0.32 → 0.32 | **19.43** → 24.41 | 36.02 → **26.45** | 55.77 → **47.96** |
| Bound / +1.5 | **184.9** → 190.8 | 0.597 → **0.559** | 2.68 → **1.44** | 10.89 → **5.52** | **18.12** → 22.05 | **12.94** → 22.95 |
| Bound / -1.5 | **159.4** → 273.0 | **0.565** → 0.708 | **1.28** → 2.07 | **7.83** → 34.33 | **34.26** → 37.52 | 36.62 → **22.62** |
| Half-bound A / +1.5 | **168.2** → 224.7 | **0.588** → 0.590 | 2.30 → **1.26** | 12.39 → **11.37** | **20.68** → 22.22 | **20.10** → 22.15 |
| Half-bound A / -1.5 | 165.4 → **151.2** | 0.533 → **0.525** | 1.60 → **1.28** | 14.32 → **9.72** | **37.48** → 42.11 | 53.84 → **48.71** |
| Half-bound B / +1.5 | **132.6** → 159.3 | **0.539** → 0.574 | 2.55 → **1.60** | **11.69** → 14.38 | **18.65** → 21.83 | 31.44 → **29.55** |
| Half-bound B / -1.5 | **145.9** → 263.8 | **0.506** → 0.639 | 2.24 → 2.24 | **20.32** → 32.23 | 41.86 → **38.36** | 58.95 → **32.29** |
| Rotary A / +1.5 | **148.9** → 155.7 | **0.507** → 0.524 | 3.13 → **0.96** | 27.39 → **10.29** | 21.25 → **21.22** | 43.93 → **22.64** |
| Rotary A / -1.5 | **134.5** → 211.6 | **0.507** → 0.599 | **1.92** → 2.88 | 30.36 → **21.08** | **32.49** → 35.44 | 59.60 → **43.03** |
| Rotary B / +1.5 | **170.2** → 185.9 | 0.512 → **0.508** | 1.38 → **0.00** | 13.40 → **8.57** | **15.26** → 21.18 | **32.13** → 34.66 |
| Rotary B / -1.5 | **162.6** → 171.0 | 0.566 → **0.531** | 1.60 → **0.64** | 10.90 → **3.99** | **33.87** → 42.76 | **42.31** → 56.62 |
| Transverse A / +1.5 | **132.5** → 137.5 | 0.500 → **0.462** | **1.75** → 2.24 | **0.47** → 1.99 | **19.67** → 20.78 | **19.74** → 26.82 |
| Transverse A / -1.5 | 147.2 → **137.4** | 0.535 → **0.483** | **1.60** → 2.23 | 15.09 → **12.66** | 37.26 → **31.37** | 57.46 → **49.55** |
| Transverse B / +1.5 | 136.1 → **132.1** | **0.513** → 0.548 | 2.88 → **1.66** | **0.68** → 24.47 | **15.01** → 24.83 | **19.03** → 35.52 |
| Transverse B / -1.5 | 131.6 → **116.4** | 0.483 → **0.477** | 2.17 → **1.28** | **12.69** → 21.38 | **34.61** → 35.37 | **51.75** → 57.02 |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / +1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Half-bound A / +1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / +1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | **50.0** → 0.0 | 100.0 → 100.0 |
| Rotary A / +1.5 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B / +1.5 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -1.5 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / +1.5 | **56.0** → 0.0 | **56.0** → 0.0 | 0.0 → 0.0 | **56.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse A / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / -1.5 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 3.0 → **100.0** | 100.0 → 100.0 |

#### Go2, vx = ±2 m/s

**Tracking and direction control**

| Gait / vx command [m/s] | vx RMSE [m/s] ↓ | vy RMSE [m/s] ↓ | Relative vx RMSE [%] ↓ | Yaw RMSE [rad/s] ↓ | Heading RMSE [degrees] ↓ | Progress [%] ≈100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | **0.439** → 0.479 | 0.080 → **0.049** | **21.94** → 23.96 | 0.133 → **0.100** | 71.8 → **50.5** | 25.4 → **47.9** |
| Trot / -2 | N/A → 0.503 | N/A → 0.146 | N/A → 25.15 | N/A → 0.122 | N/A → 52.6 | N/A → 44.2 |
| Bound / +2 | 0.459 → **0.435** | **0.048** → 0.083 | 22.93 → **21.75** | **0.115** → 0.117 | 30.4 → **7.5** | 67.4 → **79.6** |
| Bound / -2 | 0.759 → **0.439** | **0.168** → 0.197 | 37.94 → **21.94** | **0.098** → 0.224 | **35.3** → 127.9 | **53.9** → -28.4 |
| Half-bound A / +2 | 0.453 → **0.420** | **0.071** → 0.095 | 22.65 → **21.02** | **0.127** → 0.145 | 80.2 → **43.3** | 15.7 → **59.7** |
| Half-bound A / -2 | 0.717 → **0.643** | **0.113** → 0.172 | 35.85 → **32.17** | **0.133** → 0.174 | **48.4** → 121.9 | **39.9** → -15.2 |
| Half-bound B / +2 | 0.551 → **0.490** | 0.066 → **0.065** | 27.57 → **24.51** | 0.140 → **0.128** | 54.6 → **25.5** | 45.0 → **70.3** |
| Half-bound B / -2 | N/A → 0.392 | N/A → 0.274 | N/A → 19.62 | N/A → 0.305 | N/A → 117.8 | N/A → -22.8 |
| Rotary A / +2 | 0.546 → **0.514** | 0.129 → **0.094** | 27.29 → **25.71** | **0.101** → 0.138 | **43.6** → 61.5 | **56.6** → 37.1 |
| Rotary A / -2 | 0.696 → **0.363** | **0.138** → 0.155 | 34.79 → **18.16** | **0.151** → 0.196 | **128.5** → 130.3 | **-26.5** → -37.0 |
| Rotary B / +2 | 0.432 → **0.384** | 0.145 → **0.108** | 21.62 → **19.18** | 0.136 → **0.094** | 94.2 → **39.7** | 8.2 → **64.4** |
| Rotary B / -2 | 0.699 → **0.656** | **0.109** → 0.143 | 34.94 → **32.80** | **0.110** → 0.205 | **5.6** → 131.0 | **63.8** → -31.7 |
| Transverse A / +2 | **0.397** → 0.430 | **0.086** → 0.176 | **19.87** → 21.48 | 0.152 → **0.132** | **10.3** → 85.2 | **79.0** → 5.6 |
| Transverse A / -2 | 0.707 → **0.573** | **0.173** → 0.210 | 35.37 → **28.66** | 0.138 → **0.122** | **30.4** → 61.2 | **52.2** → 43.7 |
| Transverse B / +2 | **0.494** → 0.520 | 0.096 → **0.053** | **24.69** → 25.98 | 0.161 → **0.134** | 87.9 → **10.9** | 12.5 → **73.6** |
| Transverse B / -2 | **0.655** → 0.656 | **0.176** → 0.190 | **32.75** → 32.80 | 0.302 → **0.109** | 79.0 → **46.6** | 21.6 → **52.7** |

**Contact timing and execution**

| Gait / vx command [m/s] | Agreement [%] ↑ | Touchdown error [ms] ↓ | Measured duty [%], target 56 | Duty error [pp] ↓ | Swing contact [%] ↓ | Missed stance [%] ↓ | Max-foot TD/cycle ≈1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 93.6 → **95.6** | 15.0 → **11.8** | 51.75 → **52.15** | 5.20 → **3.85** | 2.50 → **0.84** | 9.16 → **6.86** | 1.74 → **1.04** |
| Trot / -2 | N/A → 90.6 | N/A → 12.4 | N/A → 48.48 | N/A → 7.52 | N/A → 1.24 | N/A → 15.17 | N/A → 1.10 |
| Bound / +2 | 79.8 → **80.1** | 38.6 → **24.8** | 54.66 → **55.04** | **6.28** → 6.73 | **20.26** → 22.19 | 20.11 → **18.22** | **1.00** → 1.97 |
| Bound / -2 | **86.9** → 79.2 | 35.6 → **20.0** | 49.11 → **50.00** | **6.89** → 10.69 | **8.22** → 17.50 | **16.54** → 23.21 | **1.16** → 2.00 |
| Half-bound A / +2 | **90.0** → 89.3 | 37.0 → **13.8** | 52.86 → **53.44** | 5.51 → **2.93** | **7.57** → 7.79 | **11.77** → 12.84 | **1.55** → 2.00 |
| Half-bound A / -2 | 88.6 → **91.6** | **19.9** → 20.0 | **50.99** → 50.97 | 6.71 → **5.87** | 7.32 → **4.97** | 14.50 → **10.96** | 2.00 → **1.04** |
| Half-bound B / +2 | 83.1 → **90.8** | 48.5 → **33.0** | **53.76** → 52.80 | **6.17** → 7.06 | 16.22 → **7.89** | 17.34 → **10.12** | **1.28** → 1.45 |
| Half-bound B / -2 | N/A → 86.0 | N/A → 15.6 | N/A → 51.19 | N/A → 6.36 | N/A → 8.19 | N/A → 18.41 | N/A → 2.00 |
| Rotary A / +2 | 83.7 → **88.6** | 19.9 → **18.2** | **54.81** → 53.03 | 5.88 → **5.80** | 17.48 → **9.62** | 15.45 → **12.76** | 2.00 → 2.00 |
| Rotary A / -2 | **89.1** → 82.3 | **26.3** → 32.6 | **52.40** → 47.76 | **4.44** → 9.09 | **8.33** → 10.36 | **12.96** → 23.49 | 2.00 → 2.00 |
| Rotary B / +2 | 79.0 → **87.9** | 31.3 → **19.5** | 53.72 → **55.73** | 3.07 → **2.97** | 20.24 → **13.82** | 21.58 → **10.82** | 2.36 → **2.00** |
| Rotary B / -2 | 81.9 → **90.3** | 37.4 → **33.0** | **56.89** → 48.72 | **6.35** → 7.28 | 21.03 → **4.76** | 15.84 → **13.58** | 2.00 → 2.00 |
| Transverse A / +2 | **92.1** → 86.3 | **34.4** → 45.8 | **52.52** → 52.01 | 6.88 → **5.37** | **6.74** → 11.82 | **8.87** → 15.12 | **1.19** → 2.01 |
| Transverse A / -2 | 91.7 → **92.6** | **15.0** → 20.0 | **51.92** → 50.86 | **4.92** → 5.78 | 7.14 → **4.76** | **9.26** → 9.45 | 2.00 → **1.00** |
| Transverse B / +2 | 87.1 → **92.3** | **26.5** → 28.2 | 53.15 → **53.19** | **5.56** → 6.54 | 11.50 → **7.72** | 14.01 → **7.69** | **1.03** → 1.20 |
| Transverse B / -2 | 87.4 → **92.7** | 23.0 → **16.2** | 47.20 → **49.04** | 8.80 → **6.96** | 5.24 → **2.38** | 18.32 → **11.11** | 1.12 → **1.00** |

**Effort, clipping, and front/rear load balance**

| Gait / vx command [m/s] | Power [W] ↓* | Normalized torque cost ↓* | Clipping [%] ↓ | Work imbalance [%] ↓† | Force imbalance [%] ↓† | Torque imbalance [%] ↓† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | **143.3** → 159.6 | 0.522 → **0.463** | **2.28** → 2.61 | 13.96 → **6.76** | **10.01** → 11.06 | 20.72 → **20.37** |
| Trot / -2 | N/A → 144.3 | N/A → 0.524 | N/A → 2.74 | N/A → 20.86 | N/A → 31.50 | N/A → 54.24 |
| Bound / +2 | **208.7** → 242.6 | **0.631** → 0.668 | **3.60** → 4.26 | **5.53** → 6.78 | **17.08** → 17.88 | **10.73** → 18.72 |
| Bound / -2 | **223.2** → 336.9 | **0.673** → 0.817 | **2.24** → 5.13 | **4.48** → 35.20 | 39.22 → **38.91** | 39.23 → **30.51** |
| Half-bound A / +2 | **214.0** → 302.6 | **0.674** → 0.758 | **3.23** → 3.43 | **8.33** → 10.27 | 21.01 → **18.09** | **15.88** → 21.46 |
| Half-bound A / -2 | **194.5** → 290.2 | **0.614** → 0.640 | 3.21 → **1.28** | **9.41** → 36.35 | **44.63** → 49.15 | 55.55 → **33.79** |
| Half-bound B / +2 | **153.1** → 217.0 | **0.589** → 0.747 | **1.82** → 2.29 | 14.30 → **0.89** | **17.13** → 22.48 | 25.86 → **23.37** |
| Half-bound B / -2 | N/A → 321.0 | N/A → 0.760 | N/A → 5.13 | N/A → 32.50 | N/A → 40.59 | N/A → 31.24 |
| Rotary A / +2 | **180.0** → 206.3 | **0.537** → 0.672 | 3.85 → **3.57** | 23.95 → **0.59** | **15.72** → 19.70 | 40.39 → **18.58** |
| Rotary A / -2 | **150.0** → 309.3 | **0.643** → 0.773 | **1.92** → 4.18 | 30.54 → **29.46** | 37.34 → **37.28** | 64.16 → **39.71** |
| Rotary B / +2 | **233.8** → 247.3 | **0.622** → 0.645 | 4.16 → **2.42** | 22.81 → **1.41** | **8.99** → 22.22 | 31.46 → **27.04** |
| Rotary B / -2 | **197.4** → 232.7 | 0.659 → **0.595** | 2.57 → **2.29** | **0.78** → 25.95 | **43.92** → 50.31 | **44.62** → 53.38 |
| Transverse A / +2 | 207.6 → **170.8** | 0.592 → **0.528** | **2.11** → 3.57 | 7.05 → **2.44** | 20.23 → **19.03** | **17.25** → 21.79 |
| Transverse A / -2 | 179.9 → **166.2** | 0.601 → **0.570** | **1.60** → 2.56 | **11.39** → 13.84 | **39.08** → 40.29 | **55.35** → 55.75 |
| Transverse B / +2 | **158.1** → 194.9 | **0.590** → 0.667 | 3.22 → **2.79** | **1.23** → 9.33 | **14.74** → 27.25 | **14.18** → 29.40 |
| Transverse B / -2 | 167.9 → **158.4** | 0.632 → **0.543** | 2.24 → 2.24 | 24.34 → **5.58** | **32.26** → 40.96 | 59.82 → **58.50** |

**Classification, success, and completion**

| Gait / vx command [m/s] | Exact template [%] ↑ | Correct family [%] ↑ | Velocity success [%] ↑ | Gait success [%] ↑ | Combined success [%] ↑ | Positive progress [%] ↑ | Completion [%] ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot / +2 | 100.0 → 100.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Trot / -2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → **100.0** |
| Bound / +2 | 0.0 → **97.0** | 0.0 → **97.0** | 0.0 → 0.0 | 0.0 → **97.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Bound / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Half-bound A / +2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound A / -2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Half-bound B / +2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Half-bound B / -2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → **100.0** |
| Rotary A / +2 | **100.0** → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary A / -2 | **100.0** → 35.0 | **100.0** → 35.0 | 0.0 → 0.0 | **100.0** → 35.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 |
| Rotary B / +2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Rotary B / -2 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | **100.0** → 0.0 | 100.0 → 100.0 |
| Transverse A / +2 | **88.0** → 0.0 | **88.0** → 0.0 | 0.0 → 0.0 | **88.0** → 0.0 | 0.0 → 0.0 | **95.0** → 92.0 | 95.0 → **100.0** |
| Transverse A / -2 | 0.0 → **100.0** | 0.0 → **100.0** | 0.0 → 0.0 | 0.0 → **100.0** | 0.0 → 0.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| Transverse B / +2 | **100.0** → 99.0 | **100.0** → 99.0 | 0.0 → 0.0 | **100.0** → 99.0 | 0.0 → 0.0 | **100.0** → 99.0 | **100.0** → 99.0 |
| Transverse B / -2 | 10.0 → **100.0** | 10.0 → **100.0** | 0.0 → 0.0 | 10.0 → **100.0** | 0.0 → 0.0 | 10.0 → **100.0** | 10.0 → **100.0** |

## Source manifests and audit provenance

The ±1 studies contribute 1,600 trials per policy; the new six-speed studies contribute 4,800 per policy. Checkpoint hashes match between the two studies for each policy. Both manifest links are supplied so this report does not depend on temporary `/tmp` analysis files.

| Robot | Method | Commands [m/s] | Manifest | Manifest SHA-256 |
| --- | ---: | ---: | ---: | ---: |
| X1 | No TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `7e940d4610351978d818463e5c95f40afdf1c9931bd57b6625f5fba2778d46ca` |
| X1 | No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `8919400a11e9fb46de58eb67cc6cc2f3a53c45456199c92413147f4697b948f6` |
| X1 | TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `27043c1e350b97c80991f0c504d7ca62f2150e9518a5785c071ae67f99d3ebac` |
| X1 | TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `6394355593d51063ed1f396c670a0aa052e5adcfc284657bfabf80bcc9d4dbd2` |
| X1_go2bend | No TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `1f2f2f6163433b930151d25120b56878990b171e3de87233c53f443b8073ce5d` |
| X1_go2bend | No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_08-45-56_x1_go2bend_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `8ec8954286b92e02dab499ef3d8265581d8088eb7f7193f4a2f18df54d4cfa8c` |
| X1_go2bend | TRS | ±1 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `bc5a07fe81ef46243ef4a3e2d779f57853dc4452910c587bf354fc580e604abd` |
| X1_go2bend | TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/dobot_x1_go2bend_symm_flat/2026-10-21_04-30-03_x1_go2bend_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `e4a111a42211923d515591550e88b7cc19409f05640789ce215dcb4738daa2b0` |
| Go2 | No TRS | ±1 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `1fcda0349bb01b5d88fe36ba93abf6c7992e4ad8b10c179e1b5560629d7d66b7` |
| Go2 | No TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `a379f3b626354d46421308a0b5e68748f6a441a5288572217a9ed1545c75483b` |
| Go2 | TRS | ±1 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) | `5f2ff12adb02e5809fc138d8472d51f3e68826e7c0465519c33171d89b26d8fa` |
| Go2 | TRS | ±0.5, ±1.5, ±2 | [study.json](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000_by_velocity_4475db700d/study.json) | `1a4f70eec0fa5880fa464119e03e3e4a6536b3035e48bae7b3059e3f53b55b15` |

| Robot | Method | Checkpoint SHA-256 |
| --- | ---: | ---: |
| X1 | No TRS | `127eb75070436c60ef8a610022c4fb12032b1b92f96f0445bc5e9be8bf4bd796` |
| X1 | TRS | `b9f9b4e942e03f8866737f943a733e6062b26d4a693e34622616f6d0463ab6d2` |
| X1_go2bend | No TRS | `6cede8964978146ba403d4fa4169189501e96a09607422edb3aca3e04de3d703` |
| X1_go2bend | TRS | `d23818ac5c17b3fbfad88a8195a58c77c604abbe19fac88f3294d15e04a82da8` |
| Go2 | No TRS | `0430dde06881765effd0a1ffaf171dc9e117b3277cbd04fd7a7e652370346f71` |
| Go2 | TRS | `1c437c00cf0ca34b8bdac5d83b3da9489306bd99abb604d0f05e31cfaef63eb9` |

Source CSV locations inside each linked study: `metrics/cell_metrics.csv`; each trial's archive and runtime status live at its manifest `relative_output_dir`. The report was generated from audited comparison CSVs containing 12 speed-pair rows, 24 direction rows, 96 gait-pair rows, and 192 gait/direction rows. Matching of time-reversed speed pairs uses the study's saved time-reversal partner indices.
