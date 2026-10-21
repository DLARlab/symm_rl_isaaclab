<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Go2 and X1 statistical evaluation: no TRS versus actor TRS 0.1

Saved: **2026-10-21**. Training runs: **October 19–20, 2026**. Checkpoint: **model_9999.pt**.

**TRS shows a useful but uneven improvement. The strongest evidence is better forward/backward speed consistency, more balanced actuator usage, and less action clipping. Gait fidelity and directional stability remain dependent on the robot, gait, and travel direction.**

This report saves the four-run statistical comparison, including every gait in both directions, the additional metrics, and the interpretation discussed in chat. It covers the statistical `evaluations/` outputs, separately from the older October 14 playback comparison.

## Main conclusions

- **X1:** average speed RMSE decreases **12.4%**, yaw-rate RMSE decreases **18.8%**, and front/rear work imbalance decreases **49.2%**. Backward bound is a convincing local improvement. Heading retention varies by direction, and neither rotary template consistently produces its requested gait identity.
- **Go2:** average speed RMSE decreases **23.2%**, and front/rear work imbalance decreases **17.8%**. Backward half-bound A improves substantially. Average yaw-rate RMSE increases **14.0%**, and rotary A regresses.
- **Duty tracking improves fairly consistently:** lower error in **11/16 X1** and **13/16 Go2** gait/direction conditions. X1 has two additional ties.
- **Touchdown timing has no consistent improvement:** X1 average error increases **21.2 → 23.2 ms**; Go2 decreases **16.8 → 15.7 ms**. Heading RMSE improves in **8/16 conditions** for each robot.
- **Overall combined success remains unchanged:** **12.5% for X1** and **6.25% for Go2**. Lower tracking errors and more even loading do not establish universally better gait execution.

A defensible statement is:

> Actor time-reversal regularization improves forward/backward speed consistency and reduces actuation imbalance in these experiments. Its effects on gait fidelity and directional stability depend on the robot, gait, and travel direction.

These results do not establish universal gait improvement, smoother motion, greater robustness, or statistical significance across training seeds.

## Runs and evaluation protocol

| Robot | Method | Run / study manifest |
| --- | ---: | ---: |
| X1 | No TRS | [2026-10-19_05-22-27_x1_no_trs](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_05-22-27_x1_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) |
| X1 | Actor TRS 0.1 | [2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/dobot_x1_symm_flat/2026-10-19_21-33-11_x1_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) |
| GO2 | No TRS | [2026-10-19_09-33-24_go2_no_trs](logs/rsl_rl/unitree_go2_symm_flat/2026-10-19_09-33-24_go2_no_trs/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) |
| GO2 | Actor TRS 0.1 | [2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1](logs/rsl_rl/unitree_go2_symm_flat/2026-10-20_04-49-06_go2_with_trs_mirror0p1_valuecoef0_actorcoef0.1/evaluations/leg_usage_grid_trained_envs100_steps1000/study.json) |

- Eight trained gait templates; commands **vx = +1 and −1 m/s**, with **vy = 0** and **yaw rate = 0**. This comparison does not test nonzero lateral or yaw commands.
- **100 parallel nominal trials per gait/command**, giving **1,600 trials per run** and **6,400 total**. All recorded trials complete without termination.
- Simulation/control step **0.02 s**; **250 settling steps / 5 s**, then **1,000 measured steps / 20 s**. Each full rollout has **1,250 steps / 25 s**.
- Metrics use complete-cycle windows within the measurement interval: **960 samples / 19.2 s / 32 cycles for X1**, **988 samples / 19.76 s / 38 cycles for Go2**.
- Fixed cycle duration: **X1 0.60 s**, **Go2 0.52 s**. Commanded stance duty factor: **0.56** for both.
- Within each robot, saved environment configurations match except for log-directory fields. Both methods use training seed **42**, **2,048 training environments**, and **24 rollout steps**.
- The TRS condition uses **actor coefficient 0.1**, **critic coefficient 0**, and **warmup 0**. The run-name/config field `mirror_loss_coeff` is the actor TRS coefficient in this implementation; it does not indicate a separate spatial mirror-loss intervention.
- Per-gait averages weight forward/backward equally. Overall averages weight all eight templates equally. These overall values can differ from family-balanced averages in the original reports.

The replicas are nominal evaluations of a fixed trained policy, not independent training runs or randomized robustness trials. X1 replicas repeat the same trajectories; Go2 shows small numerical variation. Training-seed uncertainty is not measured here.

## Reading the tables

All paired values are **No TRS → TRS**. **Bold** marks the preferred displayed value. Ties at the displayed precision are left unbolded.

- **↓:** lower error or fewer constraint interventions is preferable.
- **↑:** higher contact agreement, classifier match rate, or success rate is preferable.
- **↓†:** lower imbalance means more even front/rear loading. Equal loading is not necessarily optimal for every gait or robot.
- **↓\*:** lower mechanical effort is preferable only when tracking, gait execution, and displacement are comparable.
- **≈100%:** directed progress ideally equals the requested distance; bold marks the value closest to 100%. Negative progress means net displacement opposite the requested world direction.
- **F/B:** forward/backward, at vx = +1/−1 m/s. Half-bound and gallop A/B labels below identify the saved phase templates, not a claim that the observed gait matches that template.

## Overall metrics

| Metric | X1: No TRS → TRS | Go2: No TRS → TRS |
| --- | ---: | ---: |
| Forward-speed RMSE [m/s] ↓ | 0.085 → **0.075** | 0.083 → **0.064** |
| Forward/backward paired speed bias [m/s] ↓ | 0.083 → **0.050** | 0.087 → **0.041** |
| Yaw-rate RMSE [rad/s] ↓ | 0.079 → **0.064** | **0.073** → 0.083 |
| Heading RMSE [degrees] ↓ | 43.6 → **39.7** | 40.1 → **38.6** |
| Contact agreement [%] ↑ | 92.43 → **93.17** | 93.38 → **94.14** |
| Touchdown timing error [ms] ↓ | **21.2** → 23.2 | 16.8 → **15.7** |
| Duty error [percentage points] ↓ | 4.98 → **4.49** | 5.48 → **4.49** |
| Front/rear work imbalance [%] ↓† | 7.48 → **3.80** | 15.38 → **12.64** |
| Front/rear vertical-force impulse imbalance [%] ↓† | 5.43 → **2.91** | 27.27 → **26.46** |
| Front/rear normalized torque imbalance [%] ↓† | 14.24 → **8.12** | 39.57 → **35.40** |
| Absolute mechanical power [W] ↓\* | 134.3 → **124.2** | 103.0 → **100.1** |
| Normalized squared-torque cost per second ↓\* | 0.819 → **0.758** | 0.382 → **0.372** |
| Action clipping [%] ↓ | 0.625 → **0.064** | 1.349 → **0.300** |
| Near torque-limit joint samples [%] ↓\* | 0.447 → **0.365** | 0.020 → **0.000** |
| Correct gait family [%] ↑ | 75.00 → 75.00 | 68.75 → **75.00** |
| Combined success [%] ↑ | 12.50 → 12.50 | 6.25 → 6.25 |

Forward/backward paired speed bias decreases **39.1% for X1** and **52.8% for Go2**. Action clipping decreases **89.8%** and **77.7%**, respectively. Correct-family counts are **1,200 → 1,200 / 1,600 for X1** and **1,100 → 1,200 / 1,600 for Go2**. Combined-success counts are **200 → 200** and **100 → 100**, respectively.

Lower absolute mechanical power is descriptive here. It is not an electrical-energy measurement, and different heading/progress outcomes prevent an unconditional efficiency claim.

## X1: per-gait averages

| Gait | vx RMSE ↓ [m/s] | Yaw-rate RMSE ↓ [rad/s] | Contact agreement ↑ [%] | Work imbalance ↓† [%] |
| --- | ---: | ---: | ---: | ---: |
| Trot | 0.071 → **0.067** | **0.051** → 0.072 | 95.1 → **96.4** | **7.79** → 10.36 |
| Bound | 0.092 → **0.081** | 0.094 → **0.050** | 95.5 → **96.2** | 5.24 → **2.28** |
| Half-bound A | **0.064** → 0.069 | 0.076 → **0.063** | **95.1** → 93.2 | **3.25** → 4.05 |
| Half-bound B | 0.122 → **0.086** | 0.096 → **0.056** | 94.0 → **96.0** | 4.89 → **1.73** |
| Rotary A | **0.077** → 0.080 | 0.092 → **0.074** | 81.0 → **86.0** | 7.78 → **2.02** |
| Rotary B | 0.091 → **0.074** | 0.104 → **0.069** | **86.0** → 84.4 | 15.38 → **0.56** |
| Transverse A | 0.079 → **0.071** | **0.053** → 0.070 | **97.7** → 96.0 | 9.65 → **6.79** |
| Transverse B | 0.086 → **0.070** | 0.069 → **0.062** | 95.1 → **97.1** | 5.88 → **2.62** |

### X1: tracking in both directions

| Gait / direction | vx RMSE ↓ [m/s] | Yaw-rate RMSE ↓ [rad/s] | Contact agreement ↑ [%] | Directed progress ≈100% |
| --- | ---: | ---: | ---: | ---: |
| Trot F | **0.056** → 0.074 | **0.060** → 0.102 | 94.6 → **97.3** | **90.5** → 28.6 |
| Trot B | 0.086 → **0.060** | **0.041** → 0.043 | 95.5 → 95.5 | 90.4 → **96.4** |
| Bound F | **0.069** → 0.101 | 0.079 → **0.044** | 93.8 → **95.8** | **69.9** → 63.9 |
| Bound B | 0.116 → **0.060** | 0.109 → **0.055** | **97.3** → 96.6 | -27.2 → **63.3** |
| Half-bound A F | **0.054** → 0.081 | **0.059** → 0.073 | **94.6** → 92.9 | **95.8** → 82.1 |
| Half-bound A B | 0.074 → **0.057** | 0.093 → **0.054** | **95.5** → 93.5 | 34.9 → **89.5** |
| Half-bound B F | **0.056** → 0.101 | 0.132 → **0.054** | 93.8 → **95.5** | -2.0 → **87.4** |
| Half-bound B B | 0.189 → **0.071** | 0.060 → **0.058** | 94.2 → **96.4** | **77.6** → 41.1 |
| Rotary A F | **0.076** → 0.102 | 0.105 → **0.074** | 80.8 → **85.4** | -32.7 → **86.6** |
| Rotary A B | 0.078 → **0.058** | 0.079 → **0.073** | 81.2 → **86.6** | **78.5** → -19.6 |
| Rotary B F | 0.087 → **0.075** | 0.122 → **0.062** | **86.7** → 84.9 | **97.1** → 91.7 |
| Rotary B B | 0.095 → **0.073** | 0.086 → **0.075** | **85.4** → 83.9 | **86.9** → 74.5 |
| Transverse A F | 0.083 → **0.075** | **0.051** → 0.058 | **97.3** → 96.4 | **89.5** → 78.8 |
| Transverse A B | 0.075 → **0.067** | **0.054** → 0.082 | **98.0** → 95.5 | **88.4** → 51.7 |
| Transverse B F | 0.075 → **0.073** | 0.092 → **0.064** | 93.8 → **97.3** | 60.9 → **73.7** |
| Transverse B B | 0.097 → **0.068** | **0.047** → 0.059 | 96.4 → **96.9** | **83.9** → 69.3 |

### X1: timing, heading, and actuator constraints in both directions

| Gait / direction | Heading ↓ [degrees] | Touchdown ↓ [ms] | Duty error ↓ [pp] | Work imbalance ↓† [%] | Near torque limit ↓\* [%] | Clipping ↓ [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot F | **26.5** → 77.9 | **7.0** → 20.3 | 5.2 → 5.2 | **5.1** → 9.6 | 0.00 → 0.00 | 3.06 → **0.00** |
| Trot B | 10.4 → **2.2** | 5.5 → **5.1** | 4.3 → 4.3 | **10.5** → 11.1 | 0.00 → 0.00 | 0.00 → 0.00 |
| Bound F | 44.9 → **42.2** | **15.0** → 18.0 | 6.8 → **5.7** | 6.2 → **0.2** | 1.11 → **0.75** | 0.00 → 0.00 |
| Bound B | 124.2 → **47.9** | **10.4** → 23.3 | **3.5** → 5.8 | 4.3 → 4.3 | **0.28** → 1.12 | 0.00 → 0.00 |
| Half-bound A F | **10.4** → 25.9 | **9.0** → 19.0 | 5.6 → **3.7** | **5.6** → 5.9 | 0.83 → **0.28** | 0.83 → **0.00** |
| Half-bound A B | 72.2 → **23.7** | **19.0** → 21.3 | **3.3** → 4.9 | **0.9** → 2.2 | 0.28 → **0.08** | **0.00** → 0.19 |
| Half-bound B F | 101.0 → **4.4** | **12.9** → 18.2 | 6.3 → **4.3** | **2.8** → 3.1 | **0.30** → 1.11 | 0.83 → **0.00** |
| Half-bound B B | **3.8** → 64.6 | 29.0 → **24.0** | 4.2 → **3.8** | 7.0 → **0.4** | 0.00 → 0.00 | 0.00 → 0.00 |
| Rotary A F | 125.8 → **10.1** | 55.7 → **44.7** | 6.3 → **4.6** | 4.8 → **3.4** | **0.83** → 1.67 | 0.00 → 0.00 |
| Rotary A B | **33.3** → 113.2 | 40.5 → **38.0** | 4.2 → **3.3** | 10.8 → **0.6** | 0.56 → **0.00** | 0.28 → **0.00** |
| Rotary B F | 23.4 → **6.6** | **35.9** → 47.8 | 5.8 → **3.9** | 26.0 → **1.0** | 1.72 → **0.28** | 0.83 → **0.00** |
| Rotary B B | **14.3** → 38.6 | 48.0 → **41.3** | 4.7 → **3.3** | 4.8 → **0.1** | **0.12** → 0.56 | 0.00 → 0.00 |
| Transverse A F | **12.8** → 32.6 | 12.0 → **11.5** | 4.7 → **4.6** | 11.4 → **3.8** | 0.00 → 0.00 | 1.11 → **0.00** |
| Transverse A B | **19.6** → 62.9 | **11.0** → 15.0 | **3.2** → 6.0 | **7.9** → 9.8 | 0.56 → **0.00** | **0.28** → 0.83 |
| Transverse B F | 52.8 → **40.0** | 17.0 → **14.0** | 6.8 → **4.3** | 4.3 → **1.0** | 0.28 → **0.00** | 2.78 → **0.00** |
| Transverse B B | **22.0** → 42.3 | 11.0 → **10.0** | 4.7 → **3.9** | 7.5 → **4.2** | 0.28 → **0.00** | 0.00 → 0.00 |

### X1: load and effort in both directions

| Gait / direction | Force-impulse imbalance ↓† [%] | Normalized torque imbalance ↓† [%] | Absolute mechanical power ↓\* [W] | Normalized torque cost ↓\* |
| --- | ---: | ---: | ---: | ---: |
| Trot F | 4.3 → **0.8** | 12.1 → **2.6** | 89.2 → **74.0** | **0.722** → 0.742 |
| Trot B | **2.5** → 4.1 | 2.7 → **1.3** | 79.3 → **72.7** | **0.750** → 0.768 |
| Bound F | 4.6 → **4.4** | 19.5 → **16.6** | **143.8** → 145.6 | 0.849 → **0.705** |
| Bound B | **0.1** → 3.9 | 17.6 → **1.4** | 152.2 → **144.2** | 0.802 → **0.713** |
| Half-bound A F | 5.9 → **2.7** | 18.0 → **0.8** | **127.8** → 130.8 | 0.830 → **0.701** |
| Half-bound A B | 2.4 → **1.8** | 7.0 → **0.3** | 148.8 → **146.0** | **0.773** → 0.825 |
| Half-bound B F | 11.7 → **7.2** | **5.6** → 14.9 | 158.0 → **125.0** | 0.842 → **0.770** |
| Half-bound B B | 5.0 → **2.3** | 14.3 → **1.1** | **134.5** → 141.4 | 0.795 → **0.715** |
| Rotary A F | 8.6 → **7.1** | 24.7 → **21.7** | 157.5 → **125.5** | 0.880 → **0.807** |
| Rotary A B | 6.9 → **1.7** | 19.8 → **11.9** | 156.2 → **139.8** | 0.791 → **0.676** |
| Rotary B F | 14.9 → **3.3** | 30.7 → **27.9** | 183.1 → **133.2** | 0.976 → **0.745** |
| Rotary B B | 4.2 → **1.0** | **5.2** → 6.3 | 144.8 → **127.5** | 0.758 → **0.711** |
| Transverse A F | 3.6 → **1.6** | 14.1 → **4.6** | **108.0** → 117.6 | 0.780 → **0.756** |
| Transverse A B | 4.2 → **0.4** | 14.3 → **1.8** | **120.7** → 121.9 | 0.809 → **0.788** |
| Transverse B F | 4.7 → **3.1** | 13.7 → **9.3** | 130.3 → **115.4** | 0.923 → **0.823** |
| Transverse B B | 3.2 → **1.3** | 8.5 → **7.3** | **113.8** → 126.5 | **0.819** → 0.880 |

### X1: observed gait classification and combined success

Classifier labels below are the observed contact-pattern labels. An unclassified result is ambiguous under the configured template thresholds. A correct label does not establish perfect contact or trajectory execution.

| Requested gait / direction | Classified template: No TRS → TRS | Correct family ↑ [%] | Combined success ↑ [%] |
| --- | ---: | ---: | ---: |
| Trot F | Trot (100/100) → Trot (100/100) | 100 → 100 | 0 → 0 |
| Trot B | Trot (100/100) → Trot (100/100) | 100 → 100 | 100 → 100 |
| Bound F | Bound (100/100) → Bound (100/100) | 100 → 100 | 0 → **100** |
| Bound B | Bound (100/100) → Bound (100/100) | 100 → 100 | 0 → 0 |
| Half-bound A F | Half-bound A (100/100) → Half-bound A (100/100) | 100 → 100 | 0 → 0 |
| Half-bound A B | Half-bound A (100/100) → Half-bound A (100/100) | 100 → 100 | 0 → 0 |
| Half-bound B F | Half-bound B (100/100) → Half-bound B (100/100) | 100 → 100 | 0 → 0 |
| Half-bound B B | Half-bound B (100/100) → Half-bound B (100/100) | 100 → 100 | 0 → 0 |
| Rotary A F | Half-bound hind A (100/100) → Half-bound B (100/100) | 0 → 0 | 0 → 0 |
| Rotary A B | Half-bound hind A (100/100) → Half-bound B (100/100) | 0 → 0 | 0 → 0 |
| Rotary B F | Half-bound A (100/100) → Unclassified (100/100) | 0 → 0 | 0 → 0 |
| Rotary B B | Half-bound A (100/100) → Half-bound hind B (100/100) | 0 → 0 | 0 → 0 |
| Transverse A F | Transverse A (100/100) → Transverse A (100/100) | 100 → 100 | 0 → 0 |
| Transverse A B | Transverse A (100/100) → Transverse A (100/100) | 100 → 100 | 0 → 0 |
| Transverse B F | Transverse B (100/100) → Transverse B (100/100) | 100 → 100 | 0 → 0 |
| Transverse B B | Transverse B (100/100) → Transverse B (100/100) | 100 → 100 | **100** → 0 |

## GO2: per-gait averages

| Gait | vx RMSE ↓ [m/s] | Yaw-rate RMSE ↓ [rad/s] | Contact agreement ↑ [%] | Work imbalance ↓† [%] |
| --- | ---: | ---: | ---: | ---: |
| Trot | 0.039 → **0.037** | 0.082 → **0.066** | 96.4 → **97.4** | **17.28** → 19.88 |
| Bound | 0.119 → **0.097** | **0.065** → 0.084 | **89.5** → 88.0 | 9.11 → **2.48** |
| Half-bound A | 0.096 → **0.065** | 0.074 → **0.067** | 94.7 → **96.3** | 18.52 → **3.52** |
| Half-bound B | 0.081 → **0.076** | 0.094 → **0.075** | 91.9 → **94.3** | 13.91 → **13.52** |
| Rotary A | 0.091 → **0.078** | **0.049** → 0.104 | 92.2 → **94.3** | 21.42 → **12.77** |
| Rotary B | 0.106 → **0.055** | **0.082** → 0.096 | 93.4 → **94.3** | **13.07** → 14.82 |
| Transverse A | 0.067 → **0.059** | **0.067** → 0.100 | **96.3** → 94.8 | 16.49 → **12.52** |
| Transverse B | 0.066 → **0.044** | **0.072** → 0.076 | 92.7 → **93.8** | **13.20** → 21.65 |

### GO2: tracking in both directions

| Gait / direction | vx RMSE ↓ [m/s] | Yaw-rate RMSE ↓ [rad/s] | Contact agreement ↑ [%] | Directed progress ≈100% |
| --- | ---: | ---: | ---: | ---: |
| Trot F | 0.048 → **0.033** | 0.096 → **0.082** | 97.9 → **100.0** | **90.7** → 75.9 |
| Trot B | **0.031** → 0.042 | 0.068 → **0.050** | 94.8 → 94.8 | 74.8 → **92.5** |
| Bound F | **0.078** → 0.081 | 0.078 → **0.070** | **87.3** → 84.4 | 62.3 → **87.9** |
| Bound B | 0.161 → **0.112** | **0.052** → 0.098 | **91.7** → 91.6 | **83.0** → 51.4 |
| Half-bound A F | 0.092 → **0.064** | **0.046** → 0.079 | 94.6 → **96.9** | **104.2** → 50.0 |
| Half-bound A B | 0.099 → **0.067** | 0.101 → **0.055** | 94.8 → **95.8** | 33.1 → **94.3** |
| Half-bound B F | **0.078** → 0.090 | 0.134 → **0.076** | 90.0 → **94.8** | 8.3 → **62.9** |
| Half-bound B B | 0.084 → **0.062** | **0.054** → 0.073 | 93.8 → 93.8 | 47.8 → **74.1** |
| Rotary A F | **0.059** → 0.086 | **0.051** → 0.114 | 91.7 → **94.8** | **101.7** → 77.5 |
| Rotary A B | 0.123 → **0.070** | **0.047** → 0.094 | 92.7 → **93.8** | **21.5** → -21.4 |
| Rotary B F | 0.068 → **0.050** | 0.081 → **0.065** | 93.1 → **93.8** | 90.1 → **100.1** |
| Rotary B B | 0.143 → **0.059** | **0.084** → 0.126 | 93.7 → **94.8** | 84.3 → **98.6** |
| Transverse A F | 0.073 → **0.070** | **0.071** → 0.108 | **96.9** → 94.8 | **89.9** → 87.1 |
| Transverse A B | 0.060 → **0.048** | **0.063** → 0.092 | **95.6** → 94.8 | **86.8** → 52.3 |
| Transverse B F | 0.070 → **0.055** | **0.054** → 0.087 | 91.7 → **96.9** | **79.9** → 75.3 |
| Transverse B B | 0.061 → **0.032** | 0.089 → **0.064** | **93.8** → 90.6 | 34.3 → **93.3** |

### GO2: timing, heading, and actuator constraints in both directions

| Gait / direction | Heading ↓ [degrees] | Touchdown ↓ [ms] | Duty error ↓ [pp] | Work imbalance ↓† [%] | Near torque limit ↓\* [%] | Clipping ↓ [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trot F | **28.7** → 43.5 | 10.0 → 10.0 | 3.8 → **1.9** | **7.3** → 16.0 | 0.00 → 0.00 | 0.00 → 0.00 |
| Trot B | 43.2 → **16.9** | 10.1 → **2.1** | 5.0 → **3.5** | 27.2 → **23.8** | 0.00 → 0.00 | 0.00 → 0.00 |
| Bound F | 53.2 → **34.1** | **23.5** → 26.2 | **5.6** → 7.7 | 9.0 → **3.8** | 0.00 → 0.00 | 1.07 → **0.00** |
| Bound B | **5.2** → 56.7 | **9.9** → 14.8 | **4.8** → 5.8 | 9.2 → **1.2** | 0.00 → 0.00 | 0.96 → **0.64** |
| Half-bound A F | **4.0** → 60.2 | 26.2 → **11.2** | 5.2 → **3.7** | 23.4 → **3.8** | 0.00 → 0.00 | 1.92 → **0.00** |
| Half-bound A B | 70.4 → **0.9** | 13.8 → **10.0** | 6.7 → **2.7** | 13.6 → **3.2** | 0.00 → 0.00 | 1.60 → **0.00** |
| Half-bound B F | 90.5 → **52.0** | 36.2 → **26.2** | 8.2 → **6.8** | **12.4** → 26.2 | 0.00 → 0.00 | 2.25 → **0.32** |
| Half-bound B B | 63.0 → **41.2** | **10.0** → 15.0 | 6.7 → **3.8** | 15.5 → **0.9** | 0.00 → 0.00 | 1.92 → **0.96** |
| Rotary A F | **4.5** → 39.0 | **5.6** → 15.6 | **3.8** → 4.8 | **19.9** → 24.3 | 0.00 → 0.00 | 1.60 → **0.32** |
| Rotary A B | **76.9** → 114.5 | 18.8 → 18.8 | 5.8 → **4.0** | 22.9 → **1.3** | 0.00 → 0.00 | 1.92 → **0.64** |
| Rotary B F | 28.8 → **1.3** | 10.6 → **10.0** | 3.5 → **2.9** | 16.6 → **14.6** | 0.00 → 0.00 | 0.96 → **0.00** |
| Rotary B B | **15.6** → 17.8 | 22.6 → **12.9** | 4.8 → 4.8 | **9.6** → 15.1 | 0.00 → 0.00 | 1.28 → **0.00** |
| Transverse A F | 23.6 → **23.4** | 26.8 → **22.4** | 5.9 → **3.8** | 20.2 → **17.8** | 0.00 → 0.00 | 1.28 → **0.96** |
| Transverse A B | **23.7** → 58.1 | 16.2 → **11.2** | 5.1 → **4.9** | 12.8 → **7.2** | 0.32 → **0.00** | 1.28 → **0.64** |
| Transverse B F | **35.7** → 39.4 | **17.4** → 21.8 | 6.7 → **4.8** | **17.3** → 29.4 | 0.00 → 0.00 | 1.92 → **0.32** |
| Transverse B B | 74.0 → **18.6** | **11.2** → 22.4 | 5.9 → **5.8** | **9.1** → 13.9 | 0.00 → 0.00 | 1.60 → **0.00** |

### GO2: load and effort in both directions

| Gait / direction | Force-impulse imbalance ↓† [%] | Normalized torque imbalance ↓† [%] | Absolute mechanical power ↓\* [W] | Normalized torque cost ↓\* |
| --- | ---: | ---: | ---: | ---: |
| Trot F | **14.1** → 16.7 | **15.7** → 24.4 | 66.4 → **62.0** | 0.364 → **0.322** |
| Trot B | 30.2 → **19.6** | 52.2 → **30.6** | **58.0** → 59.0 | 0.358 → **0.325** |
| Bound F | **23.1** → 24.2 | **20.4** → 24.9 | 122.9 → **114.2** | 0.436 → **0.397** |
| Bound B | 29.4 → **25.5** | 35.4 → **22.3** | 134.6 → **131.9** | **0.385** → 0.422 |
| Half-bound A F | 25.0 → **24.2** | 36.2 → **29.7** | 120.0 → **113.2** | 0.450 → **0.358** |
| Half-bound A B | 34.7 → **30.9** | 50.4 → **39.3** | 126.3 → **116.7** | 0.383 → **0.375** |
| Half-bound B F | **27.7** → 31.2 | **38.4** → 39.3 | **103.8** → 106.2 | **0.408** → 0.429 |
| Half-bound B B | 32.9 → **29.8** | 46.1 → **29.4** | **119.6** → 122.5 | **0.347** → 0.387 |
| Rotary A F | **25.3** → 26.2 | 50.7 → **33.9** | **90.5** → 99.0 | **0.358** → 0.373 |
| Rotary A B | **30.1** → 30.9 | 49.1 → **48.3** | 118.7 → **118.4** | 0.375 → **0.367** |
| Rotary B F | 25.4 → **23.4** | 42.4 → **40.5** | **91.8** → 107.7 | 0.355 → **0.338** |
| Rotary B B | 33.4 → **32.6** | **48.7** → 55.1 | 123.4 → **106.9** | 0.381 → **0.370** |
| Transverse A F | **21.1** → 21.4 | 28.9 → **28.3** | 83.3 → **79.3** | 0.377 → **0.356** |
| Transverse A B | 33.1 → **26.9** | 53.5 → **37.7** | 114.1 → **97.8** | 0.414 → **0.372** |
| Transverse B F | **22.7** → 26.8 | **25.7** → 38.3 | 80.0 → **79.3** | **0.372** → 0.384 |
| Transverse B B | **28.1** → 33.1 | **39.3** → 44.6 | 95.1 → **87.4** | **0.353** → 0.381 |

### GO2: observed gait classification and combined success

Classifier labels below are the observed contact-pattern labels. An unclassified result is ambiguous under the configured template thresholds. A correct label does not establish perfect contact or trajectory execution.

| Requested gait / direction | Classified template: No TRS → TRS | Correct family ↑ [%] | Combined success ↑ [%] |
| --- | ---: | ---: | ---: |
| Trot F | Trot (100/100) → Trot (100/100) | 100 → 100 | 0 → 0 |
| Trot B | Trot (100/100) → Trot (100/100) | 100 → 100 | 0 → **100** |
| Bound F | Half-bound A (100/100) → Unclassified (100/100) | 0 → 0 | 0 → 0 |
| Bound B | Bound (100/100) → Bound (100/100) | 100 → 100 | 0 → 0 |
| Half-bound A F | Unclassified (100/100) → Half-bound A (100/100) | 0 → **100** | 0 → 0 |
| Half-bound A B | Half-bound A (100/100) → Half-bound A (100/100) | 100 → 100 | 0 → 0 |
| Half-bound B F | Transverse B (100/100) → Unclassified (100/100) | 0 → 0 | 0 → 0 |
| Half-bound B B | Unclassified (100/100) → Half-bound B (100/100) | 0 → **100** | 0 → 0 |
| Rotary A F | Rotary A (100/100) → Half-bound B (100/100) | **100** → 0 | 0 → 0 |
| Rotary A B | Rotary A (100/100) → Rotary A (100/100) | 100 → 100 | **100** → 0 |
| Rotary B F | Rotary B (100/100) → Rotary B (100/100) | 100 → 100 | 0 → 0 |
| Rotary B B | Unclassified (100/100) → Unclassified (100/100) | 0 → 0 | 0 → 0 |
| Transverse A F | Transverse A (100/100) → Transverse A (100/100) | 100 → 100 | 0 → 0 |
| Transverse A B | Transverse A (100/100) → Transverse A (100/100) | 100 → 100 | 0 → 0 |
| Transverse B F | Transverse B (100/100) → Transverse B (100/100) | 100 → 100 | 0 → 0 |
| Transverse B B | Transverse B (100/100) → Transverse B (100/100) | 100 → 100 | 0 → 0 |

## Interpretation by gait

### X1

- **Trot:** backward tracking and heading retention improve. Forward yaw/heading retention and touchdown timing worsen. Front/rear work balance worsens in both directions despite less clipping.
- **Bound:** backward speed, yaw, heading, and directed progress improve substantially. Forward speed error increases, while yaw tracking and work balance improve. Touchdown timing worsens in both directions. Both directions retain the bound classification.
- **Half-bound A:** backward speed, yaw, heading, and progress improve; contact agreement and duty tracking worsen. Forward speed, yaw, heading, and touchdown timing worsen despite better duty tracking and less clipping.
- **Half-bound B:** forward yaw, heading, and directed progress improve strongly, although speed and touchdown errors increase. Backward speed, duty tracking, and work balance improve, but heading retention and directed progress worsen.
- **Rotary A:** forward heading and world progress recover strongly; backward heading and world progress regress, with TRS producing negative backward directed progress. Both methods are classified as half-bound variants rather than the requested rotary gait.
- **Rotary B:** tracking errors and work imbalance generally decrease. Contact agreement worsens in both directions, and the requested rotary identity is not recovered.
- **Transverse A:** speed error and clipping improve forward; backward yaw/heading retention, contact agreement, duty tracking, and work balance worsen. The requested transverse identity is retained.
- **Transverse B:** speed error, contact agreement, duty tracking, and work balance improve in both directions. Forward yaw/heading retention improves; backward yaw/heading retention and directed progress worsen.

### Go2

- **Trot:** forward speed and contact agreement improve, while heading retention and front/rear work balance worsen. Backward yaw, heading, touchdown timing, and duty tracking improve, although speed error increases slightly.
- **Bound:** front/rear work balance improves in both directions. Forward heading/progress improve, but contact agreement and touchdown/duty tracking worsen. Backward speed improves while yaw/heading retention and progress worsen. Forward TRS is unclassified; repeated front-right contacts prevent interpreting improved pairwise timing as clean synchronized bound.
- **Half-bound A:** backward improves across speed, yaw, heading, progress, duty tracking, and work balance. Forward speed, duty, touchdown timing, work balance, and classification improve, but yaw/heading retention and directed progress worsen.
- **Half-bound B:** backward speed, heading, duty tracking, progress, and classification improve while yaw/touchdown timing worsen. Forward yaw/heading, agreement, duty, touchdown timing, and progress improve, but speed and work balance worsen. The forward classifier changes from transverse B to unclassified.
- **Rotary A:** a clear regression in directional control. Forward speed/yaw/heading and touchdown timing worsen, and the classifier changes to half-bound. Backward speed improves, but yaw/heading and world progress worsen, including negative directed progress.
- **Rotary B:** forward shows broad gains in tracking, heading, timing, duty, and progress. Backward speed, touchdown timing, and progress improve; yaw tracking worsens, and the pattern remains unclassified.
- **Transverse A:** speed error, touchdown/duty tracking, and work balance improve in both directions. Yaw error increases in both directions, and backward heading/progress deteriorate.
- **Transverse B:** backward speed, yaw, heading, and progress improve strongly, while contact agreement, touchdown timing, and work balance worsen. Forward agreement and duty improve, but yaw/heading, touchdown timing, work balance, and progress worsen.

## Metric definitions and limits

| Metric | Definition and interpretation |
| --- | --- |
| vx RMSE | Root mean square of body-frame x velocity minus commanded vx [m/s]. |
| Yaw-rate RMSE | Root mean square of measured yaw rate minus its zero command [rad/s]. This differs from accumulated heading drift. |
| Heading RMSE | Root mean square of wrapped heading error relative to the nominal reset heading of zero; converted from radians to degrees for presentation. |
| Forward/backward paired speed bias | Mean absolute sum of forward mean vx for template g and backward mean vx for its time-reversed partner TR(g): abs(mean_vx(g,+1) + mean_vx(TR(g),−1)) [m/s]. It measures paired speed symmetry, not full action/trajectory symmetry. |
| Directed progress | Requested-sign world-x displacement divided by commanded distance over the measurement window. Ideal 100%; negative values indicate opposite net world displacement. Body-frame speed tracking can remain good while heading drift makes world progress poor. |
| Contact agreement | Fraction of measured contacts matching commanded stance, excluding samples within 0.02 cycles of stance/swing boundaries. Uses recorded signed per-foot phases. |
| Touchdown timing error | Mean absolute matched touchdown timing error relative to the commanded gait clock [ms]. It is not the timing difference between two feet and does not alone reject extra contacts or an incorrect gait identity. |
| Duty error | Mean absolute difference between each foot's measured contact fraction and commanded duty 0.56; multiplied by 100 to give percentage points. |
| Work imbalance | 100 × abs(W_front − W_rear) / (W_front + W_rear), using integrated absolute joint mechanical power. |
| Force-impulse imbalance | The same front/rear imbalance formula applied to integrated positive vertical ground force. |
| Normalized torque imbalance | The same front/rear imbalance formula applied to integrated squared torque normalized by each joint's configured effort limit. |
| Absolute mechanical power | Time average of sum over joints of abs(torque × joint velocity) [W]. This is absolute mechanical effort, not measured electrical power. |
| Normalized torque cost | Time average of sum over joints of (torque / configured effort limit)^2. Dimensionless after division by duration. |
| Near torque-limit samples | Mean fraction of joint-time samples with abs(torque) / configured effort limit >= 0.95, expressed as a percentage. |
| Action clipping | Mean fraction of joint commands clipped by target constraints, expressed as a percentage. Less clipping does not establish smooth motion. |
| Classified template | Contact-pattern classifier using pairwise circular touchdown-phase differences; maximum mean error 0.12 cycles, minimum winning margin 0.01 cycles. |
| Correct gait family | Fraction whose classified family matches the requested family. This is broader than exact template agreement. |
| Combined success | Original velocity, progress, termination, event-coverage, family, and contact-agreement checks, with only the contact-agreement predicate corrected for signed phases. |

Combined success requires relative vx RMSE <= 0.2, yaw-rate RMSE <= 0.05 rad/s, positive directed progress, no termination, contact agreement >= 0.8, the correct family, at least two complete cycles, and at least two touchdown events per foot. It does not require small accumulated heading drift or near-100% directed progress. Consequently, a passing trial can still follow a poor world trajectory.

## Evaluation corrections and remaining limitations

The report uses a read-only audit of recorded data. The original evaluation source code and original metrics files were not changed for this comparison.

1. **Signed backward gait clock.** The native command advances phase with the sign of vx, but the original metrics used an increasing common clock without the backward transformation. This artificially reduced backward contact agreement. Agreement here uses archived `foot_periodicity_phases` directly. Backward matched touchdown timing uses the equivalent fixed-duty increasing-clock offset theta + (1 − duty). Recorded phases match the signed analytic formula to about 0.00002 cycles.
2. **Reset reference carryover.** The recorder captured a stale cached heading/position reference after reset, sometimes carrying the previous scenario's final pose. Heading RMSE here uses the nominal reset heading zero. Any audited lateral displacement uses the first post-reset position; its tiny first-step displacement cannot be reconstructed exactly. Directed progress uses measured-window displacement, so a constant reference offset cancels.
3. **Pairwise synchronization across cycle boundaries.** Same-phase event timing can be distorted when events are restricted to the same integer cycle. Audited pairwise timing matches events across boundaries within half a period and retains event counts. Extra contacts can still make a small nearest-event lag misleading. The Go2 TRS forward-bound front-right foot has 76 touchdowns over 38 complete cycles, so its synchronization result does not establish a clean bound.
4. **Success/classification scope.** Corrected combined success changes only the phase-agreement check; other existing checks and the template classifier are retained. A classifier match or a success flag is not a complete measure of gait quality.
5. **One training seed and nominal replicas.** There is one trained policy per method/robot. These results support descriptive comparisons, not multi-seed statistical significance, randomized robustness, or generalization to lateral/yaw commands.
6. **No smoothness claim.** Lower clipping, lower work, or more even loading does not directly measure joint jerk, torque-rate smoothness, or visual gait quality. Abrupt-motion judgments are excluded from the conclusions.

## Source provenance

Each linked study directory contains `study.json`, per-trial `sim_data.npz` archives, and `metrics/cell_metrics.csv`. All four source study hashes were checked against the audited dataset when this report was saved; the source metric files also predate the audit.

The following hashes identify the manifests and checkpoints used. This report embeds its result tables and does not depend on temporary analysis files to display them.

| Condition | study.json SHA-256 | model_9999.pt SHA-256 |
| --- | ---: | ---: |
| X1 / No TRS | `7e940d4610351978d818463e5c95f40afdf1c9931bd57b6625f5fba2778d46ca` | `127eb75070436c60ef8a610022c4fb12032b1b92f96f0445bc5e9be8bf4bd796` |
| X1 / TRS | `27043c1e350b97c80991f0c504d7ca62f2150e9518a5785c071ae67f99d3ebac` | `b9f9b4e942e03f8866737f943a733e6062b26d4a693e34622616f6d0463ab6d2` |
| GO2 / No TRS | `1fcda0349bb01b5d88fe36ba93abf6c7992e4ad8b10c179e1b5560629d7d66b7` | `0430dde06881765effd0a1ffaf171dc9e117b3277cbd04fd7a7e652370346f71` |
| GO2 / TRS | `5f2ff12adb02e5809fc138d8472d51f3e68826e7c0465519c33171d89b26d8fa` | `1c437c00cf0ca34b8bdac5d83b3da9489306bd99abb604d0f05e31cfaef63eb9` |
