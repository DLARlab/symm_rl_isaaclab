# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `58be72f743801708adb6a47355e718e82300903cbf259d891281475d3c74da8b`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0408 | 60/60 |
| Yaw RMSE [rad/s] | 0.0416 | 60/60 |
| Boundary-excluded gait agreement | 93.67% | 60/60 |
| Heading RMSE [rad] | 0.0143 | 60/60 |
| Absolute-work front/hind imbalance | 19.78% | 60/60 |
| Absolute-work worst-leg share | 37.95% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 28.22% | 60/60 |
| Normalized-torque worst-leg share | 31.26% | 60/60 |
| Velocity-only success | 45/60 | valid velocity domain |
| Gait-only success | 40/60 | valid gait domain |
| Joint velocity+gait success | 29/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 776.59 | 430.639 | 1.203 | 396.851 |
| FR | 770 | 431.015 | 1.22744 | 400.302 |
| RL | 904.358 | 520.864 | 1.2202 | 419.884 |
| RR | 781.976 | 450.349 | 1.1272 | 403.127 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
