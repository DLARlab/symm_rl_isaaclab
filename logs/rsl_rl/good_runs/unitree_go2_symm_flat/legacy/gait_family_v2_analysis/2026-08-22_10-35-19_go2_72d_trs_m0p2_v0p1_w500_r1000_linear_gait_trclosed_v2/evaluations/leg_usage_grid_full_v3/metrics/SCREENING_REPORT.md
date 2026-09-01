# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `57aeaef5ee6e998c199bd8e3017223844692f0cb10724c23580ce36633779489`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0557 | 60/60 |
| Yaw RMSE [rad/s] | 0.0927 | 60/60 |
| Boundary-excluded gait agreement | 83.39% | 60/60 |
| Heading RMSE [rad] | 0.0234 | 60/60 |
| Absolute-work front/hind imbalance | 12.10% | 60/60 |
| Absolute-work worst-leg share | 33.16% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 30.48% | 60/60 |
| Normalized-torque worst-leg share | 33.27% | 60/60 |
| Velocity-only success | 0/60 | valid velocity domain |
| Gait-only success | 17/60 | valid gait domain |
| Joint velocity+gait success | 0/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1220.58 | 509.491 | 1.39823 | 411.23 |
| FR | 1406.25 | 505.856 | 1.4604 | 419.801 |
| RL | 881.427 | 503.947 | 0.925116 | 285.687 |
| RR | 1053.19 | 548.597 | 1.07158 | 314.008 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
