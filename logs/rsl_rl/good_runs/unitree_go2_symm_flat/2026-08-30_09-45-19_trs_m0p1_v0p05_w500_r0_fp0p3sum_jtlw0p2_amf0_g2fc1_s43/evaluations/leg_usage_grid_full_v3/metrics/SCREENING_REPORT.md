# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `dff79524d302b278d653b90e80c22316bdcde7a8d1831dcec9874a589e453abd`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0406 | 60/60 |
| Yaw RMSE [rad/s] | 0.0662 | 60/60 |
| Boundary-excluded gait agreement | 85.70% | 60/60 |
| Heading RMSE [rad] | 0.0230 | 60/60 |
| Absolute-work front/hind imbalance | 14.32% | 60/60 |
| Absolute-work worst-leg share | 32.41% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 33.09% | 60/60 |
| Normalized-torque worst-leg share | 35.34% | 60/60 |
| Velocity-only success | 7/60 | valid velocity domain |
| Gait-only success | 21/60 | valid gait domain |
| Joint velocity+gait success | 5/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1373.51 | 455.213 | 1.65973 | 442.745 |
| FR | 1531.13 | 465.662 | 1.91932 | 467.054 |
| RL | 861.818 | 552.195 | 0.932631 | 266.266 |
| RR | 827.005 | 517.718 | 0.901245 | 264.186 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
