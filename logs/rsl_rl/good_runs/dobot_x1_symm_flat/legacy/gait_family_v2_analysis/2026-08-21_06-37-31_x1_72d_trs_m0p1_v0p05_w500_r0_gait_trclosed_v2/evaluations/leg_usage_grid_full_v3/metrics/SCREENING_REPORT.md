# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `16706ca26ed3edc69438520f8e6e48346bc6f016c052fd37155fca94a4625a08`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0357 | 60/60 |
| Yaw RMSE [rad/s] | 0.0368 | 60/60 |
| Boundary-excluded gait agreement | 93.92% | 60/60 |
| Heading RMSE [rad] | 0.0122 | 60/60 |
| Absolute-work front/hind imbalance | 12.38% | 60/60 |
| Absolute-work worst-leg share | 34.61% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 28.16% | 60/60 |
| Normalized-torque worst-leg share | 32.32% | 60/60 |
| Velocity-only success | 56/60 | valid velocity domain |
| Gait-only success | 48/60 | valid gait domain |
| Joint velocity+gait success | 45/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 723.749 | 386.706 | 1.26884 | 410.003 |
| FR | 730.058 | 406.697 | 1.31118 | 387.022 |
| RL | 711.245 | 348.03 | 1.09032 | 406.873 |
| RR | 739.201 | 340.582 | 1.24106 | 413.876 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
