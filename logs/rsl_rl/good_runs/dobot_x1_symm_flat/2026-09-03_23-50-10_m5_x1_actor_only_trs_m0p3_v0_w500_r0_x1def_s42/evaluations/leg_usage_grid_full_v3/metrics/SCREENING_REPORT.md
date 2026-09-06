# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `cf22bcdfe21feecce5c4fb936bc574420bcfe0339fdf9e145634db178387b97f`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0399 | 60/60 |
| Yaw RMSE [rad/s] | 0.0413 | 60/60 |
| Boundary-excluded gait agreement | 94.01% | 60/60 |
| Heading RMSE [rad] | 0.0162 | 60/60 |
| Absolute-work front/hind imbalance | 11.69% | 60/60 |
| Absolute-work worst-leg share | 35.55% | 60/60 |
| Modal absolute-work worst leg | fr | 60/60 |
| Vertical-GRF worst-leg share | 28.07% | 60/60 |
| Normalized-torque worst-leg share | 33.15% | 60/60 |
| Velocity-only success | 44/60 | valid velocity domain |
| Gait-only success | 45/60 | valid gait domain |
| Joint velocity+gait success | 32/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 658.072 | 322.428 | 1.09107 | 384.351 |
| FR | 834.889 | 425.447 | 1.41788 | 417.303 |
| RL | 712.803 | 359.497 | 1.06549 | 409.32 |
| RR | 737.33 | 344.316 | 1.08196 | 404.914 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
