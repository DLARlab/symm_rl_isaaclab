# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `256e7d8b29f5587b6a1e4fa955f45481a556420ed3458a23133bed50e31bea5a`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0524 | 60/60 |
| Yaw RMSE [rad/s] | 0.0913 | 60/60 |
| Boundary-excluded gait agreement | 79.13% | 60/60 |
| Heading RMSE [rad] | 0.0431 | 60/60 |
| Absolute-work front/hind imbalance | 23.55% | 60/60 |
| Absolute-work worst-leg share | 37.34% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 34.05% | 60/60 |
| Normalized-torque worst-leg share | 31.88% | 60/60 |
| Velocity-only success | 0/60 | valid velocity domain |
| Gait-only success | 20/60 | valid gait domain |
| Joint velocity+gait success | 0/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1260.55 | 516.896 | 1.56214 | 452.546 |
| FR | 1383.55 | 604.939 | 1.52499 | 417.297 |
| RL | 1109.95 | 751.334 | 1.20987 | 291.497 |
| RR | 1417.3 | 1135.49 | 1.43996 | 288.754 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
