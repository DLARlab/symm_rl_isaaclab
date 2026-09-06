# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `c77c261fd41f59155a84d87df45ed294b10235fb549495777ceffa556a1f5687`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0427 | 60/60 |
| Yaw RMSE [rad/s] | 0.0676 | 60/60 |
| Boundary-excluded gait agreement | 85.84% | 60/60 |
| Heading RMSE [rad] | 0.0317 | 60/60 |
| Absolute-work front/hind imbalance | 17.43% | 60/60 |
| Absolute-work worst-leg share | 33.54% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 32.73% | 60/60 |
| Normalized-torque worst-leg share | 34.46% | 60/60 |
| Velocity-only success | 7/60 | valid velocity domain |
| Gait-only success | 25/60 | valid gait domain |
| Joint velocity+gait success | 7/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1578.84 | 432.701 | 1.9433 | 459.263 |
| FR | 1392.55 | 456.767 | 1.72368 | 425.67 |
| RL | 969.976 | 623.284 | 1.03909 | 279.238 |
| RR | 1016.08 | 667.782 | 1.11235 | 278.86 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
