# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `8e78b5f388f09b641339f8bd970becb07c2b6d9f5bebf13da79ab1a3e0f1c3cd`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0418 | 60/60 |
| Yaw RMSE [rad/s] | 0.0393 | 60/60 |
| Boundary-excluded gait agreement | 94.10% | 60/60 |
| Heading RMSE [rad] | 0.0149 | 60/60 |
| Absolute-work front/hind imbalance | 15.96% | 60/60 |
| Absolute-work worst-leg share | 36.48% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 27.76% | 60/60 |
| Normalized-torque worst-leg share | 33.47% | 60/60 |
| Velocity-only success | 54/60 | valid velocity domain |
| Gait-only success | 52/60 | valid gait domain |
| Joint velocity+gait success | 47/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 715.019 | 397.552 | 1.30223 | 403.777 |
| FR | 759.183 | 412.346 | 1.37874 | 409.203 |
| RL | 733.302 | 374.303 | 1.11215 | 418.375 |
| RR | 684.15 | 365.938 | 1.0142 | 389.733 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
