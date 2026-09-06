# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `fe03f6ad9097698cb19d6a7c525334c450e51d8a527e6ac8452fd8c7344b88cc`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0390 | 60/60 |
| Yaw RMSE [rad/s] | 0.0427 | 60/60 |
| Boundary-excluded gait agreement | 95.50% | 60/60 |
| Heading RMSE [rad] | 0.0164 | 60/60 |
| Absolute-work front/hind imbalance | 21.39% | 60/60 |
| Absolute-work worst-leg share | 36.10% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 28.73% | 60/60 |
| Normalized-torque worst-leg share | 37.60% | 60/60 |
| Velocity-only success | 39/60 | valid velocity domain |
| Gait-only success | 50/60 | valid gait domain |
| Joint velocity+gait success | 33/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 987.584 | 544.384 | 1.99775 | 423.974 |
| FR | 809.446 | 426.695 | 1.57522 | 401.951 |
| RL | 654.412 | 289.527 | 1.10175 | 414.592 |
| RR | 701.974 | 293.88 | 1.14154 | 403.355 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
