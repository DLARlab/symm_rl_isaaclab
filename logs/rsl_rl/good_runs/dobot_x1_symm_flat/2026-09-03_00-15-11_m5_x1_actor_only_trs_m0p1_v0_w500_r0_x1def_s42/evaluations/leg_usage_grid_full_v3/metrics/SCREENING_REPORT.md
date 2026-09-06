# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `6b5d6bb8570bd4ef99eb7d9eca2c88add4c9ad0105903bc4c6d38c77c71b0401`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0400 | 60/60 |
| Yaw RMSE [rad/s] | 0.0328 | 60/60 |
| Boundary-excluded gait agreement | 94.25% | 60/60 |
| Heading RMSE [rad] | 0.0172 | 60/60 |
| Absolute-work front/hind imbalance | 10.92% | 60/60 |
| Absolute-work worst-leg share | 32.12% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 28.28% | 60/60 |
| Normalized-torque worst-leg share | 33.35% | 60/60 |
| Velocity-only success | 57/60 | valid velocity domain |
| Gait-only success | 60/60 | valid gait domain |
| Joint velocity+gait success | 57/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 738.457 | 382.06 | 1.42165 | 424.151 |
| FR | 691.385 | 342.938 | 1.21817 | 408.564 |
| RL | 639.403 | 293.298 | 0.950476 | 384.551 |
| RR | 659.406 | 293.514 | 1.06214 | 405.143 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
