# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `d55b557ee352164f0f5b7fccb3423c4de9fe6a40edc2a06f89e868c1ee224b33`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0496 | 60/60 |
| Yaw RMSE [rad/s] | 0.0752 | 60/60 |
| Boundary-excluded gait agreement | 87.88% | 60/60 |
| Heading RMSE [rad] | 0.0314 | 60/60 |
| Absolute-work front/hind imbalance | 19.97% | 60/60 |
| Absolute-work worst-leg share | 35.50% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 31.64% | 60/60 |
| Normalized-torque worst-leg share | 33.80% | 60/60 |
| Velocity-only success | 5/60 | valid velocity domain |
| Gait-only success | 30/60 | valid gait domain |
| Joint velocity+gait success | 4/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1203.48 | 430.992 | 1.32094 | 398.375 |
| FR | 1403.2 | 404.426 | 1.64362 | 433.677 |
| RL | 1133.54 | 638.13 | 1.12687 | 317.302 |
| RR | 1132.99 | 646.599 | 1.14863 | 298.221 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
