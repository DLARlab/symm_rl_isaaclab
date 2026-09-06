# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `cf3d8c8baaef12541dd60b646a6be64be56bf00b9167c244c368e19afb0c5528`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0429 | 60/60 |
| Yaw RMSE [rad/s] | 0.0682 | 60/60 |
| Boundary-excluded gait agreement | 87.10% | 60/60 |
| Heading RMSE [rad] | 0.0167 | 60/60 |
| Absolute-work front/hind imbalance | 11.54% | 60/60 |
| Absolute-work worst-leg share | 31.28% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 30.81% | 60/60 |
| Normalized-torque worst-leg share | 33.14% | 60/60 |
| Velocity-only success | 10/60 | valid velocity domain |
| Gait-only success | 25/60 | valid gait domain |
| Joint velocity+gait success | 7/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1328.04 | 437.099 | 1.5088 | 423.396 |
| FR | 1281.15 | 427.758 | 1.48736 | 415.438 |
| RL | 991.161 | 469.2 | 1.00983 | 305.143 |
| RR | 980.805 | 478.57 | 1.01784 | 289.765 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
