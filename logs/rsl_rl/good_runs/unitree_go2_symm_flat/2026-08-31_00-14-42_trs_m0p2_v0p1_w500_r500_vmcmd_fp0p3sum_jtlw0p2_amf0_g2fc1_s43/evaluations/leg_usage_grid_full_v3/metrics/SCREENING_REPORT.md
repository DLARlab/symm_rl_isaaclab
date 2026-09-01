# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `d98ff52ff70ebc6f6024bad97a9c0dd1b6c6ae11ad825245ddfc569c614d8439`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0428 | 60/60 |
| Yaw RMSE [rad/s] | 0.0605 | 60/60 |
| Boundary-excluded gait agreement | 87.56% | 60/60 |
| Heading RMSE [rad] | 0.0171 | 60/60 |
| Absolute-work front/hind imbalance | 10.27% | 60/60 |
| Absolute-work worst-leg share | 31.14% | 60/60 |
| Modal absolute-work worst leg | fr | 60/60 |
| Vertical-GRF worst-leg share | 31.01% | 60/60 |
| Normalized-torque worst-leg share | 33.86% | 60/60 |
| Velocity-only success | 14/60 | valid velocity domain |
| Gait-only success | 23/60 | valid gait domain |
| Joint velocity+gait success | 7/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1333.23 | 432.099 | 1.51385 | 426.672 |
| FR | 1316.55 | 440.656 | 1.59229 | 425.387 |
| RL | 901.941 | 442.766 | 0.924149 | 289.4 |
| RR | 921.259 | 456.584 | 0.933828 | 290.343 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
