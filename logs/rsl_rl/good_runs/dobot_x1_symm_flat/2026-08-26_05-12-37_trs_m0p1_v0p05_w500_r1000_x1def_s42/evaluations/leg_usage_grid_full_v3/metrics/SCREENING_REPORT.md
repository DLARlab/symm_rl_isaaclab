# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `62706ae3e91be0411b841f2d24ff2581c66c188dfd7224e9a35f7817670d27be`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0431 | 60/60 |
| Yaw RMSE [rad/s] | 0.0359 | 60/60 |
| Boundary-excluded gait agreement | 94.50% | 60/60 |
| Heading RMSE [rad] | 0.0197 | 60/60 |
| Absolute-work front/hind imbalance | 11.04% | 60/60 |
| Absolute-work worst-leg share | 33.83% | 60/60 |
| Modal absolute-work worst leg | fr | 60/60 |
| Vertical-GRF worst-leg share | 28.04% | 60/60 |
| Normalized-torque worst-leg share | 33.34% | 60/60 |
| Velocity-only success | 52/60 | valid velocity domain |
| Gait-only success | 48/60 | valid gait domain |
| Joint velocity+gait success | 42/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 746.124 | 406.756 | 1.44112 | 400.145 |
| FR | 747.663 | 401.823 | 1.34891 | 422.329 |
| RL | 730.538 | 392.276 | 1.10818 | 388.374 |
| RR | 725.629 | 395.749 | 1.15248 | 405.67 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
