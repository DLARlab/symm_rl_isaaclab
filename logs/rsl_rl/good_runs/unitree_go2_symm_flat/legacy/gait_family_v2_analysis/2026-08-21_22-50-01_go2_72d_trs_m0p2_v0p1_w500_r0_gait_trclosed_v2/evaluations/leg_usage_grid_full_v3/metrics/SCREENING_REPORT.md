# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `0472d9a599ce5783948a47b5bbe7f5f4cbc430dd343398852072e8f216e15ad5`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0458 | 60/60 |
| Yaw RMSE [rad/s] | 0.0911 | 60/60 |
| Boundary-excluded gait agreement | 79.12% | 60/60 |
| Heading RMSE [rad] | 0.0306 | 60/60 |
| Absolute-work front/hind imbalance | 10.24% | 60/60 |
| Absolute-work worst-leg share | 34.64% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 30.59% | 60/60 |
| Normalized-torque worst-leg share | 33.69% | 60/60 |
| Velocity-only success | 0/60 | valid velocity domain |
| Gait-only success | 9/60 | valid gait domain |
| Joint velocity+gait success | 0/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1530.29 | 625.246 | 1.7044 | 418.645 |
| FR | 1282.26 | 530.784 | 1.35725 | 407.929 |
| RL | 1145.05 | 725.979 | 1.17669 | 299.234 |
| RR | 1006.57 | 551.726 | 1.08522 | 311.864 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
