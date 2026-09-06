# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `025e42bdfda280bc858139d8184e07bb51e59d79eba7e3835dc9a2f76c7d070b`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0470 | 60/60 |
| Yaw RMSE [rad/s] | 0.0644 | 60/60 |
| Boundary-excluded gait agreement | 86.96% | 60/60 |
| Heading RMSE [rad] | 0.0253 | 60/60 |
| Absolute-work front/hind imbalance | 21.88% | 60/60 |
| Absolute-work worst-leg share | 35.33% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 32.09% | 60/60 |
| Normalized-torque worst-leg share | 33.44% | 60/60 |
| Velocity-only success | 12/60 | valid velocity domain |
| Gait-only success | 23/60 | valid gait domain |
| Joint velocity+gait success | 8/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1370.22 | 431.622 | 1.69528 | 446.85 |
| FR | 1340.05 | 447.813 | 1.58124 | 429.341 |
| RL | 1048.06 | 688.683 | 1.08426 | 280.27 |
| RR | 1093.93 | 749.163 | 1.12615 | 285.425 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
