# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `14989c8d0e4b6ece9ba93c41291b06cb8311126dbf707d5a6215a37ac95289e1`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0486 | 60/60 |
| Yaw RMSE [rad/s] | 0.0682 | 60/60 |
| Boundary-excluded gait agreement | 87.68% | 60/60 |
| Heading RMSE [rad] | 0.0255 | 60/60 |
| Absolute-work front/hind imbalance | 12.19% | 60/60 |
| Absolute-work worst-leg share | 32.49% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 31.05% | 60/60 |
| Normalized-torque worst-leg share | 34.81% | 60/60 |
| Velocity-only success | 11/60 | valid velocity domain |
| Gait-only success | 28/60 | valid gait domain |
| Joint velocity+gait success | 7/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1324.2 | 448.273 | 1.48772 | 417.115 |
| FR | 1341.53 | 445.647 | 1.5761 | 422.126 |
| RL | 929.095 | 425.099 | 0.944735 | 298.249 |
| RR | 1006.82 | 549.389 | 1.01153 | 298.921 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
