# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `97b19eaffca4b515ee46baf2d2dcd4d63963e8dcce1519c375b81eb04d413b9a`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0550 | 60/60 |
| Yaw RMSE [rad/s] | 0.0672 | 60/60 |
| Boundary-excluded gait agreement | 86.66% | 60/60 |
| Heading RMSE [rad] | 0.0234 | 60/60 |
| Absolute-work front/hind imbalance | 15.52% | 60/60 |
| Absolute-work worst-leg share | 34.14% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 32.27% | 60/60 |
| Normalized-torque worst-leg share | 34.70% | 60/60 |
| Velocity-only success | 11/60 | valid velocity domain |
| Gait-only success | 27/60 | valid gait domain |
| Joint velocity+gait success | 8/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1353.8 | 489.774 | 1.58815 | 434.084 |
| FR | 1231.51 | 400.348 | 1.4774 | 434.134 |
| RL | 1100.55 | 613.977 | 1.13179 | 294.739 |
| RR | 945.917 | 535.863 | 0.961057 | 280.849 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
