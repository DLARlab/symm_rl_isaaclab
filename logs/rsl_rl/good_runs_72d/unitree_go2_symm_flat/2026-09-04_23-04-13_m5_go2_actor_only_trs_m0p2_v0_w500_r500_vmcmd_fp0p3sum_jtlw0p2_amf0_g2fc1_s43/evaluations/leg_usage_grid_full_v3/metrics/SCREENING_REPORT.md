# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `d360371207da8fa4f5798711091b1137dde7312f8160668705344bfaff0fd776`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0496 | 60/60 |
| Yaw RMSE [rad/s] | 0.0628 | 60/60 |
| Boundary-excluded gait agreement | 87.31% | 60/60 |
| Heading RMSE [rad] | 0.0157 | 60/60 |
| Absolute-work front/hind imbalance | 8.26% | 60/60 |
| Absolute-work worst-leg share | 30.09% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 30.82% | 60/60 |
| Normalized-torque worst-leg share | 32.90% | 60/60 |
| Velocity-only success | 17/60 | valid velocity domain |
| Gait-only success | 20/60 | valid gait domain |
| Joint velocity+gait success | 11/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1295.69 | 461.894 | 1.53007 | 432.11 |
| FR | 1264.85 | 477.582 | 1.46891 | 413.756 |
| RL | 1002.26 | 502.989 | 1.00821 | 292.499 |
| RR | 939.527 | 493.539 | 0.938931 | 292.148 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
