# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `d147caa27a3d99d36435eaf827470bd07981d44e5babb7dc94069c86f59f963e`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0458 | 60/60 |
| Yaw RMSE [rad/s] | 0.0643 | 60/60 |
| Boundary-excluded gait agreement | 85.87% | 60/60 |
| Heading RMSE [rad] | 0.0208 | 60/60 |
| Absolute-work front/hind imbalance | 11.12% | 60/60 |
| Absolute-work worst-leg share | 31.41% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 30.65% | 60/60 |
| Normalized-torque worst-leg share | 33.04% | 60/60 |
| Velocity-only success | 3/60 | valid velocity domain |
| Gait-only success | 24/60 | valid gait domain |
| Joint velocity+gait success | 1/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1263.23 | 502.046 | 1.38382 | 406.324 |
| FR | 1187.47 | 426.003 | 1.33392 | 416.133 |
| RL | 1122.88 | 528.341 | 1.11083 | 299.3 |
| RR | 1078.25 | 482.997 | 1.04127 | 315.804 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
