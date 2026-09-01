# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `5bcb69a972c15aa17bdd9f11531a0628112627b95b24fd3eff60a97aeecba4e9`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0341 | 60/60 |
| Yaw RMSE [rad/s] | 0.0428 | 60/60 |
| Boundary-excluded gait agreement | 93.56% | 60/60 |
| Heading RMSE [rad] | 0.0122 | 60/60 |
| Absolute-work front/hind imbalance | 19.57% | 60/60 |
| Absolute-work worst-leg share | 38.21% | 60/60 |
| Modal absolute-work worst leg | rl | 60/60 |
| Vertical-GRF worst-leg share | 27.72% | 60/60 |
| Normalized-torque worst-leg share | 32.67% | 60/60 |
| Velocity-only success | 44/60 | valid velocity domain |
| Gait-only success | 42/60 | valid gait domain |
| Joint velocity+gait success | 33/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 883.106 | 525.108 | 1.4528 | 424.112 |
| FR | 985.344 | 568.078 | 1.62508 | 420.028 |
| RL | 962.968 | 599.318 | 1.19641 | 392.749 |
| RR | 835.419 | 514.792 | 1.15283 | 386.665 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
