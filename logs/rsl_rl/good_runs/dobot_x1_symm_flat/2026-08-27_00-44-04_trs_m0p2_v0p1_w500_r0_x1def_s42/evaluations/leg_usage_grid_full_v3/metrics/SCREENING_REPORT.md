# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `ad311f315d7bc31f99a3ad66f8243237a67918a48a1cbc4eb6268b2565630f12`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0387 | 60/60 |
| Yaw RMSE [rad/s] | 0.0344 | 60/60 |
| Boundary-excluded gait agreement | 93.74% | 60/60 |
| Heading RMSE [rad] | 0.0140 | 60/60 |
| Absolute-work front/hind imbalance | 15.72% | 60/60 |
| Absolute-work worst-leg share | 37.09% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 27.83% | 60/60 |
| Normalized-torque worst-leg share | 32.60% | 60/60 |
| Velocity-only success | 52/60 | valid velocity domain |
| Gait-only success | 44/60 | valid gait domain |
| Joint velocity+gait success | 38/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 685.182 | 352.432 | 1.13616 | 399.418 |
| FR | 720.77 | 345.361 | 1.30796 | 409.712 |
| RL | 783.537 | 412.233 | 1.10843 | 414.236 |
| RR | 697.071 | 355.039 | 0.990716 | 403.813 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
