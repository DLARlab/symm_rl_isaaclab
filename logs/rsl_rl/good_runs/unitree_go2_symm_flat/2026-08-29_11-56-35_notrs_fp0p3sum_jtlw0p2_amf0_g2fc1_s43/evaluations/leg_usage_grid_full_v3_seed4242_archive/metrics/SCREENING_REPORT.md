# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `eb1070b5f85b03a1233fa0308b04650cc3ac1dc39548b903dc6201cda54ed2b8`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0447 | 60/60 |
| Yaw RMSE [rad/s] | 0.0677 | 60/60 |
| Boundary-excluded gait agreement | 88.32% | 60/60 |
| Heading RMSE [rad] | 0.0213 | 60/60 |
| Absolute-work front/hind imbalance | 13.72% | 60/60 |
| Absolute-work worst-leg share | 31.30% | 60/60 |
| Modal absolute-work worst leg | rr | 60/60 |
| Vertical-GRF worst-leg share | 31.10% | 60/60 |
| Normalized-torque worst-leg share | 35.20% | 60/60 |
| Velocity-only success | 14/60 | valid velocity domain |
| Gait-only success | 35/60 | valid gait domain |
| Joint velocity+gait success | 11/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 1278.82 | 369.664 | 1.53606 | 423.592 |
| FR | 1182.84 | 375.207 | 1.41466 | 399.454 |
| RL | 1021.88 | 446.718 | 1.04403 | 313.296 |
| RR | 1035.21 | 462.337 | 1.05166 | 305.35 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
