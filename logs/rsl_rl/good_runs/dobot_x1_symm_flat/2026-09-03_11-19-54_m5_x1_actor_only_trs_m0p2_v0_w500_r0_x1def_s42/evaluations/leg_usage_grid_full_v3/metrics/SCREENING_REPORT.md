# Publication screening report

Protocol `full` / `leg_usage_grid_full_v3`; checkpoint `49cef518dd5485f35c6d2b021e671f7060d013d723eb0f53640509ed78209112`.

| Domain | Result | Coverage |
|---|---:|---:|
| Training mean reward | N/A | event-file scalar |
| vx relative RMSE | 0.0420 | 60/60 |
| Yaw RMSE [rad/s] | 0.0446 | 60/60 |
| Boundary-excluded gait agreement | 93.99% | 60/60 |
| Heading RMSE [rad] | 0.0162 | 60/60 |
| Absolute-work front/hind imbalance | 5.95% | 60/60 |
| Absolute-work worst-leg share | 32.65% | 60/60 |
| Modal absolute-work worst leg | fl | 60/60 |
| Vertical-GRF worst-leg share | 27.55% | 60/60 |
| Normalized-torque worst-leg share | 33.28% | 60/60 |
| Velocity-only success | 43/60 | valid velocity domain |
| Gait-only success | 52/60 | valid gait domain |
| Joint velocity+gait success | 40/60 | domain 60/60 planned |

## Per-leg load means

Raw torque/work remain reportable without C1 provenance; normalized torque is available only in the strict configured-effort-limit domain, and GRF only in the ground-filtered contact domain.

| Leg | Raw torque-squared [N²m²s] | Absolute work [J] | Normalized torque-squared [s] | Vertical GRF impulse [Ns] |
|---|---:|---:|---:|---:|
| FL | 779.838 | 358.109 | 1.27042 | 422.729 |
| FR | 646.419 | 288.987 | 1.26483 | 401.924 |
| RL | 663.253 | 333.387 | 0.969992 | 394.315 |
| RR | 673.073 | 324.191 | 0.908938 | 397.516 |

Failures and domain-specific exclusions are enumerated in `coverage.csv`; per-foot and per-joint results are in `foot_metrics.csv` and `joint_metrics.csv`.
