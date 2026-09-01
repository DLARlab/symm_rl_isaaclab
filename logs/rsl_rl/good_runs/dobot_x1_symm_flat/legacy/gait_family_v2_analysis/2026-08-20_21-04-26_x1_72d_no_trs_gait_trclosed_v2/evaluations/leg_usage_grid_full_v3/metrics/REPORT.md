# Leg usage grid analysis

- Robot: `x1`
- Protocol: `full` (`leg_usage_grid_full_v3`)
- Checkpoint: `D:\symm_rl_isaaclab\logs\rsl_rl\good_runs\dobot_x1_symm_flat\2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2\model_19999.pt` (SHA-256 `7cd84ae3efbb622129bba44ef0f05698f0f495303a1f663d98e543c96330d9d6`)
- Grid: 60 planned protocol cells (10 available gait rows, 6 available velocities)
- Window: 5 s settling, then 10 s measurement
- Coverage: 60/60 valid cells (100.0%)
- Domain coverage: velocity 60/60, heading 60/60, gait 60/60, raw load 60/60, GRF load 60/60, normalized load 60/60
- Tracking quality: 39/60 velocity-domain cells pass both `planar RMSE <= 0.05 + 0.25 * abs(vx)` [m/s] and `yaw RMSE <= 0.05` [rad/s]
- Analysis provenance: `analysis_provenance.json` (record `8175ba7c57f2c6141ae17c00fd765e9ae60c86cd0c3ba9a8b32804c3d4ad984a`)

Primary balance is the mean of per-cell absolute front/hind imbalance. Signed imbalance is retained in `cell_metrics.csv` and `family_metrics.csv`, but opposite signs never cancel in the primary score. The overall score gives each of trot, bound, half-bound, and gallop equal weight.

## Coverage by family

| Family | Valid | Expected | Coverage | Tracking pass |
|---|---:|---:|---:|---:|
| trot | 6 | 6 | 100.0% | 6/6 |
| bound | 6 | 6 | 100.0% | 6/6 |
| half bound | 24 | 24 | 100.0% | 19/24 |
| gallop | 24 | 24 | 100.0% | 8/24 |

## Family-balanced primary results

| Metric | All-valid mean absolute imbalance | Tracking-qualified mean | Valid complete | Tracking complete |
|---|---:|---:|:---:|:---:|
| Normalized torque squared | 26.149% | N/A | yes | no |
| Absolute mechanical work | 23.917% | N/A | yes | no |
| Vertical GRF impulse | 6.999% | N/A | yes | no |

## Tracking-qualified usage rates

Complete-cycle trimming can produce slightly different measurement durations, so rates are compared instead of raw totals.

| Metric | Family-balanced total / s | Family-balanced total / directed m |
|---|---:|---:|
| Normalized torque squared | N/A | N/A |
| Absolute mechanical work | N/A | N/A |
| Vertical GRF impulse | N/A | N/A |

A cell with missing/short data, an episode termination, invalid arrays, or nonpositive commanded-direction progress remains explicit in `coverage.csv`. Cost-per-distance fields are N/A for nonpositive progress; the cell's time-normalized balance metrics remain available when its rollout is otherwise valid.
Poor command tracking is retained rather than filtered and is flagged using `tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and `yaw_tracking_rmse_radps <= 0.05`. The category and overall SVG figures use the tracking-qualified aggregates; the CSV/JSON tables retain both views.
