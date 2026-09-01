# Leg usage grid analysis

- Robot: `go2`
- Protocol: `full` (`leg_usage_grid_full_v3`)
- Checkpoint: `D:\symm_rl_isaaclab\logs\rsl_rl\unitree_go2_symm_flat\2026-08-31_00-14-42_d10_go2_d5base_trs_m0p2_v0p1_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43\model_19999.pt` (SHA-256 `d98ff52ff70ebc6f6024bad97a9c0dd1b6c6ae11ad825245ddfc569c614d8439`)
- Grid: 60 planned protocol cells (10 available gait rows, 6 available velocities)
- Window: 5 s settling, then 10 s measurement
- Coverage: 60/60 valid cells (100.0%)
- Domain coverage: velocity 60/60, heading 60/60, gait 60/60, raw load 60/60, GRF load 60/60, normalized load 60/60
- Tracking quality: 14/60 velocity-domain cells pass both `planar RMSE <= 0.05 + 0.25 * abs(vx)` [m/s] and `yaw RMSE <= 0.05` [rad/s]
- Analysis provenance: `analysis_provenance.json` (record `e17b994e83d6f5acfaeb806963364926427cf2b6e74d3c7e8c6e398b0206e062`)

Primary balance is the mean of per-cell absolute front/hind imbalance. Signed imbalance is retained in `cell_metrics.csv` and `family_metrics.csv`, but opposite signs never cancel in the primary score. The overall score gives each of trot, bound, half-bound, and gallop equal weight.

## Coverage by family

| Family | Valid | Expected | Coverage | Tracking pass |
|---|---:|---:|---:|---:|
| trot | 6 | 6 | 100.0% | 2/6 |
| bound | 6 | 6 | 100.0% | 6/6 |
| half bound | 24 | 24 | 100.0% | 5/24 |
| gallop | 24 | 24 | 100.0% | 1/24 |

## Family-balanced primary results

| Metric | All-valid mean absolute imbalance | Tracking-qualified mean | Valid complete | Tracking complete |
|---|---:|---:|:---:|:---:|
| Normalized torque squared | 25.810% | N/A | yes | no |
| Absolute mechanical work | 10.992% | N/A | yes | no |
| Vertical GRF impulse | 18.875% | N/A | yes | no |

## Tracking-qualified usage rates

Complete-cycle trimming can produce slightly different measurement durations, so rates are compared instead of raw totals.

| Metric | Family-balanced total / s | Family-balanced total / directed m |
|---|---:|---:|
| Normalized torque squared | N/A | N/A |
| Absolute mechanical work | N/A | N/A |
| Vertical GRF impulse | N/A | N/A |

A cell with missing/short data, an episode termination, invalid arrays, or nonpositive commanded-direction progress remains explicit in `coverage.csv`. Cost-per-distance fields are N/A for nonpositive progress; the cell's time-normalized balance metrics remain available when its rollout is otherwise valid.
Poor command tracking is retained rather than filtered and is flagged using `tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and `yaw_tracking_rmse_radps <= 0.05`. The category and overall SVG figures use the tracking-qualified aggregates; the CSV/JSON tables retain both views.
