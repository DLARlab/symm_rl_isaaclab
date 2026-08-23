# Leg usage grid analysis

- Robot: `x1`
- Checkpoint: `D:\symm_rl_isaaclab\logs\rsl_rl\dobot_x1_symm_flat\2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2_family_balanced_halfbound_frontrear_phasev4_contclock_footphase0p30_legperm0p20_hip0p10_sync_tol0p02_seed42\model_19999.pt` (SHA-256 `16706ca26ed3edc69438520f8e6e48346bc6f016c052fd37155fca94a4625a08`)
- Grid: 10 gait rows x 6 velocities
- Window: 5 s settling, then 10 s measurement
- Coverage: 60/60 valid cells (100.0%)
- Tracking quality: 56/60 valid cells pass both `planar RMSE <= 0.05 + 0.25 * abs(vx)` [m/s] and `yaw RMSE <= 0.05` [rad/s]
- Analysis provenance: `analysis_provenance.json` (record `3e4ca64e66deee91f266e6629e6c113e08d6713cbf5c5ef2b80c66ed52a4c0c8`)

Primary balance is the mean of per-cell absolute front/hind imbalance. Signed imbalance is retained in `cell_metrics.csv` and `family_metrics.csv`, but opposite signs never cancel in the primary score. The overall score gives each of trot, bound, half-bound, and gallop equal weight.

## Coverage by family

| Family | Valid | Expected | Coverage | Tracking pass |
|---|---:|---:|---:|---:|
| trot | 6 | 6 | 100.0% | 6/6 |
| bound | 6 | 6 | 100.0% | 6/6 |
| half bound | 24 | 24 | 100.0% | 24/24 |
| gallop | 24 | 24 | 100.0% | 20/24 |

## Family-balanced primary results

| Metric | All-valid mean absolute imbalance | Tracking-qualified mean | Valid complete | Tracking complete |
|---|---:|---:|:---:|:---:|
| Normalized torque squared | 6.642% | N/A | yes | no |
| Absolute mechanical work | 12.800% | N/A | yes | no |
| Vertical GRF impulse | 3.806% | N/A | yes | no |

## Tracking-qualified usage rates

Complete-cycle trimming can produce slightly different measurement durations, so rates are compared instead of raw totals.

| Metric | Family-balanced total / s | Family-balanced total / directed m |
|---|---:|---:|
| Normalized torque squared | N/A | N/A |
| Absolute mechanical work | N/A | N/A |
| Vertical GRF impulse | N/A | N/A |

A cell with missing/short data, an episode termination, invalid arrays, or nonpositive commanded-direction progress remains explicit in `coverage.csv`. Cost-per-distance fields are N/A for nonpositive progress; the cell's time-normalized balance metrics remain available when its rollout is otherwise valid.
Poor command tracking is retained rather than filtered and is flagged using `tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and `yaw_tracking_rmse_radps <= 0.05`. The category and overall SVG figures use the tracking-qualified aggregates; the CSV/JSON tables retain both views.
