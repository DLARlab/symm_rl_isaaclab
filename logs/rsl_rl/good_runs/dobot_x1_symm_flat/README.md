# Dobot X1 curated runs

The current gait-family V2 cohort contains three seed-42, 20,000-iteration
training runs and a matched 60-cell leg-usage grid for every terminal
checkpoint. Each grid evaluates all 10 training gaits at
`vx = ±{0.5, 1.0, 1.5} m/s`, with 5 seconds of settling and 10 seconds of
measurement per cell.

## Gait-family V2 runs

- [No TRS](2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2/)
- [TRS m0.1/v0.05, w500/r0](2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/)
- [TRS m0.2/v0.1, w500/r0](2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/)
- [Learning and fixed-grid comparison](gait_famili_v2_analysis/REPORT.md)

All 180 grid cells are valid and planar-tracking-qualified. In this
single-training-seed cohort, `m0.1/v0.05` is the observed Pareto winner: it has
the best learning curve and the lowest family-balanced torque-squared, work,
and vertical-GRF front/hind imbalance and per-metre exposure. These are
checkpoint comparisons, not multi-seed method-level significance claims.

Older retained studies and their reports remain available beside this cohort.
The full chronology and reproduction commands are in the
[gait-family V2 run log](../RUN_LOG_GAIT_FAMILY_V2.md).
