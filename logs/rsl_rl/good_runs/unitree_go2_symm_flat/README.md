# Unitree Go2 curated runs

The current gait-family V2 cohort contains four seed-42, 20,000-iteration
training runs and a matched 60-cell leg-usage grid for every terminal
checkpoint. Each grid evaluates all 10 training gaits at
`vx = ±{0.5, 1.0, 1.5} m/s`, with 5 seconds of settling and 10 seconds of
measurement per cell.

## Gait-family V2 runs

- [No TRS](2026-08-20_21-04-16_go2_72d_no_trs_gait_trclosed_v2/)
- [TRS m0.1/v0.05, w500/r0](2026-08-21_10-22-39_go2_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/)
- [TRS m0.2/v0.1, w500/r0](2026-08-21_22-50-01_go2_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/)
- [TRS m0.2/v0.1, w500/r1000 linear](2026-08-22_10-35-19_go2_72d_trs_m0p2_v0p1_w500_r1000_linear_gait_trclosed_v2/)
- [Learning and fixed-grid comparison](gait_famili_v2_analysis/REPORT.md)

All 240 grid cells are valid and planar-tracking-qualified, but no TRS run
dominates this cohort. No TRS retains the best full learning-curve AUC and
last-1,000 reward. The TRS policies improve selected front/hind allocation or
load-exposure measures, but still pay a reward cost. Go2 reward recovery while
preserving those leg-usage gains is therefore an explicit open objective, not
a completed milestone claim.

The strongest next coefficient screen is to hold the empirically better
`w500/r1000` schedule and decouple actor mirror strength from critic
time-reversal consistency, starting with `m0.2/v0.05` and `m0.15/v0.05`.
Winners should then be repeated across matched training seeds.

Older retained studies and their reports remain available beside this cohort.
The full chronology and reproduction commands are in the
[gait-family V2 run log](../RUN_LOG_GAIT_FAMILY_V2.md).
