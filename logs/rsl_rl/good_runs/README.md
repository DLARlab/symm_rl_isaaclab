# Curated symmetric-locomotion runs

This archive is maintained on branch
`72d-symm-v4-integration`. It contains every retained artifact from each
curated training run: TensorBoard events, parameters, training-time Git
provenance, plots, simulation data, videos, and deployment exports.

Only the iteration checkpoint series is reduced. Each run publishes its
numerically latest `model_*.pt`; intermediate iteration checkpoints remain
local and are excluded by [`.gitignore`](.gitignore).

## Milestone documentation

- [60D-to-72D milestone](MILESTONE_60D_TO_72D.md)
- [Phase Mapping V2 milestone](MILESTONE_PHASE_MAPPING_V2.md)
- [Leg-permutation fix and gait-family V2 milestone](MILESTONE_LEG_PERMUTATION_FIX_GAIT_FAMILY_V2.md)
- [Gait-family V2 run log](RUN_LOG_GAIT_FAMILY_V2.md)

## Current gait-family V2 status

The August 20-22 cohort uses the time-reversal-closed V2 training gait library
and the fixed 10-gait by 6-velocity leg-usage grid. All 420 recorded grid cells
are valid and pass the planar tracking criterion. See the robot-specific
[X1](dobot_x1_symm_flat/README.md) and [Go2](unitree_go2_symm_flat/README.md)
indexes for the retained checkpoints and concise results.

X1 has a clear observed winner: `m0.1/v0.05` improves both learning and the
family-balanced leg-usage measures. Go2 does not yet have a joint learning and
leg-usage winner. Its no-TRS checkpoint still has the best reward, so improving
Go2 TRS reward while retaining its allocation gains remains active work.

## Dobot X1 runs

- [`2026-07-11_02-59-13_no_trs`](dobot_x1_symm_flat/2026-07-11_02-59-13_no_trs/)
- [`2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral`](dobot_x1_symm_flat/2026-07-13_01-31-40_more_trs_lr1e4_fixed_zero_lateral/)
- [`2026-07-19_10-33-04_x1_no_trs_pitch0p35`](dobot_x1_symm_flat/2026-07-19_10-33-04_x1_no_trs_pitch0p35/)
- [`2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70`](dobot_x1_symm_flat/2026-07-19_17-46-27_x1_trs_m0p1_v0p05_w500_pitch0p35_pterm0p70/)
- [`2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500`](dobot_x1_symm_flat/2026-07-20_16-24-19_x1_trs_m0p20_v0p10_w500/)
- [`2026-08-02_10-28-45_x1_trs_m0p2_v0p10_w500_minv0_20k_512`](dobot_x1_symm_flat/2026-08-02_10-28-45_x1_trs_m0p2_v0p10_w500_minv0_20k_512/)
- [`2026-08-02_10-35-06_x1_no_trs_20k_512`](dobot_x1_symm_flat/2026-08-02_10-35-06_x1_no_trs_20k_512/)
- [`2026-08-02_21-02-49_x1_trs_m0p1_v0p05_w500_minv0_20k_512`](dobot_x1_symm_flat/2026-08-02_21-02-49_x1_trs_m0p1_v0p05_w500_minv0_20k_512/)
- [`2026-08-02_21-07-58_x1_trs_m0p3_v0p15_w500_minv0_20k_512`](dobot_x1_symm_flat/2026-08-02_21-07-58_x1_trs_m0p3_v0p15_w500_minv0_20k_512/)
- [`2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2`](dobot_x1_symm_flat/2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2/)
- [`2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2`](dobot_x1_symm_flat/2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/)
- [`2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2`](dobot_x1_symm_flat/2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/)
- [Phase Mapping V2 X1 analysis](dobot_x1_symm_flat/phase_mapping_v2_x1_trs_run_analysis/REPORT.md)
- [Gait-family V2 X1 fixed-grid analysis](dobot_x1_symm_flat/gait_famili_v2_analysis/REPORT.md)

## Unitree Go2 runs

- [`2026-07-07_00-11-14_no_trs`](unitree_go2_symm_flat/2026-07-07_00-11-14_no_trs/)
- [`2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral`](unitree_go2_symm_flat/2026-07-13_01-30-42_more_trs_lr1e4_fixed_zero_lateral/)
- [`2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20`](unitree_go2_symm_flat/2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20/)
- [`2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20`](unitree_go2_symm_flat/2026-07-19_17-37-55_go2_trs_m0p1_v0p05_w500_minv0_pitch0p50_pterm1p20/)
- [`2026-07-20_16-23-32_go2_trs_m0p20_v0p10_w500`](unitree_go2_symm_flat/2026-07-20_16-23-32_go2_trs_m0p20_v0p10_w500/)
- [`2026-07-31_22-48-10_go2_no_trs_20k_512`](unitree_go2_symm_flat/2026-07-31_22-48-10_go2_no_trs_20k_512/)
- [`2026-07-31_22-48-38_go2_trs_m0p2_v0p1_w500_20k_512`](unitree_go2_symm_flat/2026-07-31_22-48-38_go2_trs_m0p2_v0p1_w500_20k_512/)
- [`2026-08-01_11-09-46_go2_trs_m0p1_v0p05_w500_minv0_20k_512`](unitree_go2_symm_flat/2026-08-01_11-09-46_go2_trs_m0p1_v0p05_w500_minv0_20k_512/)
- [`2026-08-01_22-39-14_go2_trs_m0p3_v0p15_w500_minv0_20k_512`](unitree_go2_symm_flat/2026-08-01_22-39-14_go2_trs_m0p3_v0p15_w500_minv0_20k_512/)
- [`2026-08-20_21-04-16_go2_72d_no_trs_gait_trclosed_v2`](unitree_go2_symm_flat/2026-08-20_21-04-16_go2_72d_no_trs_gait_trclosed_v2/)
- [`2026-08-21_10-22-39_go2_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2`](unitree_go2_symm_flat/2026-08-21_10-22-39_go2_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/)
- [`2026-08-21_22-50-01_go2_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2`](unitree_go2_symm_flat/2026-08-21_22-50-01_go2_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/)
- [`2026-08-22_10-35-19_go2_72d_trs_m0p2_v0p1_w500_r1000_linear_gait_trclosed_v2`](unitree_go2_symm_flat/2026-08-22_10-35-19_go2_72d_trs_m0p2_v0p1_w500_r1000_linear_gait_trclosed_v2/)
- [Phase Mapping V2 Go2 analysis](unitree_go2_symm_flat/phase_mapping_v2_go2_trs_run_analysis/REPORT.md)
- [Gait-family V2 Go2 fixed-grid analysis](unitree_go2_symm_flat/gait_famili_v2_analysis/REPORT.md)

## Branch visibility

The repository's `main` branch contains only the early archive subset. Select
`72d-symm-v4-integration` in GitHub to view this complete curated set.
