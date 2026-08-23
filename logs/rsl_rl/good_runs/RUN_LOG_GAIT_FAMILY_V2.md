# Gait-family V2 run log

This log records the retained runs used by the leg-permutation fix and
gait-family V2 milestone. All training runs use seed 42, 512 environments, 24
steps per environment per iteration, and 20,000 iterations. The terminal
checkpoint is `model_19999.pt`.

The fixed-grid evaluation uses the time-reversal-closed V2 library: 10 gait
rows grouped into trot, bound, half-bound, and gallop; six signed commands
`vx = ±{0.5, 1.0, 1.5} m/s`; 5 seconds of settling; and 10 seconds of
cycle-trimmed measurement. Every listed checkpoint has 60/60 valid grid cells
with no evaluation termination.

## Dobot X1

| Started | Configuration | Run | Tail reward | Full AUC | Grid |
|---|---|---|---:|---:|---:|
| 2026-08-20 21:04 | No TRS | [`2026-08-20_21-04-26`](dobot_x1_symm_flat/2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2/) | 42.307 | 34.667 | 60/60 |
| 2026-08-21 06:37 | m0.1/v0.05, w500/r0 | [`2026-08-21_06-37-31`](dobot_x1_symm_flat/2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/) | 43.040 | 36.124 | 60/60 |
| 2026-08-21 22:50 | m0.2/v0.1, w500/r0 | [`2026-08-21_22-50-05`](dobot_x1_symm_flat/2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/) | 41.081 | 32.683 | 60/60 |

Observed status: `m0.1/v0.05` leads both learning and every reported
family-balanced leg-usage domain. The fixed grid is complete for all three
checkpoints.

## Unitree Go2

| Started | Configuration | Run | Tail reward | Full AUC | Grid |
|---|---|---|---:|---:|---:|
| 2026-08-20 21:04 | No TRS | [`2026-08-20_21-04-16`](unitree_go2_symm_flat/2026-08-20_21-04-16_go2_72d_no_trs_gait_trclosed_v2/) | 39.493 | 32.084 | 60/60 |
| 2026-08-21 10:22 | m0.1/v0.05, w500/r0 | [`2026-08-21_10-22-39`](unitree_go2_symm_flat/2026-08-21_10-22-39_go2_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2/) | 37.919 | 31.281 | 60/60 |
| 2026-08-21 22:50 | m0.2/v0.1, w500/r0 | [`2026-08-21_22-50-01`](unitree_go2_symm_flat/2026-08-21_22-50-01_go2_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2/) | 35.678 | 27.502 | 60/60 |
| 2026-08-22 10:35 | m0.2/v0.1, w500/r1000 linear | [`2026-08-22_10-35-19`](unitree_go2_symm_flat/2026-08-22_10-35-19_go2_72d_trs_m0p2_v0p1_w500_r1000_linear_gait_trclosed_v2/) | 37.092 | 29.953 | 60/60 |

Observed status: the ramp recovers part of the high-dose reward loss, and TRS
improves selected leg-allocation measures, but no TRS checkpoint matches the
no-TRS reward. Go2 training therefore remains open for reward improvement.

## Analysis reproduction

From the repository root:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\gait_famili_v2_analysis\reproduce.py
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\gait_famili_v2_analysis\reproduce.py
```

The per-run TensorBoard event is the authoritative training log. Each retained
run also includes resolved parameters, training-source provenance, deployment
exports, playback artifacts, the terminal checkpoint, and the raw fixed-grid
cell data used by the reports.
