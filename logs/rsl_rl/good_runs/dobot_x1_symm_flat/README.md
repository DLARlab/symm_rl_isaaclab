# Dobot X1 curated runs

Actor TRS V5 is the current Dobot X1 archive. It contains the no-TRS baseline
and three actor-only TRS coefficient treatments. Every training directory
publishes all non-checkpoint artifacts and only its terminal checkpoint,
`model_19999.pt`.

## Current Actor TRS V5 cohort

All four seed-42 policies were trained for 20,000 iterations with 512
environments and evaluated at evaluation seed 42 on the same 60-cell full-V3
grid: 10 gait rows at `vx = +/-{0.5, 1.0, 1.5} m/s`. TR critic consistency and
TR minibatch augmentation are disabled in all three actor treatments.

- [NoTRS](2026-08-25_06-08-02_notrs_x1def_s42/)
- [Actor-m0.1, v0, w500/r0](2026-09-03_00-15-11_m5_x1_actor_only_trs_m0p1_v0_w500_r0_x1def_s42/)
- [Actor-m0.2, v0, w500/r0](2026-09-03_11-19-54_m5_x1_actor_only_trs_m0p2_v0_w500_r0_x1def_s42/)
- [Actor-m0.3, v0, w500/r0](2026-09-03_23-50-10_m5_x1_actor_only_trs_m0p3_v0_w500_r0_x1def_s42/)
- [Milestone 5 audit](../MILESTONE_5_ACTOR_TRS_V5.md)
- [Comparison report and reproducible outputs](actor_trs_v5_analysis/REPORT.md)

## Performance summary

`AUC` is the full 20,000-iteration learning-curve area and `Tail` is the mean
reward over the last 1,000 iterations. Evaluation metrics are family-balanced
across all 60 cells. Lower is better for velocity RMSE and the four
front/hind imbalance columns; higher is better for AUC, Tail, and gait
agreement.

| Run | AUC | Tail | vx RMSE | yaw RMSE | Gait | Torque^2 | Norm. torque^2 | Work | GRF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoTRS | 34.667 | 42.307 | 0.03136 | 0.03559 | 96.747% | 17.687% | 26.149% | 23.917% | 6.999% |
| Actor-m0.1 | **35.652** | **42.729** | 0.03232 | **0.02910** | 95.575% | 7.764% | 16.743% | 13.028% | 4.748% |
| Actor-m0.2 | 32.918 | 42.066 | 0.03222 | 0.03747 | 95.626% | **5.148%** | 15.966% | **7.466%** | **3.668%** |
| Actor-m0.3 | 32.418 | 41.398 | **0.03163** | 0.03547 | **95.735%** | 5.468% | **11.276%** | 12.849% | 4.024% |

Bold values identify the best actor-only treatment in each column, not a
statistically confirmed winner. Actor-m0.1 is the only actor treatment that
exceeds no TRS on both AUC and tail reward, and it has the lowest yaw RMSE.
All three actor treatments improve all four load-balance metrics relative to
no TRS. NoTRS retains slightly lower forward-velocity RMSE and the highest gait
agreement overall.

Initial actor, critic, and optimizer hashes and all declared matched
signatures agree across the four runs. The baseline and actor treatments were
nevertheless produced from different recorded source revisions or dirty-tree
snapshots, so the comparison report correctly treats their deltas as
descriptive rather than a strict causal coefficient ablation.

## Reproduction

Windows PowerShell:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\actor_trs_v5_analysis\reproduce.py
```

Linux:

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs/dobot_x1_symm_flat/actor_trs_v5_analysis/reproduce.py
```

Generated reports and provenance files may retain absolute paths and the
former branch name captured during training. Preserve those historical
strings; the relative links above are authoritative for this archive.

Historical raw evidence is intentionally unpublished. Its conclusions and
limitations remain documented in Milestones 1–4 at the parent archive level;
no `legacy/` directory is part of Actor TRS V5.
