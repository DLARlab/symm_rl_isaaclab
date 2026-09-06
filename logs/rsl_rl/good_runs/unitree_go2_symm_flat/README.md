# Unitree Go2 curated runs

Actor TRS V5 is the current Unitree Go2 archive. It contains the no-TRS
baseline, three actor-only coefficient treatments with hard post-warmup
activation, and two matched linear-ramp treatments. Every training directory
publishes all non-checkpoint artifacts and only its terminal checkpoint,
`model_19999.pt`.

## Current Actor TRS V5 cohort

All six seed-43 policies were trained for 20,000 iterations with 512
environments and evaluated at evaluation seed 42 on the same 60-cell full-V3
grid: 10 gait rows at `vx = +/-{0.5, 1.0, 1.5} m/s`. Actor treatments use the
Milestone-4 Go2 recovery profile, command validity, and zero TR critic
consistency; TR minibatch augmentation is disabled. The r500 treatments ramp
linearly from zero after the 500-iteration warmup to full strength at
iteration 1,000.

- [NoTRS](2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Actor-m0.1, v0, w500/r0](2026-09-03_00-14-58_m5_go2_actor_only_trs_m0p1_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Actor-m0.2, v0, w500/r0](2026-09-03_12-49-47_m5_go2_actor_only_trs_m0p2_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Actor-m0.3, v0, w500/r0](2026-09-04_00-27-07_m5_go2_actor_only_trs_m0p3_v0_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Actor-m0.1, v0, w500/r500](2026-09-04_23-04-00_m5_go2_actor_only_trs_m0p1_v0_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Actor-m0.2, v0, w500/r500](2026-09-04_23-04-13_m5_go2_actor_only_trs_m0p2_v0_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
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
| NoTRS | 32.534 | 40.528 | 0.03439 | **0.05904** | **88.803%** | 26.651% | 28.274% | 13.557% | 19.102% |
| Actor-m0.1 | **32.779** | **41.156** | 0.03900 | 0.06061 | **88.354%** | 19.893% | 22.666% | 10.152% | 17.209% |
| Actor-m0.2 | 32.697 | 40.898 | 0.03916 | 0.05953 | 88.057% | 18.178% | 20.534% | 19.530% | 21.524% |
| Actor-m0.3 | 32.387 | 40.355 | **0.03335** | 0.06124 | 86.983% | 22.019% | 25.826% | 14.908% | 23.473% |
| Actor-m0.1-r500 | 32.562 | 40.366 | 0.03500 | 0.06158 | 87.972% | **16.796%** | **19.757%** | 10.874% | **16.399%** |
| Actor-m0.2-r500 | 32.549 | 40.993 | 0.04028 | **0.05907** | 88.101% | 19.817% | 23.060% | **7.576%** | 18.199% |

Bold values identify the best actor-only treatment in each column; the bold
NoTRS entries are overall cohort leaders. Actor-m0.1-r0 has the strongest
observed learning, while Actor-m0.3 has the lowest forward-velocity RMSE.
NoTRS retains the best yaw RMSE and gait agreement overall.

Ramp-up is not uniformly beneficial. At `m0.1`, r500 lowers both learning
summaries and slightly lowers gait agreement relative to r0, while improving
forward tracking and three of four load metrics. At `m0.2`, r500 slightly
lowers AUC but raises tail reward, improves yaw tracking, gait agreement, work,
and GRF balance, and worsens forward tracking and both torque-balance metrics.
Both r500 treatments improve all four load metrics relative to no TRS.

Actor-m0.1-r0 has one reproducible metric-complete terminated evaluation cell,
`gait_08_gallop__vx_neg_0p5__seed_0042`, caused by `calf_height`. Its available
post-settle data are retained under the declared outcome-free aggregation and
the termination remains explicit in the comparison outputs.

Initial actor, critic, and optimizer hashes and all declared matched
signatures agree across the six runs. The baseline and actor treatments—and
the r0 and r500 captures—were nevertheless produced from different recorded
source revisions or dirty-tree snapshots. Their deltas are therefore
descriptive rather than strict causal estimates.

## Reproduction

Windows PowerShell:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\actor_trs_v5_analysis\reproduce.py
```

Linux:

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs/unitree_go2_symm_flat/actor_trs_v5_analysis/reproduce.py
```

Generated reports and provenance files may retain absolute paths and the
former branch name captured during training. Preserve those historical
strings; the relative links above are authoritative for this archive.

Historical raw evidence is intentionally unpublished. Its conclusions and
limitations remain documented in Milestones 1–4 at the parent archive level;
no `legacy/` directory is part of Actor TRS V5.
