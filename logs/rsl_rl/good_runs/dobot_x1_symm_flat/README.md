# Dobot X1 curated runs

This directory separates the current gait-closure parameter V4 cohort from
the historical Phase Mapping V2 and gait-family V2 evidence. Every training
directory publishes its complete non-checkpoint artifacts and only its latest
checkpoint, `model_19999.pt`.

## Current gait-closure parameter V4 cohort

All five seed-42 policies were trained for 20,000 iterations and evaluated on
the same 60-cell full-V3 grid: 10 gait rows at
`vx = +/-{0.5, 1.0, 1.5} m/s`.

- [No TRS](2026-08-25_06-08-02_notrs_x1def_s42/)
- [Low TRS, m0.1/v0.05, w500/r0](2026-08-25_16-56-41_trs_m0p1_v0p05_w500_r0_x1def_s42/)
- [Low TRS, m0.1/v0.05, w500/r1000](2026-08-26_05-12-37_trs_m0p1_v0p05_w500_r1000_x1def_s42/)
- [High TRS, m0.2/v0.1, w500/r0](2026-08-27_00-44-04_trs_m0p2_v0p1_w500_r0_x1def_s42/)
- [High TRS, m0.2/v0.1, w500/r1000](2026-08-27_23-34-38_trs_m0p2_v0p1_w500_r1000_x1def_s42/)
- [Comparison report and reproducible outputs](gait_closure_parameter_v4_analysis/REPORT.md)

Low-r0 has the best full learning-curve AUC (`36.012`) and tail reward
(`43.182`), compared with `34.667` and `42.307` for no TRS. Every TRS policy
reduces the family-balanced front/hind imbalance of raw torque squared,
normalized torque squared, work, and vertical GRF relative to no TRS, although
different schedules lead individual domains. Tracking remains similar and
gait agreement is slightly lower. These are descriptive checkpoint results,
not multi-seed causal estimates.

Run the analysis wrapper with the Isaac Lab Python environment:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
```

## Historical evidence

- [Phase Mapping V2 and legacy permutation-reward archive](legacy/phase_mapping_v2_legacy_permutation_reward/REPORT.md)
- [Gait-family V2 archive](legacy/gait_family_v2_analysis/REPORT.md)

The Phase Mapping archive predates the synchronized-pair leg-permutation
reward fix and must not be used as empirical evidence for that later reward.
The gait-family V2 archive contains the fixed-grid cohort that established the
Milestone 3 baseline.
