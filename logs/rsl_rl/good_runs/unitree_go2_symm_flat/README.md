# Unitree Go2 curated runs

This directory separates the current gait-closure parameter V4 cohort from
the historical Phase Mapping V2 and gait-family V2 evidence. Every training
directory publishes its complete non-checkpoint artifacts and only its latest
checkpoint, `model_19999.pt`.

## Current gait-closure parameter V4 cohort

All five seed-43 policies were trained for 20,000 iterations and evaluated on
the same 60-cell full-V3 grid: 10 gait rows at
`vx = +/-{0.5, 1.0, 1.5} m/s`.

- [No TRS](2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Low TRS, m0.1/v0.05, w500/r0](2026-08-30_09-45-19_trs_m0p1_v0p05_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Low TRS, m0.1/v0.05, w500/r500](2026-08-31_00-14-06_trs_m0p1_v0p05_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [High TRS, m0.2/v0.1, w500/r0](2026-08-30_09-45-34_trs_m0p2_v0p1_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [High TRS, m0.2/v0.1, w500/r500](2026-08-31_00-14-42_trs_m0p2_v0p1_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43/)
- [Comparison report and reproducible outputs](gait_closure_parameter_v4_analysis/REPORT.md)

The Milestone 3 reward regression is no longer present in this coefficient
screen. Low-r0 has the best full learning-curve AUC (`33.051`) and tail
reward (`41.189`), versus `32.534` and `40.528` for no TRS. High-r500 also
exceeds the no-TRS AUC and tail reward while reducing all four reported
front/hind imbalance domains. High-r0 gives the strongest overall leg-usage
balance but retains a reward cost. Tracking is similar and gait agreement is
slightly lower for every TRS policy.

Milestone 4 also changes the Go2 reward/configuration baseline and uses seed 43
instead of Milestone 3's seed 42, so these deltas are descriptive and cannot be
attributed to TRS coefficients alone.

Run the analysis wrapper with the Isaac Lab Python environment:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
```

## Historical evidence

- [Phase Mapping V2 and legacy permutation-reward archive](legacy/phase_mapping_v2_legacy_permutation_reward/REPORT.md)
- [Gait-family V2 archive](legacy/gait_family_v2_analysis/REPORT.md)

The Phase Mapping archive predates the synchronized-pair leg-permutation
reward fix and must not be used as empirical evidence for that later reward.
The gait-family V2 archive contains the fixed-grid cohort that established the
Milestone 3 baseline.
