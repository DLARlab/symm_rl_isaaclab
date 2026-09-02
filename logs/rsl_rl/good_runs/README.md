# Curated symmetric-locomotion runs

This directory is the publication archive for the retained Unitree Go2 and
Dobot X1 symmetric-locomotion studies. It contains the evidence needed to
inspect a trained policy and reproduce the committed comparisons: resolved
configuration and command metadata, training-time source provenance,
TensorBoard events, terminal checkpoints, exports, rollout data, evaluation
grids, tables, figures, and reports.

## Milestone timeline

| Milestone | Date | Principal change |
| --- | --- | --- |
| [1: 60D to 72D](MILESTONE_1_60D_TO_72D.md) | 2026-07-21 | Expanded the observation and straight-line controller contract and documented the original TRS comparison. |
| [2: Phase Mapping V2 and leg-permutation fix](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md) | 2026-08-03 to 2026-08-21 | Separated command direction from time reversal, added the duty-aware TR map, and later restricted the leg-permutation reward to truly synchronous pairs. |
| [3: Gait-family V2](MILESTONE_3_GAIT_FAMILY_V2.md) | 2026-08-24 | Introduced the closed ten-gait family, family-balanced sampling, and the fixed 10-gait by 6-speed evaluation. |
| [4: Gait-closure parameter V4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md) | 2026-08-25 to 2026-09-01 | Screened hard and ramped TRS schedules, added the full-V3 comparison, and recovered Go2 reward under the documented recovery profile. |

Every milestone states what changed, what remained stable, which evidence
supports the result, and which limitations prevent a broader claim.

## Archive layout

```text
good_runs/
  MILESTONE_1_*.md ... MILESTONE_4_*.md
  dobot_x1_symm_flat/
    2026-08-25_* ... 2026-08-27_*       # current V4 training cohort
    gait_closure_parameter_v4_analysis/ # current comparison
    legacy/
      phase_mapping_v2_legacy_permutation_reward/
      gait_family_v2_analysis/
  unitree_go2_symm_flat/
    2026-08-29_* ... 2026-08-31_*       # current V4 training cohort
    gait_closure_parameter_v4_analysis/ # current comparison
    legacy/
      phase_mapping_v2_legacy_permutation_reward/
      gait_family_v2_analysis/
```

The Phase Mapping V2 runs used the legacy Gaussian leg-permutation reward.
They validate the phase/TR mapping study, not the later synchronized-pair
reward fix. Milestone 2 records this evidence boundary explicitly.

Generated per-run reports may retain absolute checkpoint paths recorded on the
capture host. Those strings are capture-time provenance; the curated relative
paths in this archive and its robot indexes are authoritative for a downloaded
checkout.

## Current V4 evidence

- [Dobot X1 runs and result summary](dobot_x1_symm_flat/README.md)
- [Dobot X1 comparison report](dobot_x1_symm_flat/gait_closure_parameter_v4_analysis/REPORT.md)
- [Unitree Go2 runs and result summary](unitree_go2_symm_flat/README.md)
- [Unitree Go2 comparison report](unitree_go2_symm_flat/gait_closure_parameter_v4_analysis/REPORT.md)

For X1, low-r0 has the strongest observed learning while all four TRS
treatments improve every reported front/hind imbalance domain relative to the
no-TRS checkpoint. For Go2, low-r0 has the strongest learning, and high-r500
exceeds no TRS in both full AUC and tail reward while improving all four
front/hind imbalance domains. Tracking, gait fidelity, and load metrics still
trade off. These are single-training-seed development results.

## Reproduce from a fresh clone

Follow the repository [installation and environment setup](../../../README.md),
including the matching Isaac Lab/Isaac Sim runtime. Then regenerate the
committed V4 tables and figures through the Isaac Lab Python wrapper.

Windows PowerShell:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
```

Linux:

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs/dobot_x1_symm_flat/gait_closure_parameter_v4_analysis/reproduce.py
./isaaclab.sh -p logs/rsl_rl/good_runs/unitree_go2_symm_flat/gait_closure_parameter_v4_analysis/reproduce.py
```

The V4 comparison validates the archived run identity, terminal checkpoint,
TensorBoard scalar input, full-V3 metrics, manifest, and provenance before
writing outputs. The Milestone 3 Go2 folder carries its pinned historical
analysis engine, while the Milestone 2 phase-mapping wrappers retain an
archival notice and the exact detached commit required for their original
regeneration.

Archived policy playback must use the historical worktree recorded in that
run's provenance. From that worktree, pass the checkpoint path directly, for
example:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43\model_19999.pt
```

## Checkpoint publication policy

Every curated training directory publishes all non-checkpoint artifacts but
only the numerically latest iteration checkpoint. All 25 current and legacy
training directories end at `model_19999.pt`; those 25 files are explicitly
allowed by [`.gitignore`](.gitignore). The 500 intermediate checkpoints remain
local and ignored. When adding a completed run, update the exact allow-list and
verify that precisely one `model_*.pt` path from that run is staged.

The three legacy X1 gait-family runs also retain checksum-only initialization
records. These preserve the observed `model_0.pt` digest for provenance while
stating explicitly that the initialization checkpoint bytes are not
published and cannot be independently rehashed from a clone.

Do not remove the retained event file, resolved configuration, source
provenance, evaluation inputs, or terminal checkpoint. Together they bind the
published metrics to the archived policy and are required for independent
inspection.
