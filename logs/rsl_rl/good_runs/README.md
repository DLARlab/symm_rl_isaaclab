# Curated symmetric-locomotion runs

This directory is the publication archive for the retained Unitree Go2 and
Dobot X1 symmetric-locomotion studies. Actor TRS V5 is the current archive. It
contains the evidence needed to inspect each retained policy and reproduce the
committed comparisons: resolved configuration and command metadata,
training-time source provenance, TensorBoard events, terminal checkpoints,
exports, rollout data, evaluation grids, tables, figures, and reports.

## Milestone timeline

| Milestone | Date | Principal change |
| --- | --- | --- |
| [1: 60D to 72D](MILESTONE_1_60D_TO_72D.md) | 2026-07-21 | Expanded the observation and straight-line controller contract and documented the original TRS comparison. |
| [2: Phase Mapping V2 and leg-permutation fix](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md) | 2026-08-03 to 2026-08-21 | Separated command direction from time reversal, added the duty-aware TR map, and later restricted the leg-permutation reward to truly synchronous pairs. |
| [3: Gait-family V2](MILESTONE_3_GAIT_FAMILY_V2.md) | 2026-08-24 | Introduced the closed ten-gait family, family-balanced sampling, and the fixed 10-gait by 6-speed evaluation. |
| [4: Gait-closure parameter V4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md) | 2026-08-25 to 2026-09-01 | Screened hard and ramped combined actor/critic TRS schedules, added the full-V3 comparison, and recovered Go2 reward under the documented recovery profile. |
| [5: Actor TRS V5](MILESTONE_5_ACTOR_TRS_V5.md) | 2026-09-02 to 2026-09-05 | Made actor-only TRS the default, audited policy coefficients and Go2 ramp-up, and retained critic consistency as an optional ablation. |

Every milestone states what changed, what remained stable, which evidence
supports the result, and which limitations prevent a broader claim.

## Current archive layout

```text
good_runs/
  MILESTONE_1_*.md ... MILESTONE_5_*.md
  dobot_x1_symm_flat/
    README.md
    <4 current training directories>
    actor_trs_v5_analysis/
  unitree_go2_symm_flat/
    README.md
    <6 current training directories>
    actor_trs_v5_analysis/
```

The ten current training directories comprise one no-TRS baseline and three
actor-only treatments for X1, plus one no-TRS baseline and five actor-only
treatments for Go2. Their exact identities are listed in the robot indexes.

Historical evidence formerly stored under `legacy/` is intentionally not
published on this branch. Its findings, implementation boundaries, and
limitations remain summarized in Milestones 1–4; the raw legacy directories
are not part of the Actor TRS V5 archive.

Generated per-run reports may retain absolute checkpoint paths or the former
publication-branch name recorded on the capture host. Those strings are
immutable capture-time provenance. The curated relative paths in this archive
and its robot indexes are authoritative for a downloaded checkout.

## Current Actor TRS V5 evidence

- [Milestone 5 audit](MILESTONE_5_ACTOR_TRS_V5.md)
- [Dobot X1 runs and result summary](dobot_x1_symm_flat/README.md)
- [Dobot X1 comparison report](dobot_x1_symm_flat/actor_trs_v5_analysis/REPORT.md)
- [Unitree Go2 runs and result summary](unitree_go2_symm_flat/README.md)
- [Unitree Go2 comparison report](unitree_go2_symm_flat/actor_trs_v5_analysis/REPORT.md)

For X1, Actor-m0.1 has the strongest observed actor-only learning result
(`AUC 35.652`, `tail 42.729`) against the no-TRS values `34.667` and `42.307`.
All three actor-only checkpoints reduce all four reported front/hind imbalance
metrics, while no TRS retains the highest gait agreement. For Go2,
Actor-m0.1-r0 has the highest AUC and tail reward (`32.779`, `41.156`), while
the r500 treatments expose metric-dependent ramp-up tradeoffs. Both Go2 r500
treatments improve all four load-imbalance metrics relative to no TRS, but no
single treatment leads learning, tracking, gait fidelity, and load balance.

These are single-training-seed development results. Initial policy, critic,
and optimizer states and the declared matched experimental signatures agree
within each robot, but training source provenance differs across runs.
Baseline deltas are therefore descriptive rather than strict single-factor
causal estimates.

## Reproduce from a fresh clone

Follow the repository [installation and environment setup](../../../README.md),
including the matching Isaac Lab/Isaac Sim runtime. Then regenerate the
committed Actor TRS V5 tables and figures through the Isaac Lab Python wrapper.

Windows PowerShell:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\actor_trs_v5_analysis\reproduce.py
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\actor_trs_v5_analysis\reproduce.py
```

Linux:

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs/dobot_x1_symm_flat/actor_trs_v5_analysis/reproduce.py
./isaaclab.sh -p logs/rsl_rl/good_runs/unitree_go2_symm_flat/actor_trs_v5_analysis/reproduce.py
```

The comparison validates archived run identities, terminal checkpoints,
TensorBoard scalar inputs, full-V3 metrics, manifests, and provenance before
writing outputs.

To play an archived policy, pass its checkpoint path directly, for example:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43\model_19999.pt
```

## Checkpoint publication policy

Each of the ten current training directories publishes every non-checkpoint
artifact and exactly one checkpoint: `model_19999.pt`. Intermediate
`model_*.pt` checkpoints remain local and ignored. When adding a completed
run, update the exact allow-list and verify that precisely one checkpoint from
that run is staged.

Do not remove retained event files, resolved configurations, source
provenance, exports, rollout data, evaluation inputs, reports, or terminal
checkpoints. Together they bind the published metrics to the archived policy
and are required for independent inspection.
