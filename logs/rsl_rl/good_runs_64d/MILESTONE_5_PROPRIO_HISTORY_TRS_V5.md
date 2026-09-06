<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Milestone 5: Proprioceptive History TRS V5

**Date:** 2026-09-06

**Branch:** `jding/proprio-history-trs-v5`

**Archive contract:** `good_runs_64d`

## Scope

This milestone records the implementation and evidence-storage boundary for
the history-aware V5 experiment. It does not yet designate a good run or make
an empirical performance claim.

The policy uses the immutable `hardware_proprio_history_64d_v1` frame contract.
Each frame contains 64 proprioceptive, command, gait, and two-action-history
values. Native Isaac Lab history stores 30 oldest-to-newest frames and flattens
them term-major for a 1920D MLP input. The critic receives the same observation
and has no privileged velocity or recurrent state.

## Principal changes

- Replaced the prior 72D policy input with a 64D hardware-oriented frame and a
  default 30-frame causal history.
- Reconstructed transition-aligned reverse histories from authentic future
  rollout records instead of fabricating unavailable reversed history.
- Kept reverse candidates outside PPO rollout storage, likelihood ratios,
  advantages, returns, value targets, KL inputs, and entropy samples.
- Added strict episode, command, gait, disturbance, history-completeness, and
  policy-version validity boundaries for auxiliary samples.
- Kept the existing shared `scripts/symm_locomotion/` launcher surface for both
  robots and separated this branch's curated evidence into `good_runs_64d`.

The complete algorithm and observation contract are documented in
[`docs/symm_locomotion/PROPRIO_HISTORY_TRS_V5.md`](../../../docs/symm_locomotion/PROPRIO_HISTORY_TRS_V5.md).

## Utility contract

Training, playback, recording, full-V3 evaluation, comparison, TensorBoard,
ablation, and scheduling use the same shared launcher families as the 72D
Actor TRS V5 branch. History-specific CLI options and provenance fields are
additions to that common surface. The branch also carries the same strict
complete-cycle future-versus-past return diagnostic; actor-only publication
plot inventories remain tied to their own archived run cohort.

The shared training defaults match Actor TRS V5: policy mirror consistency is
enabled at `0.1`, auxiliary critic consistency defaults to `0.0`, and standard
PPO value regression remains enabled. A positive `--tr-value-coef` is an
explicit critic-consistency ablation.

On this branch, new comparisons search these Go2 roots by default, in order:

1. `logs/rsl_rl/unitree_go2_symm_flat`
2. `logs/rsl_rl/good_runs_64d/unitree_go2_symm_flat`

Historical 72D reproduction commands must pass
`logs/rsl_rl/good_runs/unitree_go2_symm_flat` explicitly.

## 72D parity audit

The Go2 and Dobot X1 robot models, actuator settings, ordered joints, contact
sensors, flat-scene physics, gait command ranges, action geometry, domain
randomization magnitudes, termination limits, and robot-specific reward
arguments match the 72D Actor TRS V5 branch.

The remaining environment differences are required by this branch's declared
experiment: the 64D observation/history contract, causal disturbance-boundary
tracking, and a straight-line reward that excludes absolute world lateral
position and heading recovery (`pose_weight=0.0`). The 72D branch retains those
world-pose recovery terms (`pose_weight=0.30`). These differences must not be
silently overwritten when synchronizing reusable utilities.

## Evidence status

No training directory is identified as a retained good run at this milestone.
The robot indexes therefore report an empty inventory. When a run is promoted,
its terminal checkpoint and all claim-bearing provenance, evaluation, and
analysis inputs must be committed together.

Required evidence includes:

- resolved environment and agent configuration;
- resolved launch command, cohort metadata, and initialization provenance;
- the TensorBoard event file used for learning-curve claims;
- the terminal checkpoint and policy exports;
- full-V3 study, progress, metrics, and analysis provenance;
- comparison manifests, tables, figures, report, and reproducer; and
- a milestone update distinguishing observations from causal claims.

## Limitations

- The 64D archive label records the instantaneous observation-frame width, not
  the flattened network input width. The default 30-frame policy input remains
  1920D, and every curated run must record its history length explicitly.
- The `--no-history` mode shares the 64D frame contract but is a distinct
  ablation with a 64D effective input. Keep its evidence and claims separate
  from the default 30-frame cohort.
- Implementation tests establish invariants; they do not establish locomotion
  quality, robustness, hardware transfer, or statistical significance.
- Future single-seed development runs must be described as descriptive evidence
  unless a predeclared replicated design supports stronger inference.
