<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Curated 64D symmetric-locomotion runs

This directory is the publication archive for runs produced by the default
proprioceptive-history policy contract on the
`64d_history_trs_v1` branch. The archive name records the
instantaneous observation-frame width: one policy frame has 64 values. The
default native history contains 30 frames, so its flattened MLP input still has
`30 * 64 = 1920` values.

Do not place historical 72D policies in this archive. Their milestones and
checkpoints remain on branch `jding/72d_actor_trs_v5` under
`logs/rsl_rl/good_runs_72d/`. That directory may be visible here as an ignored
local working copy, but it is not part of this branch's commits. A curated
`--no-history` ablation may live here because it uses the same 64D frame
contract, but its 64D effective input must remain clearly separated from the
default 1920D history cohort.

## Milestone timeline

| Milestone | Date | Principal change |
| --- | --- | --- |
| [5: Proprioceptive History TRS V5](MILESTONE_5_PROPRIO_HISTORY_TRS_V5.md) | 2026-09-06 | Introduced the 64D hardware-oriented frame, 30-frame causal history, and transition-aligned reverse-sequence objective. |

This milestone currently documents the implementation and archive contract.
It does not claim a successful training result. Add evidence-backed result
statements only after the corresponding run artifacts have been curated here.

## Archive layout

```text
good_runs_64d/
  MILESTONE_5_PROPRIO_HISTORY_TRS_V5.md
  dobot_x1_symm_flat/
    README.md
    <curated training directories>
    <comparison directories>
  unitree_go2_symm_flat/
    README.md
    <curated training directories>
    <comparison directories>
```

Routine training still writes beneath the normal robot roots:

```text
logs/rsl_rl/dobot_x1_symm_flat/
logs/rsl_rl/unitree_go2_symm_flat/
```

Copy a completed, reviewed run into the matching directory here only when it
is intentionally curated. The shared launchers under
`scripts/symm_locomotion/` remain the supported training, playback, recording,
evaluation, comparison, TensorBoard, and scheduling interfaces; this archive
changes data placement, not launcher usage.

## Curating a run

For each retained training run:

1. Confirm that its resolved policy contract is
   `hardware_proprio_history_64d_v1`, then record whether history is enabled,
   the history length, and the effective flattened policy width. The default
   milestone cohort uses history length 30 and width 1920.
2. Retain resolved configuration, command and initialization provenance,
   TensorBoard events, exports, rollout/evaluation inputs, tables, figures, and
   reports needed to audit published claims.
3. Retain only the numerically latest `model_*.pt` checkpoint. Add that exact
   relative path as a negated exception in [`.gitignore`](.gitignore); never
   unignore checkpoints with a wildcard.
4. Run the shared full-V3 evaluation and comparison utilities. Their default
   curated Go2 search root on this branch is
   `logs/rsl_rl/good_runs_64d/unitree_go2_symm_flat`.
5. Update the robot index and milestone document with observed evidence and
   explicit limitations.

The default comparison root intentionally excludes the historical 72D archive,
preventing incompatible policies from being selected by name. A historical
reproduction should run from a `jding/72d_actor_trs_v5` worktree and pass its
`good_runs_72d` path explicitly with `--run_root`.
