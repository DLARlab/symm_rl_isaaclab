<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Proprioceptive History TRS V5

V5 is a hardware-oriented, history-aware time-reversal consistency objective
for gait-conditioned hybrid locomotion. It retains the V4 gait library,
forward-time gait clock, duty-aware phase reflection, reward boundary, and
joint-target safety path. Its default consistency mapping reconstructs an exact
causal reversed observation sequence from authentic future decisions; it does
not augment PPO transitions.

## Policy observation contract

The immutable contract identifier is `hardware_proprio_history_64d_v1`. One
instantaneous frame is exactly 64 values:

| Term | Dimension | Slice | Scale | Time reversal |
| --- | ---: | --- | --- | --- |
| Projected gravity | 3 | `[0:3]` | `(1, 1, 1)` | even |
| Body velocity command `(vx, vy, wz)` | 3 | `[3:6]` | `(2, 2, 0.25)` | odd |
| Relative joint position | 12 | `[6:18]` | `1` | even |
| Joint velocity | 12 | `[18:30]` | `0.05` | odd |
| Previous action | 12 | `[30:42]` | `1` | reconstructed |
| Second-previous action | 12 | `[42:54]` | `1` | reconstructed |
| Dimensionless gait period | 1 | `[54:55]` | `1` | even |
| Stance duty factor | 1 | `[55:56]` | `1` | even |
| Foot-phase sine | 4 | `[56:60]` | `1` | duty-reflected |
| Foot-phase cosine | 4 | `[60:64]` | `1` | duty-reflected |

Measured base linear/angular velocity, absolute heading, expanded desired
twist, gait-theta features, phase ratios, and world pose are not policy inputs.
Simulator-only velocity, contact, disturbance, and segment metadata may gate or
weight auxiliary samples, but are never concatenated into actor or critic
observations. Actor and critic receive the same policy observation; there is no
privileged critic velocity and no recurrent network state.

Projected gravity describes tilt relative to gravity and contains no absolute
yaw. The scalar motion reward therefore uses body-frame forward tracking,
lateral-velocity and yaw-rate stabilization, roll, pitch, and base-height or
support quality rather than absolute cross-track or heading recovery.

## Action history and period normalization

At decision `t`, write the physical, non-action features as `z_t` and the raw
sampled/executed joint-position-offset action as `a_t`. The policy frame is

```text
y_t = (z_t, a_{t-1}, a_{t-2}).
```

After `a_t` executes, Isaac Lab exposes it as `ActionManager.action` and exposes
`a_{t-1}` as `ActionManager.prev_action`; the next returned frame is therefore
`y_{t+1} = (z_{t+1}, a_t, a_{t-1})`. Selected-environment resets clear both
action tensors. The action time-reversal operator is the identity for this
affine joint-position-offset controller.

The period feature is dimensionless:

```text
T* = T sqrt(g / L)
```

where `T` is the active period [s], `g = 9.81 m/s^2`, and `L` is the midpoint
of `base_height_range` [m]. Non-finite or non-positive periods and lengths are
rejected.

## Native history packing and inference

New runs default to native Isaac Lab observation history with `H=30` and
`flatten_history_dim=true`, so the unchanged MLP receives `30 * 64 = 1920`
values. `--no-history` uses `H=1` for sequence reconstruction and presents only
the latest 64-D frame to the policy.

Packing is `term_major_oldest_to_newest_flattened`: each observation term
stores its oldest sample first and newest sample last, and flattened term
histories are concatenated in contract-term order. Task-local pack, unpack, and
latest-frame helpers preserve arbitrary leading dimensions, dtype, and device.
Reset clears selected native histories, and the first post-reset append fills
them with the new initial frame rather than pre-reset data.

Deployment remains strictly causal. It keeps only the previous two executed
actions and the native past-observation history. Future actions are required
only while constructing training-time reverse targets from already executed
rollout records.

## Duty-aware physical feature transform

Let stance duty factor be `beta`, `r = 1 - beta`, and a foot phase be `psi`.
The physical transform uses

```text
psi_TR = remainder(r - psi, 1)
alpha  = 2 pi r
s_TR   = sin(alpha) c - cos(alpha) s
c_TR   = cos(alpha) c + sin(alpha) s.
```

`time_reverse_physical_policy_features` preserves projected gravity, joint
position, period, and duty factor; negates the three command values and joint
velocity; and transforms all four phase pairs. It deliberately does not decide
the two action-history slots. `build_reversed_causal_policy_frame` applies that
physical map to `y_j`, then overwrites the slots with `(a_j, a_{j+1})`:

```text
ybar_j = (R_Z z_j, R_A a_j, R_A a_{j+1}),  R_A = identity.
```

The phase map is involutive, preserves unit-circle norm, and preserves contact
mode away from event boundaries. Liftoff and touchdown boundaries exchange.
This is an auxiliary map only: commanded backward locomotion still advances
the ordinary gait clock forward.

## Transition-aligned causal sequence

An authentic forward edge and its reverse pairing are

```text
x_t --a_t--> x_{t+1}
R_X x_{t+1} --R_A a_t--> R_X x_t.
```

For the edge at `t`, the default reverse policy input is associated with the
successor. From forward frames `y_{t+1}, ..., y_{t+H}` and forward actions
through `a_{t+H+1}`, the builder constructs

```text
Hbar_{t+1} = (ybar_{t+H}, ..., ybar_{t+1}),
```

which is oldest-to-newest in reversed time and is repacked in the native
term-major layout. It needs `H+2` consecutive decision records: three for
`H=1`, and 32 for the default `H=30`. Because the PPO rollout length is 24,
the task-local sequence buffer intentionally spans rollout boundaries; the PPO
rollout length is not increased to manufacture a candidate.

The canonical modes are:

- `transition_aligned_sequence` (default): exact causal reconstruction with
  mapping version `transition_aligned_causal_sequence_v1`.
- `framewise_feature_approx`: ablation-only instantaneous physical transforms
  in unchanged history order with frozen visible action-history slots. It is
  explicitly approximate and logs that history was not causally reversed.
- `none`: no actor/value time-reversal consistency objective.

The deprecated `history_trs_mode=framewise_feature` configuration alias maps to
`framewise_feature_approx` with a warning. It is not the default for new runs.

## Auxiliary objectives

The source/target alignment is deliberately asymmetric across the edge:

```text
L_TR_policy = masked_mean(
    ||mu(Hbar_{t+1}) - R_A stop_gradient(mu(H_t))||^2
)

L_TR_value = masked_mean(
    (V(Hbar_{t+1}) - stop_gradient(V(H_{t+1})))^2
).
```

Both actor means are recomputed with the current actor; an optional normalized
joint-target coordinate is applied to both means. The actor source is `H_t`.
The value source is `H_{t+1}`, never `H_t`. Stopped targets prevent the source
branch from receiving auxiliary gradients.

These losses are auxiliary only. Reverse candidates never duplicate or replace
old log probabilities, likelihood ratios, advantages, returns, value targets,
clipped PPO value terms, adaptive KL inputs, or entropy samples.
`use_data_augmentation=false` remains mandatory for this path. Authentic PPO
quantities are captured before any auxiliary actor forward can update cached
distribution parameters. Zero coefficients or an empty valid-candidate set
produce exact zero without extra actor/critic forwards.

## Sequence buffer and validity

The bounded task-local sequence buffer is independent of PPO rollout storage
and the model-based sidecar. Per environment it records the full `H_t`, latest
64-D frame, raw action `a_t`, episode/command/gait/disturbance segment IDs,
collection update, transition validity, done, timeout, and detached
simulator-only diagnostics. It is active only when exact consistency or its raw
diagnostics need candidates. Transient contents are cleared on fresh reset and
checkpoint load and are never checkpointed.

Distributed PPO fails closed when exact sequence consistency is active because
the task-local candidates are not synchronized across ranks.

A candidate requires a finite, complete native history and a contiguous window
within one episode, command/task segment, gait segment, disturbance generation,
and permitted policy-version span. It rejects done, timeout, reset, command or
gait changes, pushes, incomplete warm-up, and an unexecuted final future action.
The original `H_t` must itself be task-homogeneous. Relative gait offsets are
inferred from phase ratios and are invariant to common phase.

Measured body-frame linear velocity/yaw rate, uprightness, contact impulse,
slip, reverse residual, saturation, and push generation remain detached
validity or confidence metadata. Command tracking compares measurements to the
command with absolute and relative tolerances; it is not an alias for command
magnitude. Changing simulator-only metadata cannot change actor or critic
inputs. Accepted samples have maximum age one update and span at most one
policy-version update, both recorded in checkpoint and run metadata.

The separately learned model-based sidecar remains experimental and independent
of this buffer. It supports instantaneous (`H=1`) policies only. Enabling it
with native history remains a startup error.

## Staged TR-orbit curriculum

`command_curriculum_mode=tr_orbit_reward_threshold_v1` is optional and disabled
by default. For each gait row and signed forward-speed bin it stores:

- `eligible` and `mastered` Boolean tensors;
- sampling `priority`;
- success EWMA and visit count.

Every tensor is closed over the authoritative gait/velocity time-reversal
orbit. At initialization, only nonzero-gait-prior cells whose speed-bin center
satisfies `abs(v_x) <= curriculum_initial_max_abs_speed` are eligible; their
partners are made eligible atomically. Locked cells use
`curriculum_locked_cell_weight` (zero by default). The exploration floor is
applied only to eligible cells, and zero-prior gait rows never activate.

Outcomes in one update are first aggregated by canonical orbit. One EWMA update
uses the orbit's mean success and visits increase by the number of samples, so
permuting outcomes is invariant. Duplicating all outcomes leaves the EWMA
update unchanged and doubles the visit-count increment. If both increments
remain on the same side of `curriculum_min_visits`, eligibility, mastery, and
priority are also unchanged. At the threshold, however, the required
`visits_new = visits_old + sample_count` and
`mastered = visits_new >= minimum and EWMA >= threshold` equations make
universal duplication-invariant mastery mathematically impossible: a doubled
batch can cross the visit threshold when the original batch does not. The
implementation follows those explicit equations and records batch multiplicity
as part of the training protocol rather than claiming the contradictory
invariant. Once an eligible orbit is mastered, it remains eligible and is
marked exactly once. The next larger absolute-speed orbit and its partner are
then unlocked and initialized from their gait prior/exploration floor;
repeated successes do not unlock it again.

Full-resume state contains gait prior, priority, eligible/mastered masks, EWMA,
visits, curriculum RNG, semantic curriculum configuration, and absolute
training iteration. Current cells, partial segment errors, commands, gait,
timers, phase, action history, observation history, and episode state are
transient. Resume restores global curriculum competence, discards transient
state, invalidates every cell, clears segment accumulators, and requests a
fresh sample for every environment from the fresh simulator state. Legacy V5
adaptive-weight curriculum state is migrated into global competence tensors;
its stale per-environment runtime is discarded with a warning.
Distributed PPO likewise rejects an enabled command curriculum until global
competence updates and RNG state are synchronized across ranks.

## Checkpoints, launch controls, and provenance

Train, play, record, and fixed-grid evaluation share `--history`,
`--no-history`, `--history-length`, and `--tr-consistency-mode`. Training and
ablation expose every staged-curriculum setting. Generated training names use
`trseq`, `trff`, or `notr` so exact, approximate, and disabled runs cannot be
confused.

Each run stores resolved environment and agent configurations plus an immutable
`policy_contract.json`. Policy/checkpoint metadata includes the consistency
mode and mapping version, action-history length two, sequence history length,
required `H+2` records, candidate age and policy-version limits, and actor/value
edge alignment. Forwarded Hydra arguments remain available, but a value that
contradicts launcher-owned curriculum or policy metadata fails closed.

A full resume rejects incompatible observation or causal-mapping semantics.
Actor-only play/evaluation may load weights when the observation contract and
actor width match, because the training-only sequence buffer is not needed for
inference. Explicit weights-only initialization is likewise permitted. A 72-D
V4 actor is incompatible: V5 does not pad, truncate, project, or silently
migrate its inputs.

Example Windows launches:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --history --history-length 30 --tr-consistency-mode transition_aligned_sequence --tr-policy-coef 0.1 --tr-value-coef 0.05
.\scripts\symm_locomotion\train.ps1 --robot x1 --history --tr-consistency-mode framewise_feature_approx
.\scripts\symm_locomotion\train.ps1 --robot x1 --no-history --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest --history --history-length 30
```

## Physical limitations

The exactness claimed here is exact indexing and causal observation
reconstruction for the declared discrete edge—not reversibility of the robot
or simulator physics. Motor damping, friction, inelastic impacts, actuation
lag, controller saturation, observation delay, contact discontinuities,
external pushes, and termination break physical reversibility. Validity gates
and detached confidence weights reduce obvious violations but do not prove
that a candidate is dynamically realizable. The value residual remains a
task-conditioned inductive bias.

Ordinary checkpoints do not snapshot the complete simulator physics state, so
a resumed rollout is not claimed to be a bitwise continuation of contact
dynamics. The curriculum restores global learning competence and deliberately
starts fresh transient simulator/task state.
