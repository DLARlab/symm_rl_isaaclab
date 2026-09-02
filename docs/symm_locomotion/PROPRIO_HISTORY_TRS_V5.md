<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Proprioceptive History TRS V5

V5 is a history-aware, hardware-oriented, duty-aware time-reversal actor/value
regularizer for gait-conditioned hybrid locomotion. It retains the V4 gait
library, command timing, reward boundary, joint-target safety path, and
forward-time gait clock.

## Policy observation contract

The contract identifier is `hardware_proprio_history_64d_v1`. One instantaneous
frame is exactly 64 values:

| Term | Dimension | Slice | Scale | Time reversal |
| --- | ---: | --- | --- | --- |
| Projected gravity | 3 | `[0:3]` | `(1, 1, 1)` | even |
| Body velocity command `(vx, vy, wz)` | 3 | `[3:6]` | `(2, 2, 0.25)` | odd |
| Relative joint position | 12 | `[6:18]` | `1` | even |
| Joint velocity | 12 | `[18:30]` | `0.05` | odd |
| Previous action | 12 | `[30:42]` | `1` | even |
| Second-previous action | 12 | `[42:54]` | `1` | even |
| Dimensionless gait period | 1 | `[54:55]` | `1` | even |
| Stance duty factor | 1 | `[55:56]` | `1` | even |
| Foot-phase sine | 4 | `[56:60]` | `1` | duty-reflected |
| Foot-phase cosine | 4 | `[60:64]` | `1` | duty-reflected |

Measured base linear and angular velocity, the expanded six-dimensional desired
twist, gait-theta features, the two phase ratios, and sagittal world-pose state
are not policy inputs. Simulator velocities remain available to rewards,
terminations, diagnostics, and evaluation. Actor and critic receive the same
policy observation; V5 adds no privileged critic velocity.

Projected gravity describes tilt relative to gravity. It does **not** contain
absolute yaw or world heading. V5 therefore removes absolute cross-track and
heading recovery from the scalar motion reward. The active reward uses body-frame
forward tracking, lateral-velocity and yaw-rate stabilization, roll, pitch, and
base-height/support quality. World lateral position and heading may be retained
as explicitly simulator-only diagnostics.

## Action history

After action `u_k` has executed, Isaac Lab's `ActionManager.action` is `u_k` and
`ActionManager.prev_action` is `u_{k-1}`. Accordingly:

```text
previous_action        = u_k
second_previous_action = u_{k-1}
```

A selected environment reset clears both tensors. Actions are joint-position
target offsets, so their time-reversal transform is the identity.

## Period normalization

The policy observes

```text
T* = T sqrt(g / L)
```

where `T` is the active gait period in seconds, `g = 9.81 m/s^2`, and `L` is the
midpoint of the command configuration's `base_height_range` in metres. This is
the same characteristic-length convention used by the existing speed/period
curve. The helper rejects non-finite or non-positive periods and lengths.

## Native 30-frame history

New V5 runs enable Isaac Lab `ObservationManager` history with 30 frames and
`flatten_history_dim=true`. The unchanged MLP therefore receives `30 * 64 =
1920` values. `--no-history` selects one 64-D current frame.

Native packing is `term_major_oldest_to_newest_flattened`: each observation
term stores its oldest sample first and its newest sample last, then the
flattened term histories are concatenated in contract term order. Task-local
pack, unpack, inference, and latest-frame helpers preserve arbitrary leading
dimensions, dtype, and device. Observation-manager reset clears selected
environment histories; the first post-reset append fills the history with the
new initial frame, never pre-reset data.

## Duty-aware time reversal

Let the stance duty factor be `beta`, `r = 1 - beta`, and a foot phase be `psi`.
V5 uses

```text
psi_TR = remainder(r - psi, 1)
alpha  = 2 pi r
s_TR   = sin(alpha) c - cos(alpha) s
c_TR   = cos(alpha) c + sin(alpha) s
```

The full frame transform preserves projected gravity, joint position, both
action-history values, period, and duty factor; negates all three command values
and joint velocity; and applies the formula above to all four phase pairs. It is
an involution and preserves the phase unit-circle norm. Away from boundaries it
preserves contact mode; liftoff and touchdown boundaries exchange.

This is an auxiliary feature map only. Negative commanded forward velocity does
not reverse the actual gait clock: ordinary forward and backward locomotion use
the same forward-time gait schedule.

With history, `history_trs_mode=framewise_feature` applies the instantaneous map
to every visible frame while preserving oldest-to-newest order. It is an
involutive feature-level prior whose latest frame is the transformed latest
state. It is not an exact causal history from a physically reversed rollout;
such a history would require future samples from the original rollout.

## PPO objectives

The ordinary PPO surrogate, value regression, entropy term, adaptive KL,
gradient clipping, and action-bound safeguards remain unchanged. The auxiliary
objective is

```text
L_total = L_PPO + lambda_policy L_TR_policy + lambda_value L_TR_value

L_TR_policy = masked_mean(
    ||mu(T_H(H)) - T_action(stop_gradient(mu(H)))||^2
)

L_TR_value = masked_mean(
    (V(T_H(H)) - stop_gradient(V(H)))^2
)
```

The command mask reads the latest frame's scaled forward command through the
observation schema, for either 64 or `H * 64` input. The value term is
task-conditioned approximate time-reversal value consistency, not an exact
consequence of reversible dissipative contact dynamics. Coefficients retain the
existing warm-up/ramp schedules.

Four mechanisms remain distinct:

- Policy consistency is the actor forward-pass residual above.
- Value consistency is the task-conditioned critic residual above.
- Model-based sidecar augmentation is optional filtered reverse-action actor
  supervision from authentic transitions.
- PPO transition augmentation would duplicate ratios, returns, advantages, or
  targets; it remains disabled (`use_data_augmentation=false`).

## Model-based sidecar support

| History | Sidecar | Support |
| --- | --- | --- |
| Off | Off | Supported |
| Off | On | Supported, experimental |
| On | Off | Supported, primary V5 path |
| On | On | Rejected at startup |

The sidecar uses centralized 64-D slices when history is off. History plus the
sidecar is rejected because transforming only the current frame would leave an
inconsistent context, while fabricating a causal reversed history would require
unavailable future observations.

## TR-orbit competence curriculum

`command_curriculum_mode=tr_orbit_reward_threshold_v1` optionally maintains a
grid `(gait row, signed forward-speed bin)`. It is disabled by default and does
not replace the existing gait sampling profile or its iteration schedule.

The gait partner comes from the authoritative gait library; signed-speed bins
span the symmetric command range and the velocity partner contains the negated
bin. Weights, success EWMA, visit counts, unlock state, and sampling eligibility
are identical across every partner orbit. Reset samples a joint cell. A
velocity-only boundary samples a speed bin conditional on the retained gait; a
gait-only boundary samples a gait conditional on the retained speed bin. The two
timers remain independent, and deterministic evaluation overrides bypass the
curriculum.

Each segment accumulates mean body-frame XY velocity error, yaw-rate error,
completion/termination, and visits. Success uses the command generator's
absolute plus relative thresholds and requires nontermination. A successful
EWMA above the unlock threshold increases the current orbit and neighboring
speed-magnitude orbits, with a nonzero exploration floor and a configured
maximum. Checkpoint state includes weights, the active iteration-dependent gait
prior, EWMA, visits, unlocks, current per-environment cells, partial segment
accumulators, and the curriculum RNG. The task-local checkpoint hook also saves
and restores the active command, gait, timing, current phase, and segment runtime;
a full training resume refuses either direction of a curriculum-mode mismatch.
Actor-only play/evaluation loads synchronize the gait-schedule iteration but do
not restore per-environment curriculum runtime, so inference may use a different
environment count or keep the curriculum disabled.

## Checkpoints and launch controls

Train, play, record, and fixed-grid evaluation share `--history`, `--no-history`,
and `--history-length`. New runs default to `--history --history-length 30`.
Train and ablation additionally expose the optional command curriculum. Run
names include `h30` or `h0` and a compact curriculum label.

Each V5 run stores resolved environment/agent configuration, a
`policy_contract.json` manifest, and matching checkpoint metadata. Loading
checks the actor input width and contract before rollout and reports the expected
and received widths plus the required history flags. A 72-D V4 checkpoint is
incompatible: V5 never pads, truncates, projects, or silently migrates it.

Example Windows launches:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --history --history-length 30 --tr-policy-coef 0.1 --tr-value-coef 0.05
.\scripts\symm_locomotion\train.ps1 --robot x1 --no-history --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest --history --history-length 30
```

## Physical limitations

The time-reversal feature map is an inductive bias, not a claim that the robot
and contact process are reversible. Motor damping, friction, inelastic impact,
actuation lag, controller saturation, observation delay, contact discontinuity,
and termination all break physical reversibility. Framewise history does not
repair those effects. The value residual is approximate and task-conditioned,
and the experimental sidecar must still reject dynamically implausible reverse
candidates. Curriculum checkpointing exactly restores its sampler and task-local
command/gait bookkeeping, but ordinary RSL-RL checkpoints do not snapshot the
complete simulator physics state; resumed rollouts are therefore not claimed to
be bitwise continuations of contact dynamics.
