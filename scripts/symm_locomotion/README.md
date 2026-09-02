# Symmetric Locomotion Scripts

This folder is the shared launcher surface for the symmetric quadruped tasks.
Use it for both Unitree Go2 and Dobot X1, and add future robots here instead
of creating another robot-specific script tree.

Supported robot keys:

```text
go2    Unitree Go2
x1     Dobot X1
```

Aliases are also accepted: `unitree-go2`, `unitree_go2`, `dobot`,
`dobot-x1`, and `dobot_x1`.

Complete the repository setup in the root [README](../../README.md#requirements)
before using these launchers. In particular, rerun `isaaclab.bat -i core` or
`./isaaclab.sh -i core` after copying or moving the checkout so editable
package paths do not continue pointing at the old workspace.

The publication study's copy-ready schedule is in the
[seven-day command matrix](SEVEN_DAY_COMMANDS.md).

## Quick Commands

Windows PowerShell:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 20000 --num-envs 512 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 20000 --num-envs 512 --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
.\scripts\symm_locomotion\evaluation.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\symm_locomotion.ps1 compare --robots go2 x1
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1
.\scripts\symm_locomotion\scheduler.ps1 --help
```

Ubuntu/bash:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --iterations 20000 --num-envs 512 --no-trs
bash scripts/symm_locomotion/train.sh --robot x1 --iterations 20000 --num-envs 512 --no-trs
bash scripts/symm_locomotion/play.sh --robot go2 --checkpoint latest
bash scripts/symm_locomotion/play.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/record.sh --robot x1 --checkpoint latest --gif
bash scripts/symm_locomotion/evaluation.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/symm_locomotion.sh compare --robots go2 x1
bash scripts/symm_locomotion/tensorboard.sh --robots go2 x1
bash scripts/symm_locomotion/scheduler.sh --help
```

Direct Python style still works from an activated environment:

```bash
python scripts/symm_locomotion/train.py --robot go2 --iterations 20000 --no-trs
python scripts/symm_locomotion/play.py --robot x1 --checkpoint latest
python scripts/symm_locomotion/evaluation.py --robot go2 --checkpoint latest
python scripts/symm_locomotion/scheduler.py --help
```

The generic launcher accepts the command as its first argument:

```bash
bash scripts/symm_locomotion/symm_locomotion.sh train --robot go2 --smoke --dry-run
```

```powershell
.\scripts\symm_locomotion\symm_locomotion.ps1 train --robot x1 --smoke --dry-run
```

## Script lifecycle and inventory

The supported user surface and its implementation are deliberately separated.
Use the launcher families below for routine work; do not invoke internal
modules directly or extend historical study reproducers for new experiments.

| Lifecycle | Files | Purpose and recommendation |
|---|---|---|
| General launchers | `train.*`, `play.*`, `record.*`, `evaluation.*`, `comparison.*`, `tensorboard.*`, `ablation.*`, `symm_locomotion.{sh,ps1}`, `symm_cli.py` | Supported interfaces for routine training, playback, recording, evaluation, comparison, TensorBoard, and ablation workflows. Prefer the platform wrapper or the matching Python entry point. |
| General study utilities | `launch_study.py`, `scheduler.*` | Supported advanced tools for manifest-defined studies and delayed sequential shell jobs. |
| Internal implementation | `_run.{sh,ps1}`, `_tensorboard_scalars.py`, `_leg_usage_metrics.py`, `training_provenance.py`, `study_registry.py`, `_mp4_to_gif.py` | Required implementation and validation modules. They are not additional launcher families; keep them beside the public entry points. |
| Archival study reproduction | `analyze_matched_trs_study.py`, `analyze_trs_grid.py`, `plot_good_runs_tensorboard.py`, `plot_trs_tensorboard.py`, `update_gait_family_v3_analysis.py`, `paired_tr_consistency.py` | Retained so existing studies remain reproducible. Use `comparison.*` and `evaluation.*` for new work. |
| Compatibility modules | `leg_usage_metrics.py`, `mp4_to_gif.py` | Stable historical import and script paths; current launchers use their underscored implementation modules. |
| Deprecated compatibility | `compare.*`, `analyze_leg_usage.*` | Temporary forwarding interfaces. Migrate to `symm_locomotion.* compare` and `evaluation.*`; these files remain for the required deprecation window. |

The former unreleased `compare_gait_closure_runs.py` alias was removed; use
`comparison.py`, `comparison.sh`, or `comparison.ps1`. Tests under `test/` and
the JSON facts under `study_registry/` support the files above and are not user
launchers.

## Delayed Command Scheduler

`scheduler.py` is the platform-neutral scheduler implementation;
`scheduler.ps1` and `scheduler.sh` select PowerShell and bash, respectively.
All three run any number of shell commands sequentially. Repeat a
`--delay`/`--command` pair for every step. The first delay begins immediately,
and each later delay begins only after the preceding command has finished, so
the commands do not overlap.

For example, this waits five hours before training, ten minutes after training
finishes before recording, and another ten minutes after recording finishes
before evaluation:

```powershell
.\scripts\symm_locomotion\scheduler.ps1 `
  --update_interval 5m `
  --delay 5h --command {
    .\scripts\symm_locomotion\train.ps1 --robot go2 --run-name delayed_go2
  } `
  --delay 10m --command {
    .\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
  } `
  --delay 10m --command {
    .\scripts\symm_locomotion\evaluation.ps1 --robot go2 --checkpoint latest --protocol light
  }
```

The equivalent bash form uses quoted bash command strings:

```bash
bash scripts/symm_locomotion/scheduler.sh \
  --update_interval 5m \
  --delay 5h --command 'bash scripts/symm_locomotion/train.sh --robot go2 --run-name delayed_go2' \
  --delay 10m --command 'bash scripts/symm_locomotion/record.sh --robot go2 --checkpoint latest --gif' \
  --delay 10m --command 'bash scripts/symm_locomotion/evaluation.sh --robot go2 --checkpoint latest --protocol light'
```

Durations accept short or long unit suffixes such as `500ms`, `10min`, `5h`,
and `2days`, or a clock duration such as `01:30:00`.
`--update_interval` controls countdown output and defaults to `5m`. Use
`--dry_run` to validate and print a schedule without waiting or executing it.
A failed command stops later steps by default; add `--continue_on_error` to run
them anyway. The PowerShell wrapper accepts either script blocks (`{ ... }`) or
quoted command strings. Every step runs in an isolated child shell, so a
command containing `exit` reports that child's exit code and cannot terminate
the scheduler before remaining `--continue_on_error` steps run. Direct
`scheduler.py` use selects PowerShell on Windows and bash elsewhere; use
`--shell powershell|bash|sh` or `--shell_executable PATH` to override it.

## Common Options

Every command accepts:

```text
--robot go2|x1
--conda-env symm_rl_isaaclab
--use-conda-run
--no-conda-run
--expected-branch BRANCH
--dry-run
```

Training options:

```text
--iterations / --max-iterations
--num-envs
--run-name
--seed
--mirror
--tr-value-coef
--tr-warmup-iterations
--tr-rampup-iterations
--tr-ramp-shape linear|half_cosine
--tr-min-abs-cmd-vel
--foot_phase_weight
--foot_phase_reduction sum|mean
--joint_target_limit_mode legacy_clamped|requested_overflow
--joint_target_limit_weight
--actor_mean_bound_mode legacy_global|per_joint_feasible
--tr_policy_output_space raw_action_mean|normalized_requested_joint_target
--gait_sampling_profile trclosed_v2_equal_family|trclosed_v2_v1_equivalent|trclosed_v2_halfbound_anneal
--gait_curriculum_iterations
--no-trs
--smoke
```

New launcher options use the snake_case spellings shown above. Their
hyphenated forms remain accepted as compatibility aliases.

Training and ablation runs default to 20,000 iterations across 512 environments.

`--tr-warmup-iterations` sets the fully unregularized updates before TRS starts.
`--tr-rampup-iterations 0` preserves the hard switch; positive values ramp both
TRS coefficients with the selected linear or half-cosine shape.

Use `--expected-branch BRANCH` to abort before launch when the checkout is on a
different branch or detached HEAD.

`--tr-min-abs-cmd-vel` defaults to `0.0`, so TRS losses also train on
zero-velocity commands used for in-place behavior.

`--no-trs` disables symmetry data augmentation, mirror loss, and TRS value
loss by forwarding these Hydra overrides:

```text
agent.algorithm.symmetry_cfg.use_data_augmentation=False
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
agent.algorithm.symmetry_cfg.use_mirror_loss=False
agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0
agent.algorithm.symmetry_cfg.value_loss_coeff=0.0
```

### Publication time-reversal controls

`TimeReversalPPO` keeps the ordinary on-policy PPO surrogate, value loss,
entropy loss, GAE, returns, adaptive KL schedule, and rollout storage intact.
Let `N` be a PPO minibatch, `D_a=12` the action dimension, `m_n` the historical
validity mask, and `stopgrad` a detached target. The implemented auxiliary
objectives are exactly:

```text
r^pi_nd = mu_d(T_O(o_n)) - T_A(stopgrad(mu(o_n)))_d
r^V_n   = V(T_O(o_n)) - stopgrad(V(o_n))

L_TR_policy = sum_n sum_d m_n (r^pi_nd)^2 / max(1, D_a sum_n m_n)
L_TR_value  = sum_n       m_n (r^V_n)^2  / max(1,     sum_n m_n)

L_TR_DA = -sum_j v_j w_j log pi(a_tilde_j | o_tilde_j)
           / (sum_j v_j w_j + 1e-8)
```

Here `v_j` selects an accepted reverse candidate and `w_j` is either one or
its detached confidence. Policy MSE is therefore averaged over selected
sample-action elements, not over per-sample squared vector norms. An empty
policy/value mask returns exactly zero. Reverse-action NLL also returns exactly
zero when there is no positive selected weight.

The four mechanisms that can otherwise be confused are:

| Mechanism | Data used | Implemented effect in this path |
|---|---|---|
| Inherited instantaneous symmetry augmentation | A transformed copy of the current PPO minibatch and its PPO quantities | Deprecated compatibility path retains the online minibatch duplication behavior after its warmup |
| Policy consistency | Historical `o_n` and `T_O(o_n)` | Auxiliary actor-mean MSE only; no transition is inserted |
| Value consistency | Historical `o_n` and `T_O(o_n)` | Auxiliary critic MSE only; no return or value target is inserted |
| Trajectory reverse-action augmentation | Complete authentic state-action-successor segments | Learned-filtered actor NLL only; no PPO ratio, GAE, return, critic, KL, or entropy term |

The existing same-phase leg synchronization reward is a phase-gated task
reward; it is not a complete morphological-equivariance constraint.

The canonical policy and value switches are
`use_tr_policy_consistency` and `use_tr_value_consistency`. The legacy
`use_mirror_loss`, `mirror_loss_coeff`, `value_loss_coeff`, shared
`warmup_iterations`, shared `rampup_iterations`, and shared `ramp_shape`
fields remain accepted. A canonical switch takes precedence when it is set.
`None` resolves through the legacy master `use_time_reversal_regularization`
and that term's legacy switch/nonzero coefficient; `schedule.enabled` can only
disable the resolved term. A schedule `target_coeff` overrides the legacy
coefficient. Each canonical term has its own
`tr_policy_schedule` or `tr_value_schedule`, with `enabled`, `target_coeff`,
`warmup_iterations`, `rampup_iterations`, `hold_iterations`,
`decay_iterations`, `final_scale`, and `ramp_shape`. Policy, value, and
augmentation use independent absolute PPO-update schedules. With
`W=warmup_iterations`, `R=rampup_iterations`, `H=hold_iterations`,
`D=decay_iterations`, `F=final_scale`, `b=W+R`, `h=b+H`, and integer update
`k>=0`, define:

Durations must be nonnegative integers, `target_coeff` finite and nonnegative,
`F` in `[0,1]`, and the shape either `linear` or `half_cosine`.

```text
f_linear(p)     = p
f_half_cosine(p)= 0.5 * (1 - cos(pi*p))

q(k) = 0                                      , k < W
       f((k-W)/R)                             , R > 0 and W <= k <= b
       1                                      , b <= k <= h
       1 + (F-1) f((k-h)/D)                   , D > 0 and h < k <= h+D
       F                                      , k > h+D

lambda_eff(k) = target_coeff * q(k) when enabled, otherwise 0
```

The overlapping ramp/hold endpoint is one in both branches. `R=0` is a hard
switch with `q(W)=1`; `D=0` switches to `F` on the first update after the hold.
A positive decay reaches `F` at its last decay update and remains there.

### Reward and feasible-action semantics

For nonnegative per-foot phase violations `p_i`, the direct launcher selects
either `P_sum=sum_i p_i` or `P_mean=(sum_i p_i)/4`; the reward contribution is
`R_phase=-w P`. Consequently `--foot_phase_weight 0.4
--foot_phase_reduction mean` is exactly the same tensor calculation as weight
`0.1` with `sum`. Existing saved configurations remain `0.3 * sum` unless
overridden. Training logs the configured reduction, the sum-equivalent and
per-foot effective weights, raw sum/mean, and weighted contribution.

`legacy_clamped` retains the historical joint-target-limit reward exactly. In
`requested_overflow`, with requested target `q_req=q0+s*a`, soft interval
`[l,u]`, range `r=u-l`, and margin `eta*r`, the per-environment penalty is:

```text
d_minus = relu(l + eta*r - q_req)
d_plus  = relu(q_req - (u - eta*r))
p_limit = mean_i smoothL1((d_minus_i + d_plus_i) / (r_i + epsilon))
```

It is evaluated before the execution safety clamp and is not capped. Requested
overflow and actual execution clipping are recorded separately, including
hip/thigh/calf clipping fractions.

`per_joint_feasible` derives raw-action bounds in action-manager joint order:

```text
a_min_i = min((l_i-q0_i)/s_i, (u_i-q0_i)/s_i)
a_max_i = max((l_i-q0_i)/s_i, (u_i-q0_i)/s_i)

p_mu = mean_i [smoothL1(relu(a_min_i+m_i-mu_i)/(a_max_i-a_min_i))
             + smoothL1(relu(mu_i-a_max_i+m_i)/(a_max_i-a_min_i))]
```

This is a soft learner penalty; stochastic policy samples are never replaced
by hard-clipped means. `legacy_global` preserves the historical squared excess
beyond absolute raw action 10.

For `normalized_requested_joint_target`, policy TR consistency uses the
unclipped target coordinate
`z_i=(2*(q0_i+s_i*mu_i)-(u_i+l_i))/(u_i-l_i+epsilon)`. The current temporal
action operator is the identity. Raw-action and normalized-target consistency
errors are both logged; only the selected output space is optimized.

The equal-family gait profile has weights `(4,4,1,1,1,1,1,1,1,1)`. The
V1-equivalent profile has `(4,4,2,2,0,0,1,1,1,1)`. The annealed profile moves
continuously from the latter to the former over the requested absolute PPO
update count. Every temporal partner pair has equal weight at every point;
resume restores the absolute update before the next rollout. Full evaluation
continues to cover all ten rows regardless of training probability.

Every direct training run writes immutable
`provenance/resolved_command.json` and `provenance/cohort_metadata.json` files
alongside `provenance/initialization.json`. The initial cohort status is
`incomplete`; later registry classification uses protocol facts and never the
observed reward quality.

The resolved command records the active Python entry point using a
repository-relative, forward-slash path (for example
`scripts/symm_locomotion/train.py`) plus its exact launcher arguments. It does
not claim that Linux, direct-Python, or generic CLI launches used the Windows
PowerShell wrapper.

Do not rewrite that initial `incomplete` record after training. Terminal
checkpoint and evaluation facts belong in a registry manifest. The repaired
seed-42 V4 main sweeps are recorded in
`study_registry/go2_v4_main_development.json` and
`study_registry/x1_v4_main_development.json`; each references all five main
treatments, their final checkpoints, and their full-V3 analysis provenance.
They remain development evidence and are not members of either prospective
confirmatory cohort.

New V5 checkpoints store schema-3 `time_reversal_state` with the exact policy
observation contract plus
`last_completed_update`, the consecutive `next_absolute_update`, and the exact
resolved policy, value, and augmentation schedules. The runner `iter` must be
either that last or next update; loading advances execution to the saved next
absolute update. A pre-schema checkpoint falls back to `iter+1`. Enabled
trajectory augmentation additionally requires matching side-model state,
semantic configuration, RNG state, and schedule iteration; it refuses a
silent random restart or a checkpoint whose schedule semantics differ.
An enabled command curriculum likewise stores its grid, RNG, active cells, and
task-local command/gait runtime, and full resume rejects a curriculum-mode
mismatch in either direction.

The TR validity mask is computed from the latest 64D frame in the saved policy
minibatch observation, whether the policy input is instantaneous or a native
term-major flattened history. Available modes are
`command`, `command_tracking`, `command_upright_phase`, and
`command_tracking_upright_phase`. Their components are:

```text
abs(vx_cmd) >= v_min
sqrt(projected_gravity_x^2 + projected_gravity_y^2) <= g_tolerance
d_S1(phase_i, 0) >= epsilon_phase and
d_S1(phase_i, swing_ratio) >= epsilon_phase for every foot
```

Measured base velocity is intentionally absent from the hardware policy
contract, so the legacy `command_tracking` variants reduce to the command gate;
reward-side tracking diagnostics remain simulator-only.

These are heuristic stable-phase validity gates, not proof that a physical
transition is reversible. Low-frequency gradient diagnostics use
`torch.autograd.grad` on only the first selected minibatch and report norms,
weighted norm ratios, cosine similarities, parameter coverage, and
non-finite status without changing `.grad`, optimizer state, or the training
loss. When evaluated, diagnostics expose an empty mask, which contributes an
exact zero auxiliary loss.

Trajectory augmentation is disabled by default. When explicitly enabled,
`tr_augmentation.mode` must be
`dynamics_filtered_reverse_action_supervision` and its dynamics filter must
remain enabled. A project-local sidecar records authentic `(x_t, a_t,
x_{t+1})` transitions, rejects auto-reset successors, splits sequences at
episode and task discontinuities, reverses complete constant-task segments,
and rebuilds the previous-action observation channel from reversed action
history. A forward-model EMA admits candidates only after held-out validation;
the reverse action is either the analytic identity transform for the current
position-target action or a separately trained inverse-dynamics prediction.

The dynamics state makes temporal parity explicit. Root position relative to
the environment origin, root 6D rotation, joint position relative to the
default pose, actuator target, previous action, swing/stance ratios, gait
period, body masses, material properties, and the currently empty additional
actuator state are even. Body-frame root linear/angular velocity, joint
velocity, and the three-component velocity command are odd. Time reversal also
applies:

```text
common_phase' = (swing_ratio - common_phase) mod 1
foot_phase_offsets' = -foot_phase_offsets
gait_row' = declared_time_reversal_partner[gait_row]
T_A(a) = a
```

The current declared partner map is involutive. The identity action map is
runtime-asserted involutive and volume preserving. In a constructed reversed
segment, actuator targets and previous actions are replaced with adjacent
reversed values, not copied from chronological history. The first reverse
transition is dropped because its preceding reverse action cannot be
reconstructed.

For `B` runtime rigid bodies, `S` runtime rigid shapes, and `Z` hidden actuator
state values, the continuous dynamics feature width is
`D_x=89+B+3S+Z`; the gait row contributes a ten-way one-hot vector and the
action width is `D_a=12`. Euclidean features are normalized as
`z=(x-mean)/sqrt(variance+1e-6)`. Let `E` be the normalized feature indices
after excluding the six orientation coordinates, the common-phase sine/cosine
pair, and four foot-phase sine/cosine pairs. For physical rotation matrices
`R_hat,R`, and any predicted/target phase pairs `p=(p_s,p_c)` and
`t=(t_s,t_c)`, the component-aware forward residual is:

```text
alpha = 2 asin(clamp(||R_hat-R||_F / (2 sqrt(2)), 0, 1))
delta(p,t) = atan2(p_s t_c - p_c t_s, p_c t_c + p_s t_s) / (2 pi)
C(p,t) = delta(p,t)^2 + (||p||_2 - ||t||_2)^2

e = [sum_{d in E}(z_hat_d-z_d)^2
     + (alpha/pi)^2 + C(common_hat,common)
     + sum_{i=1}^4 C(foot_hat_i,foot_i)] / (|E| + 1 + 1 + 4)
```

Thus `|E|=D_x-16` and the residual denominator is `D_x-10`. The forward
model minimizes `mean(e)`. The inverse model minimizes elementwise normalized
action MSE with a separate Adam optimizer. To prevent direct target leakage,
both actuator-target and previous-action slices are removed from both inverse
model endpoints, so its input width is `2(D_x-24)` and output width is 12.
Learned actions are denormalized and then used to rebuild every retained
previous-action and processed-target channel. Live forward/inverse models have
separate EMA targets.

Validation is a deterministic hash split:

```text
h = (73,856,093*environment_id + 19,349,663*rollout_step + rng_seed) mod 10,000
validation = h < floor(10,000*validation_fraction)
```

State-normalizer moments use the complete recent real-transition batch,
whereas forward/inverse parameter updates use only its training split.
For held-out real forward residuals, the acceptance threshold is:

```text
beta = threshold_multiplier * quantile_validation_quantile(e_real)
model_ready = (heldout_count >= minimum_validation_samples)
              and isfinite(beta)
              and mean(e_real) <= maximum_validation_loss
```

A candidate is accepted only when the observation, action, and residual are
finite; `max(abs(action))` is within the configured limit; every foot at both
the current and successor endpoint is at least the phase margin from liftoff
and touchdown; no touchdown or liftoff is crossed over the phase interval; the
contact-impulse estimate is within its limit; `e<=beta`; and `model_ready` is
true. The contact estimate takes the contact-force vector norm, averages each
sensor history over the control interval, multiplies by the control step
(`[N s]`), and then takes the maximum over bodies and contact sensors. With
confidence weighting, `w=exp(-e/(beta+1e-8))`. At most
`floor(max_augmented_to_original_ratio * original_minibatch_size)` candidates
are sampled with a private checkpointed RNG. No accepted or sampled candidate
means exact zero augmentation loss.

A reverse transition starts at `T_X(x_{t+1})`, not `T_X(x_t)`, and its action
was not sampled by the rollout behavior policy at that reversed state. The
original old log probability, advantage, return, and value are therefore not
valid for it. Reverse trajectory candidates never enter a PPO likelihood
ratio, clipping, GAE, returns, clipped value loss, adaptive KL, entropy
averaging, or critic regression. The deprecated
`use_data_augmentation=True` flag is a separate compatibility path: at its
historical warmup boundary it duplicates the current PPO minibatch and repeats
the stored PPO quantities exactly as the online implementation did.
`TimeReversalPPO` warns when that path is enabled; new studies should keep it
false and use the independently filtered trajectory augmentation objective. A
future replay-based objective would need a separate off-policy derivation.

Maintained assumptions and limitations are intentionally strict:

- Publication use is flat terrain with exactly one 12D affine joint-position
  action term and wrapper `clip_actions=None`. Analytic `T_A(a)=a` is specific
  to these even position-target offsets; another controller needs a new action
  transform and target reconstruction.
- Recurrent actor/critics are rejected because reversed hidden-state semantics
  are undefined. Distributed/multi-GPU augmentation is rejected because side
  models are not synchronized.
- Isaac Lab exposes returned observations after auto-reset but no authentic
  terminal dynamics state here. Every done or timeout transition is therefore
  rejected; no post-reset observation is paired with a terminal state.
- Body masses and rigid-shape material properties are required and fail closed
  when unavailable. The current actuator hidden state has zero width; a new
  stateful actuator needs an explicit parity transform.
- The control-step contact impulse is a coarse sensor-history estimate, not an
  exact collision impulse. Phase, impulse, and learned-residual gates are
  heuristics and do not prove physical reversibility.
- Held-out samples come from the same recent on-policy rollout, and normalizer
  moments include that rollout. Validation is not an out-of-distribution or
  simulator-replay guarantee. Analytic and learned-inverse action sources must
  therefore be reported separately.

The following exact Hydra override profiles can be appended after the
launcher's `--` delimiter. The common legacy field is kept false in every
profile so it cannot invoke instantaneous minibatch copying:

```text
# No TRS
agent.algorithm.symmetry_cfg.use_data_augmentation=false
agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false
agent.algorithm.symmetry_cfg.use_tr_value_consistency=false
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false

# Actor only (append to the common use_data_augmentation=false override)
agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true
agent.algorithm.symmetry_cfg.use_tr_value_consistency=false
agent.algorithm.symmetry_cfg.tr_policy_schedule.enabled=true
agent.algorithm.symmetry_cfg.tr_policy_schedule.target_coeff=0.1
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false

# Value only
agent.algorithm.symmetry_cfg.use_tr_policy_consistency=false
agent.algorithm.symmetry_cfg.use_tr_value_consistency=true
agent.algorithm.symmetry_cfg.tr_value_schedule.enabled=true
agent.algorithm.symmetry_cfg.tr_value_schedule.target_coeff=0.05
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false

# Actor + value
agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true
agent.algorithm.symmetry_cfg.use_tr_value_consistency=true
agent.algorithm.symmetry_cfg.tr_policy_schedule.enabled=true
agent.algorithm.symmetry_cfg.tr_policy_schedule.target_coeff=0.1
agent.algorithm.symmetry_cfg.tr_value_schedule.enabled=true
agent.algorithm.symmetry_cfg.tr_value_schedule.target_coeff=0.05
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false

# Actor + value + optional analytic reverse-action supervision
agent.algorithm.symmetry_cfg.use_tr_policy_consistency=true
agent.algorithm.symmetry_cfg.use_tr_value_consistency=true
agent.algorithm.symmetry_cfg.tr_augmentation.enabled=true
agent.algorithm.symmetry_cfg.tr_augmentation.mode=dynamics_filtered_reverse_action_supervision
agent.algorithm.symmetry_cfg.tr_augmentation.action_source=analytic
agent.algorithm.symmetry_cfg.tr_augmentation.filter_enabled=true
agent.algorithm.symmetry_cfg.tr_augmentation.coefficient=0.02
agent.algorithm.symmetry_cfg.tr_augmentation.schedule.enabled=true
agent.algorithm.symmetry_cfg.tr_augmentation.schedule.target_coeff=0.02
```

For a controlled cohort, prefer the immutable launcher described in
[STUDY_MANIFEST.md](STUDY_MANIFEST.md). It captures the exact expanded command,
treatment variables, initialization hashes, and requested evaluation protocol.
Each initialization record is bound back to its study, branch, run, condition
digest, command, and training/environment seed. Cohort validation compares the
full resolved environment and agent configurations after masking only the
documented run-output paths and exact configuration paths derived from declared
treatments. Unmapped simulator, environment, PPO, or task changes fail the
match. A reward-profile name is provenance only—its resolved Hydra overrides
define behavior—and an unresolved external robot asset is an explicit
publication-validation failure until an authoritative content hash is supplied.

Policy consistency adds one transformed actor forward for each minibatch where
its raw loss is requested; value consistency analogously adds one transformed
critic forward. Diagnostics add two `autograd.grad` evaluations per enabled
term on only the first minibatch at their configured cadence. Disabled raw
logging and diagnostics add no such work when an effective coefficient is zero.

For the default two-layer `256x256` side models and
`D_x=89+B+3S+Z`, exact parameter counts are:

```text
P_forward = 513 D_x + 69,120
P_inverse = 512 D_x + 56,844    # input is 2(D_x-24), output is 12
```

Live weights, EMA weights, and Adam's two moment tensors dominate the
persistent/checkpoint payload at approximately
`4(P_forward+P_inverse)` FP32 values, or `16(P_forward+P_inverse)` bytes,
excluding gradients, activations, optimizer step scalars, normalizers, and
metadata. The logical sidecar payload per environment-control-step is:

```text
state raw floats = 74 + B + 3S + Z, plus one int64 gait row
transition       = 348 + 2B + 6S + 2Z float32 values
                   + 7 int64 values + 5 bool values
rollout bytes    ~= N_env H [4(348+2B+6S+2Z) + 7*8 + 5]
```

This excludes tensor/Python allocator overhead, the pending pre-step record,
the accepted pool, gradients, and activations. Enabled augmentation also runs
forward/inverse model updates, EMA inference, and filter calibration once per
rollout. When it is disabled, side models, optimizers, EMA copies, sidecar, and
pool are not allocated.

At the Go2/X1 control step of `0.02 s`, the nominal 15-second cell contains 750
samples. Light therefore records 6,000 samples across 8 cells (120 simulated
seconds), while full records 45,000 across 60 cells (900 simulated seconds), a
7.5x simulation-step ratio. Wall time is hardware- and rendering-dependent.
`sim_data.npz` uses compression, so archive bytes are data-dependent; the
uncompressed payload is exactly the sum of `array.nbytes` for its stored
arrays, plus container metadata, rather than a fixed advertised size.

`--smoke` uses one environment and one training iteration.

Play and record resolve `--checkpoint latest` from the newest run under the
selected robot experiment directory. You can also use:

```text
--run RUN_FOLDER_OR_PATH
--model 9999
--checkpoint PATH_TO_MODEL_PT
```

Play and record default to a deterministic six-gait sequence with five seconds
per gait: trot, bound, front-spread half-bound, hind-spread half-bound, rotary
gallop, then transverse gallop. Interactive play repeats the sequence; the
default 30-second recording captures one complete cycle. Use
`--gait-sequence-duration SECONDS` to change both the dwell time and the
default one-cycle recording length, or
`--no-gait-sequence` to restore random gait sampling and the legacy 30-second
recording length. Playback disables phase-offset noise, retains period noise,
and resamples the velocity command once at 10 seconds. Velocity resampling and
sequence-row assignment both refresh period and duty factor from the current x
command. When they coincide, as they do at 10 seconds, they share one timing
refresh using the new velocity. The clock is piecewise-integrated and
re-anchored before every timing change, so common phase is continuous and then
advances using the new period. The completed transition is recorded under its
old desired signal; the next policy observation receives the new velocity,
gait offsets, period, duty factor, and continuous phase together. The playback
episode time limit is one control step longer than the full gait cycle, which
keeps its reset pose out of the recording. Physical fall and safety terminations
remain active; the global row-sequence clock continues across resets while the
per-environment oscillator phase restarts at zero.

This continuous-clock behavior changes phase observations relative to archived
v2/v3 checkpoints. Retrain those policies for correctness-aligned deployment,
or use their archived environment implementation for exact legacy playback.

Training uses a separate ten-row, fore/hind-balanced, time-reversal-closed gait
library. Its row weights `(4, 4, 1, 1, 1, 1, 1, 1, 1, 1)` assign equal
probability to the trot, bound, half-bound, and gallop families while splitting
the half-bound family evenly between two front-spread and two hind-spread rows.
The closure-only partners do not alter the six-gait playback sequence. The
shared training configuration sums foot-phase violations over feet and averages
leg-permutation errors over active synchronized pairs. It weights direct
foot-phase tracking at `0.30`, leg-permutation symmetry at `0.20`, and the
hip-action penalty at `0.10`. During each 30-second training episode, velocity
is sampled at reset and resampled once at 10 seconds, an XY velocity disturbance
is applied at 15 seconds, and the gait row is resampled once at 20 seconds.
The velocity resample updates period and duty factor immediately. Playback uses
the same one-shot velocity schedule without the disturbance while its gait rows
change every five seconds.

Recordings are 30 seconds by default (1,500 environment steps at 50 Hz).
Pass `--video-length` to override the length in environment steps.

Play and record save the following rollout diagnostics under the selected
checkpoint run's `plots/play/` directory by default:

```text
sim_data.npz
figure1_linear_velocities_and_position.png
figure2_E_C_frc_and_contact_forces.png
figure3_E_C_spd_and_foot_velocities.png
figure4_agg_E_C_frc_vs_contact.png
figure5_policy_actions_and_joint_limits.png
figure6_straight_line_reward_diagnostics.png
figure7_foot_clearance.png
figure8_leg_motor_torques.png
figure9_leg_motor_powers.png
figure10_leg_ground_reaction_forces.png
```

These reproduce the IsaacGym rollout plots for measured versus desired base
velocity/position, `E_C_frc` versus foot contact force, and `E_C_spd` versus
foot speed, with additional policy-action, joint-limit, and straight-line reward
diagnostics. The leg-usage figures are ordered front-left, front-right,
rear-left, rear-right. They plot the absolute value of each motor torque, each
motor's absolute mechanical power (`abs(torque * joint velocity)`), and each
absolute world-frame ground-reaction-force component. The black aggregate trace
is the L1 sum for that leg: `sum(abs(component))`, not `abs(sum(component))` or
the Euclidean force norm. Standard Go2/X1 playback filters contact to the flat
ground and adds the tangential friction force to the ground-normal force.

Each reported magnitude has a thin raw curve and a thicker dashed centered
1-second moving arithmetic mean. At the standard 50 Hz control rate this uses
51 samples spanning `t - 0.5 s` through `t + 0.5 s`. Plot edges use the available
partial window with the correct sample count, and smoothing never crosses an
episode reset. This centered, edge-corrected mean avoids the phase delay and
zero-padding bias of a causal or convolution-padded moving average.

`sim_data.npz` retains the signed source arrays for regeneration or signed power
analysis, as well as the derived absolute and smoothed arrays, actions, targets,
positions, soft limits, limit utilization, and measured/target swing-foot
heights. Use `--no-plots` to disable plots, `--plots_dir PATH` to override the
output directory, or `--plot_env_index INDEX` to select another environment.
Plot collection is limited to 30 seconds by default; use
`--plot_duration SECONDS` to change the window.

The main leg-usage arrays in `sim_data.npz` are:

- `joint_torques`, `joint_velocities`, `joint_powers`: `(T, 12)` in the saved `joint_names` order.
- `leg_joint_torques`, `leg_joint_powers`: `(T, 4, 3)` in FL, FR, RL, RR order.
- `leg_joint_torque_magnitudes`, `leg_joint_power_magnitudes`: absolute per-motor values;
  `leg_torque_magnitude_sums`, `leg_power_magnitude_sums`: their per-leg L1 sums.
- `leg_torque_sums`, `leg_power_sums`: signed sums retained for compatibility and analysis.
- `foot_ground_reaction_forces_w`: `(T, 4, 3)` world-frame force vectors; the
  boolean `ground_reaction_force_includes_friction` records whether each sample contains friction.
- `foot_ground_reaction_force_abs_components`, `foot_ground_reaction_force_abs_sums`:
  absolute force components and their L1 sums.
- Every plotted magnitude key also has a `_centered_moving_mean` array. The scalar
  `usage_plot_smoothing_window_s` and `usage_plot_smoothing_window_samples` fields
  record the configured duration and actual odd sample count; `episode_done`
  records the boundaries applied during smoothing.

Relative `--run` values are resolved under the selected robot's routine log
directory, such as `logs/rsl_rl/unitree_go2_symm_flat/`. For curated
`logs/rsl_rl/good_runs/` checkpoints, pass the checkpoint path directly with
`--checkpoint`.

## Policy evaluation

`evaluation.py`, `evaluation.ps1`, and `evaluation.sh` evaluate a checkpoint at
fixed commands. They report velocity and heading tracking, gait/contact
fidelity, transient response and progress, actuator/foot loading, leg-use
balance, and combined success. The default protocol is
the Cartesian grid of all ten time-reversal-closed v2 training gait rows and
`vx = -1.5, -1.0, -0.5, +0.5, +1.0, +1.5 m/s`. Every cell runs in its own
episode with zero lateral/yaw command and the nominal evaluation profile. Its
first 5 seconds are nominal settling time and the following 10 seconds are the
nominal measurement window. Analysis trims the post-settle slice to complete
common gait cycles when at least one full cycle is available, so its actual
measurement start, stop, duration, and sample count can be shorter and are
recorded per cell.

The publication methods are immutable and distinct: `full` uses
`leg_usage_grid_full_v3`, while `light` uses `leg_usage_grid_light_v2`. The
eight-cell light screen resolves rows through stable gait names and declared
time-reversal partners: trot at `+/-1 m/s`, bound at `+/-1 m/s`,
`half_bound_front_a` at `+1 m/s` with its partner at `-1 m/s`, and `gallop_a`
at `+1 m/s` with its partner at `-1 m/s`. It does not select rows by copied
phase literals. The two protocols use separate output roots and cannot share a
study identity.

New full-v3 artifacts use `evaluations/leg_usage_grid_full_v3/`. Historical
and newly requested legacy-v1 custom grids remain under
`evaluations/leg_usage_grid/`, so either can be analyzed or resumed without
deleting the other. Legacy resume verifies the existing v1 manifest and every
requested grid/runtime control, then uses that manifest as the plan of record;
this avoids rewriting its historical source identity.
Full-v3 manifests created before the directory split are detected by their
stored method and remain available to `--resume` and `--analyze_only` in the
old root; all newly initialized full-v3 studies use the new root.

For each foot, measured contact is a hysteretic state driven only by the
nonnegative world-vertical force filtered to the literal collision path
`/World/ground/terrain/mesh`. Contact state starts false at the beginning of
the analyzed, cycle-trimmed slice. It enters at `Fz >= F_on`, exits at
`Fz <= F_off`, and confirms a change only after:

```text
n_dwell = max(1, ceil(minimum_dwell_s / step_dt - 1e-12))
```

Once confirmed, the new state is backfilled to the first threshold-crossing
sample. Thresholds satisfy `F_on > F_off >= 0`; body-weight mode uses
`g=9.80665 m/s^2` and resolves:

```text
F_on  = max(minimum_on_n,  alpha_on  * mass * g / 4)
F_off = max(minimum_off_n, alpha_off * mass * g / 4)
```

The resolved force thresholds, effective sample dwell, robot mass, and ground
filter paths are recorded and checked against each cell's recording manifest.
`contact.ground_filtered_required` defaults to `true`; setting it to `false`
allows contact analysis from a declared but non-ground-filtered force trace.
With the archived commanded duty trace `beta_k`, swing ratio `s_k=1-beta_k`,
and wrapped foot phase `psi_ki`, desired stance is
`c_ki*=1{psi_ki >= s_k}`. Boundary-excluded scores retain a sample only when
its circular distances from liftoff (`psi=0`) and touchdown (`psi=s_k`) are
both strictly greater than `boundary_exclusion_cycles`; equality is excluded.
Contact metrics are:

```text
A_contact       = 1 - mean(|c_i - c_i*|)
false swing     = sum(1{c_i=1 and c_i*=0}) / sum(1{c_i*=0})
missed stance   = sum(1{c_i=0 and c_i*=1}) / sum(1{c_i*=1})
precision       = TP / (TP + FP)
recall          = TP / (TP + FN)
F1              = 2 * precision * recall / (precision + recall)
beta_i_measured = mean_k(c_ki)
beta_commanded  = mean_k(beta_k)
duty error_i    = beta_i_measured - beta_commanded
abs duty error_i= abs(duty error_i)
```

The separately reported sampled desired-stance fraction is `mean_k(c_ki*)`;
it is not substituted for the archived commanded `beta_commanded`. A ratio
whose denominator is nonpositive is `N/A` (`None`). Thus an empty scored mask
makes all six classification metrics `N/A`; precision or recall with no
positive denominator is `N/A`; and F1 is `N/A` if either input is unavailable
or their sum is zero.

Agreement is reported both with and without event-boundary samples. Event
errors match measured touchdown/liftoff to commanded events in the same
complete cycle and use
`d_S1(a,b)=abs(remainder(a-b+0.5,1)-0.5)`. Mean, median, p95, matched count,
and per-foot event coverage remain explicit. For `M` matches, `U_e` missing
expected events, and `U_a` extra actual events, the combined unmatched fraction
is `(U_e+U_a)/(M+U_e+U_a)`; expected-only and actual-only fractions use
`U_e/(M+U_e)` and `U_a/(M+U_a)`. A zero denominator is `N/A`; with no match,
event error summaries are `N/A` and matched count is zero.

Same-phase pairs are configured when their declared circular offset difference
is at most `phase_sync_tolerance_cycles` (default 0.02, distinct from the 0.04
simultaneous-order tolerance). A pair-cycle is matched only when each foot has
exactly one event in that complete cycle. Aggregate coverage is
`matched_pair_cycles/(configured_pairs*complete_cycles)` and is `N/A` when the
denominator is zero. Touchdown and liftoff disagreement are reported in seconds
and, when period is available, cycles.

Cyclic order uses the first touchdown of each foot in each cycle and scores
only cycles containing all four feet. Events within the simultaneous tolerance
form an equivalence class, including a merge across the circular cycle
boundary. Modal ties are resolved lexicographically. For classification, each
foot's touchdown samples first form a circular mean `phi_bar_i`. Only the six
pairwise phase differences enter a canonical-row score:

```text
score_r = (1/6) sum_{i<j} d_S1(phi_bar_j-phi_bar_i,
                               phi^r_j-phi^r_i)
```

Here `phi^r_i=(-offset^r_i) mod 1` is the row's canonical touchdown phase.

Classification runs only after the configured complete-cycle and per-foot
touchdown/liftoff coverage minima are met. It returns `unclassified` if the
best score exceeds `classifier_max_error_cycles` or the second-best-minus-best
margin is below `classifier_min_margin_cycles`. Row/family predictions,
accuracy, confusion, coverage, complete-cycle count, and failure reasons are
preserved per cell; insufficient domains are never silently averaged.

Velocity and gait fidelity remain separate domains. For measurement samples,
with `e_x=v_x-v_x_cmd` and configured numerical `epsilon`, the velocity fields
use:

```text
vx_RMSE       = sqrt(mean(e_x^2))
vx_MAE        = mean(abs(e_x))
vx_bias       = mean(e_x)
gain          = mean(v_x) / v_x_cmd
relative_RMSE = vx_RMSE / (abs(v_x_cmd) + epsilon)
pair_bias(v)  = abs(mean(v_x | +v) + mean(v_x | -v))
pair_bias_norm(v) = pair_bias(v) / (2*abs(v) + epsilon)
```

All means above use the cycle-trimmed measurement slice. The command-direction
sign-error fraction counts only products `v_x_cmd*v_x<0`; an exact zero is not
a sign error. The 5/10/20-percent bands test
`abs(e_x)<=p(abs(v_x_cmd)+epsilon)`, and gain is `N/A` when
`abs(v_x_cmd)<=epsilon`. Lateral-velocity RMSE/MAE, yaw-rate RMSE/MAE, wrapped
heading-error RMSE/p95, lateral-position RMSE/p95, directed progress, progress
per commanded distance, and termination/loss-of-progress status are retained.
The configured `tracking.yaw_rmse_limit_radps` controls yaw tracking
qualification. `tracking.vx_relative_error_limit` controls the separately
reported `vx_relative_tracking_success` diagnostic. The established planar
qualification remains `tracking_rmse_mps <= 0.05 + 0.25*abs(vx_command)` so
default online report selection is unchanged.

Rise and settling use the complete recorded transient, not the trimmed slice.
Rise time is the first sample with commanded-direction velocity at least the
configured fraction of `abs(mean(full command))`. Settling is the first index
whose entire remaining suffix stays within the configured relative band. If
`T` is the full sample count, `k_m` the measurement start, and `k_s` the
settling index, the post-settle measurement fraction is
`(T-max(k_s,k_m))/(T-k_m)`; settling time and this fraction are `N/A` if no
such suffix exists. Heading metrics are independently `N/A` when authentic
heading state is unavailable, without invalidating velocity metrics.

Cell success thresholds live in `study.json`; they are protocol inputs rather
than scientific constants. Under the default rules, velocity-only success
requires no termination, positive command-direction progress, bounded relative
x-velocity RMSE, and bounded yaw-rate RMSE. Gait-only success requires no
termination, sufficient contact agreement, the correct gait family, and the
configured cycle/event coverage; it does not require positive progress. The
resolved `require_no_termination`, `require_positive_progress`, and
`require_correct_family` flags can disable their respective checks. Joint
success is the conjunction. When the gait domain is invalid, gait-only and
joint success are `N/A`, not failures, and their domain coverage remains
explicit.

Configured actuator limits are resolved from each articulation actuator in the
exact action-joint order. The archive records joint names, source per joint,
source-file hash, resolved vector, and fallback status. Publication normalized
load analysis rejects a missing, non-finite, non-positive, fallback, or solver-
sentinel limit. For leg `i`, its three joints `J_i`, analyzed samples `k`, and
control step `dt`, the integrated load definitions are:

```text
u_tau2_i = dt sum_k sum_{j in J_i} tau_kj^2
u_norm_i = dt sum_k sum_{j in J_i} (tau_kj / limit_j)^2
u_work_i = dt sum_k sum_{j in J_i} |power_kj|
         = dt sum_k sum_{j in J_i} |tau_kj qdot_kj|
u_grf_i  = dt sum_k max(Fz_ki, 0)
```

Vertical impulse integrates nonnegative ground-filtered `Fz` over every sample;
the contact state is required to validate the GRF domain and to compute contact
and impact summaries, but it does not gate the impulse sum. Totals per second
divide by `T*dt`; per-cycle totals are `N/A` for zero/unknown complete cycles;
per-directed-metre totals are `N/A` for nonpositive directed progress.

For any nonnegative per-leg exposure `u`, concentration fields are:

```text
CV(u)          = std(u) / (mean(u) + epsilon)
maximum share  = max_i(u_i) / (sum_i(u_i) + epsilon)
max:min        = max_i(u_i) / (min_i(u_i) + epsilon)
front/hind     = ((FL+FR) - (RL+RR)) / (sum_i(u_i) + epsilon)
left/right     = ((FL+RL) - (FR+RR)) / (sum_i(u_i) + epsilon)
```

Signed and absolute imbalances, worst-leg identity/value, raw and normalized
torque-squared exposure, absolute work, and vertical impulse are reported per
second, complete gait cycle, and positive directed metre. Joint tables include
worst normalized torque-squared and absolute-work identities, p95/p99 torque
utilization, saturation fraction, processed-target soft-limit utilization,
and action-clamp fraction. Foot tables include contact-only mean/p95/p99/max
vertical force, total impulse, and touchdown impact peak/impulse summaries.
An impact window has
`max(1,ceil(impact_window_s/dt))` samples beginning at each measured touchdown;
no pre-impact baseline is subtracted. No contact samples makes force summaries
`N/A`; no touchdown makes impact event count zero and impact summaries `N/A`.

Concentration ratios remain epsilon-regularized even when all values or a
minimum leg value are zero. `zero_leg_count` and a max:min reason field expose
that condition; the deterministic worst-leg tie is FL. The older percent
front/hind field is instead `N/A` when total exposure is nonpositive. These
quantities are load-allocation and concentration proxies only; they do not
estimate fatigue life or failure probability.

```powershell
.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --run 2026-08-21_example --model 19999 `
  --expected_branch jding/proprio-history-trs-v5
```

Use the exact light and full commands below for a resolved checkpoint:

```powershell
.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --checkpoint C:\path\to\model_19999.pt --protocol light `
  --expected_branch jding/proprio-history-trs-v5

.\scripts\symm_locomotion\evaluation.ps1 `
  --robot go2 --checkpoint C:\path\to\model_19999.pt --protocol full `
  --expected_branch jding/proprio-history-trs-v5
```

The utility resolves `--run`, `--model`, and `--checkpoint latest` in the same
way as `play` and `record`. It compiles the selected gait and velocity sequence
into a shared `study.json` plan consumed by the playback process. This keeps
the model loaded while the runner executes one isolated gait/velocity cell at
a time and prints completed-cell counts and an ETA. Available plan controls are:

```text
--protocol full|light|legacy
--velocities -1.5 -1.0 -0.5 0.5 1.0 1.5
--gait_indices 0 1 2 3 4 5 6 7 8 9
--settle_s 5.0
--measure_s 10.0
--evaluation_seed 42
--evaluation_config PATH_TO_JSON
--render_cell_plots
--resume
--analyze_only
```

Custom `--velocities` and `--gait_indices` belong to the deprecated `legacy`
profile. Immutable `full` requires its exact ten-by-six inventory, while
immutable `light` rejects any nondefault list and always resolves its eight
named/partner cells. Timing, evaluation seed, evaluation-config JSON, plot
choice, and ordered runtime overrides become part of each publication
protocol's study identity. Legacy v1 does not accept `--evaluation_config`.
Settling and measurement durations must be exact positive multiples of the
robot control step (`0.02 s` for Go2 and X1). Forwarded runtime options cannot
override the plan, task, checkpoint, video mode, environment count, seed, or RL
library. Detailed per-cell plots are off by default; the compressed raw arrays
needed for analysis are always retained.

All artifacts are stored directly under the resolved training run, without a
checkpoint-named intermediate folder. Full-v3 uses
`<training-run>/evaluations/leg_usage_grid_full_v3/`; light uses
`<training-run>/evaluations/leg_usage_grid_light/`; and legacy-v1 uses
`<training-run>/evaluations/leg_usage_grid/`. Each has this layout:

```text
<evaluation-root>/
  study.json
  cells/gait_00_trot/vx_neg_0p5/seed_0042/
    sim_data.npz
    metadata.json
    status.json
    recording_manifest.json
  metrics/cell_metrics.csv
  metrics/family_metrics.csv
  metrics/stratified_fidelity.csv
  metrics/stratified_fidelity.json
  metrics/overall_metrics.json
  metrics/analysis_provenance.json
  metrics/coverage.csv
  metrics/joint_metrics.csv                    # when rows exist
  metrics/foot_metrics.csv                     # when rows exist
  metrics/gait_classifications.csv             # when rows exist
  metrics/gait_confusion_matrix.csv            # when rows exist
  metrics/same_phase_pair_metrics.csv          # when pairs exist
  metrics/directional_pair_metrics.csv         # when partner cells exist
  metrics/REPORT.md
  metrics/SCREENING_REPORT.md
  figures/trot.svg
  figures/bound.svg
  figures/half_bound.svg
  figures/gallop.svg
  figures/overall.svg
  figures/coverage.svg
  figures/screening_report.svg
```

The manifest records the resolved checkpoint path, iteration and SHA-256,
source/configuration hashes, exact grid, timing, and seed. A folder can contain
only one checkpoint/protocol: a mismatch is rejected instead of overwriting or
mixing data. Use `--resume` to continue an interrupted identical grid, or
`--analyze_only` (`--analyze-only` remains an alias) to regenerate summaries
from its existing cell files. Any
Isaac Lab/Hydra arguments forwarded after `--` are recorded in order as part of
the immutable protocol; repeat them for `--resume` or `--analyze_only`.
Analyze-only verifies the recorded manifest and checkpoint without requiring
the current source hash to equal the recording source. It writes the current
analyzer, metrics-module, and Git identity to `metrics/analysis_provenance.json`.

Scientific domains remain independent. Velocity is valid from finite required
kinematic arrays; heading has its own authentic-state mask. Gait requires the
per-cell recording manifest, the configured force-source policy, valid
thresholds, and cycle/event coverage. Raw torque/work does not depend on contact or effort
limits. GRF load requires an available validated ground-force and contact-
classification path, but not successful gait coverage/classification.
Normalized load additionally requires the exact non-fallback effort-limit
provenance. A
failure in one domain is `N/A` there and does not erase other valid domains.
`coverage.csv`, both reports, and `overall_metrics.json` expose these counts;
`stratified_fidelity.*` repeats domain denominators by gait row, family,
velocity, direction, and checkpoint/policy hash.

Aggregation skips `None` and non-finite values and returns `None` for an empty
set. Metric-specific headline values are emitted only when every planned cell
is valid in that metric's domain; explicitly named `observed_*` fields retain
partial-domain summaries. The overall legacy-compatible `complete` flag uses
the velocity, gait, and normalized-load intersection, while raw-load, GRF,
heading, classification, and the three success domains retain separate
coverage. Reports render unavailable scalars as `N/A`, never as zero.

Front/hind imbalance is `100 * (front - hind) / (front + hind)`. Tables retain
that signed value for diagnosis, while primary summaries average its absolute
value so forward/backward or gait-row signs cannot cancel. Rows and velocities
are equally weighted within each family; the overall value is an equal mean of
the trot, bound, half-bound, and gallop family means. Missing, short, terminated,
invalid, and nonpositive-progress cells remain explicit in coverage outputs.
Per-distance metrics are reported only for positive commanded-direction
progress. Tracking quality is retained, not filtered: each cell must pass both
`tracking_rmse_mps <= 0.05 + 0.25 * abs(vx)` and
the configured yaw limit (`yaw_tracking_rmse_radps <= 0.05` by default).
Category and overall figures use
tracking-qualified cells; tables retain both views. Normalized
torque uses the 12 action-ordered effort limits recorded from the loaded robot.
Directed progress uses commanded-sign world-x displacement across the same
intervals as effort integration, with integrated body-x velocity retained as a
consistency diagnostic.

Extra Isaac Lab or Hydra overrides can be passed after `--`. Launcher arguments
before that delimiter are parsed strictly, and the delimiter itself is not
forwarded. For compatibility, delimiter-free Hydra `key=value` overrides are
also accepted, while unknown `--options` are rejected by the launcher:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --no-trs -- \
  env.commands.base_velocity.ranges.lin_vel_x='(-1.0, 2.0)'
```

## Gait-closure run comparison

`comparison.py` contains the CLI, validation, aggregation, and rendering for
the reusable gait-closure comparison. It accepts any ordered set of two or more
completed `leg_usage_grid_full_v3` runs. Each `--run` value
must be `ABBREVIATION=RUN_NAME`; the abbreviation is the compact label used in
the legends. Run arguments retain their command-line order, except that the
explicit `--baseline` is placed first in comparison tables and figures. The
baseline is always black.

The following command reproduces the five-run Go2 comparison. The resolver
checks the normal experiment root first and then the curated `good_runs` root;
this matters because the final run currently lives only in `good_runs`.

```powershell
.\isaaclab.bat -p .\scripts\symm_locomotion\comparison.py `
  --run "NoTRS=2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43" `
  --run "Low-r0=2026-08-30_09-45-19_trs_m0p1_v0p05_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43" `
  --run "Low-r500=2026-08-31_00-14-06_trs_m0p1_v0p05_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43" `
  --run "High-r0=2026-08-30_09-45-34_trs_m0p2_v0p1_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43" `
  --run "High-r500=2026-08-31_00-14-42_trs_m0p2_v0p1_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43" `
  --baseline NoTRS `
  --run_root .\logs\rsl_rl\unitree_go2_symm_flat `
  --run_root .\logs\rsl_rl\good_runs\unitree_go2_symm_flat `
  --output_dir .\logs\rsl_rl\unitree_go2_symm_flat\gait_closure_v4_analysis
```

The equivalent platform launchers are `comparison.ps1` on Windows and
`comparison.sh` on Linux/macOS. They forward every argument to
`comparison.py` through the shared environment launcher. For example:

```powershell
.\scripts\symm_locomotion\comparison.ps1 --manifest `
  .\logs\rsl_rl\unitree_go2_symm_flat\gait_closure_v4_analysis\study.json
```

```bash
bash scripts/symm_locomotion/comparison.sh --manifest \
  logs/rsl_rl/unitree_go2_symm_flat/gait_closure_v4_analysis/study.json
```

Do not confuse this analysis launcher family with the deprecated `compare.py`,
`compare.sh`, and `compare.ps1` compatibility launchers. Those launchers
forward to the supported `compare` subcommand, which only prints recent run
directories and their latest checkpoint names; it does not load evaluation
data, aggregate metrics, or generate comparison artifacts. Use
`symm_locomotion.ps1 compare` or `symm_locomotion.sh compare` while the
compatibility launchers complete their deprecation cycle.

The example's treatment order deliberately preserves the v2 semantic colors:
Low-r0 is blue, Low-r500 is light blue, High-r0 is orange, and High-r500 is
green. `--run_root` is repeatable. When it is omitted, the two roots shown
above are the defaults. Identical archived copies resolve to the first root;
copies that differ in any comparison-consumed evaluation, checkpoint,
training-metadata, or TensorBoard input are rejected as ambiguous.
Duplicate abbreviations, duplicate resolved paths, a missing baseline, and
incompatible evaluation protocols are also errors. The v2 palette provides
unique colors for up to eight runs, including the black baseline; larger
cohorts are rejected instead of reusing ambiguous colors. `--repo_root` and
`--evaluation_subdir` may be used for a non-default checkout or evaluation
folder. The default evaluation subdirectory is
`evaluations/leg_usage_grid_full_v3`.

### Required run artifacts

The comparison consumes analyzer outputs rather than recalculating metrics
from simulation arrays. Each selected directory must have this structure:

```text
RUN_NAME/
  events.out.tfevents.*
  model_<checkpoint_iteration>.pt
  provenance/                         # normal modern training metadata
    initialization.json
  evaluations/
    leg_usage_grid_full_v3/
      study.json
      progress.json
      metrics/
        analysis_provenance.json
        cell_metrics.csv
        overall_metrics.json
```

Legacy curated runs that predate initialization provenance may instead provide
`params/agent.yaml`, `params/env.yaml`, `model_0.pt`, and a registered Git
snapshot. The comparison accepts this fallback only after the run's study
registry and every registered artifact hash validate. It marks the weaker
metadata source in its manifest and provenance and adds a report warning
because pre-rollout RNG and runtime identity cannot be reconstructed.

The checkpoint named by the evaluation `study.json` must still be present and
must match its recorded SHA-256. The study must contain the same robot, task,
gait library and definitions, velocity grid, evaluation seed, timing, contact
configuration, and effort-limit provenance for every run. Every requested
metric domain must be complete. If `cell_metrics.csv` and its provenance do not
exist yet, first run the full-v3 evaluator's analyze-only workflow; that step
also requires the archived cell `sim_data.npz`, metadata, status, and recording
manifest files.

Other normal run artifacts, including `provenance/cohort_metadata.json`,
`provenance/resolved_command.json`, family/stratified metric tables, and raw
cell recordings, remain useful audit material but are not read by this
comparison once the validated full-v3 summaries exist.

### Manifest and reproduction modes

A successful direct run writes `study.json` with the resolved paths, ordered
abbreviations, colors, baseline, protocol identity, and plotting settings.
`analysis_provenance.json` records the input hashes together with the exact
unified `comparison.py`, `comparison.sh`, `comparison.ps1`, and TensorBoard
scalar-parser hashes. The output also includes a small
`reproduce.py`.
`source_manifest_snapshot.json` preserves the exact
manifest supplied to a manifest-mode run; in direct mode it mirrors the
generated canonical study. Either entry point can regenerate the same analysis:

```powershell
.\isaaclab.bat -p .\scripts\symm_locomotion\comparison.py `
  --manifest .\logs\rsl_rl\unitree_go2_symm_flat\gait_closure_v4_analysis\study.json

.\isaaclab.bat -p `
  .\logs\rsl_rl\unitree_go2_symm_flat\gait_closure_v4_analysis\reproduce.py
```

In manifest mode, `--output_dir` defaults to the manifest's parent directory.
Do not hand-edit generated CSV files or figures; edit the manifest or rerun the
direct command so the recorded configuration remains reproducible.

### Aggregation and metric definitions

The three velocity scopes are all nonzero commands, negative commands, and
positive commands. Within a commanded gait family, each valid gait-row,
velocity, and evaluation-seed cell has equal weight. Overall values are the
equal mean of the family means:

```text
family_mean(m) = mean(m(cell) for cells in that commanded family and scope)
overall(m)     = mean(family_mean(m) for each commanded gait family)
```

This prevents the four half-bound rows and four gallop rows from outweighing
the single trot and bound rows. The script uses all valid cells and never
filters a policy by tracking, gait, or joint-success outcomes. The five-run Go2
grid has 60 cells per run: 6 trot, 6 bound, 24 half-bound, and 24 gallop. Each
direction contains half of those cells.

Forward-velocity tracking is
`sqrt(mean((vx - vx_command)^2))` in `[m/s]`; yaw-velocity tracking is
`sqrt(mean((yaw_rate - yaw_rate_command)^2))` in `[rad/s]`. Gait agreement is
the boundary-excluded desired-versus-measured contact agreement, reported in
percent.

All four leg-usage panels report the mean absolute per-cell front/hind
imbalance, `100 * abs(front - hind) / (front + hind)`, where front is FL+FR and
hind is RL+RR. Torque squared is the raw actuator exposure
`sum(integral(torque^2 dt))` in `[N^2 m^2 s]`. Normalized torque squared is the
distinct utilization exposure `sum(integral((torque / effort_limit)^2 dt))` in
`[s]`; it uses each action-ordered joint's physical configured effort limit.
Its canonical comparison ID is `normalized_torque_squared`; its archived
full-v3 source prefix is `normalized_torque_utilization`.
The other two sources are absolute work `sum(integral(abs(power) dt))` in `[J]`
and positive vertical-GRF impulse in `[N s]`. Their plotted imbalance is
dimensionless percent in every case; the integral units describe the front and
hind quantities used to form that ratio.

### Figure catalog

Every figure is emitted as PNG and SVG with stable IDs and filenames:

- **Fig1** (`fig01`) — training efficiency, with the learning-curve, sample-efficiency,
  and wall-time content and styling of the v2 template.
- **Fig2** (`fig02`) — a 3-by-2 overall velocity grid. Rows are all, negative, and
  positive velocities; columns are forward-velocity RMSE and yaw-velocity
  RMSE.
- **Fig3.1**, **Fig3.2**, and **Fig3.3** (`fig03_01`, `fig03_02`, and
  `fig03_03`) — two stacked gait-family velocity panels for all, negative, and
  positive velocities. **Fig3** (`fig03`) stacks those three figures from top
  to bottom.
- **Fig4** (`fig04`) — three overall leg-usage rows, one for each velocity
  scope. Every row contains raw torque squared, normalized torque squared,
  absolute work, and vertical-GRF impulse.
- **Fig5.1**, **Fig5.2**, and **Fig5.3** (`fig05_01`, `fig05_02`, and
  `fig05_03`) — four stacked gait-family leg-usage panels for all, negative,
  and positive velocities. **Fig5** (`fig05`) stacks those three figures from
  top to bottom.
- **Fig6** (`fig06`) — three stacked overall gait-agreement panels for all,
  negative, and positive velocities.
- **Fig7.1**, **Fig7.2**, and **Fig7.3** (`fig07_01`, `fig07_02`, and
  `fig07_03`) — one gait-family agreement panel for all, negative, and positive
  velocities. **Fig7** (`fig07`) stacks those three figures from top to bottom.

The summary CSVs are tidy long-form data: one plotted run/metric/scope/family
value per row, with the cell count, family count, unit, direction of
improvement, and delta from the designated baseline. The combined figures
reuse their numbered component data and do not perform a second aggregation.
For the five-run, four-family cohort, `evaluation_cells.csv` has 300 rows;
the velocity, leg-usage, and gait-fidelity summary tables have 150, 300, and 75
rows respectively.

## Logs

The script maps each robot to its task and experiment directory:

```text
go2 train: Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
go2 play:  Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
go2 logs:  logs/rsl_rl/unitree_go2_symm_flat/

x1 train:  Isaac-Velocity-Flat-Dobot-X1-Symm-v0
x1 play:   Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0
x1 logs:   logs/rsl_rl/dobot_x1_symm_flat/
```

`compare` prints recent run folders and latest checkpoints across robots.
`tensorboard` starts TensorBoard through `python -m tensorboard.main`. On
Windows, multiple selected robots use the shared `logs/rsl_rl/` root because
TensorBoard's named logdir grammar conflicts with drive letters. On Linux and
macOS, the launcher uses named log directories for the selected robots.

## Background Runs

On Linux, `train.sh` and `ablation.sh` support `--nohup`:

```bash
bash scripts/symm_locomotion/train.sh --nohup --robot go2 --iterations 20000 --no-trs
bash scripts/symm_locomotion/ablation.sh --nohup --robot x1 --iterations 20000 --seeds 1 2 3
```

Logs are written under:

```text
logs/symm_locomotion/
```

## Adding Another Robot

Add a new robot in four places:

1. Register the Isaac Lab tasks under
   `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/<robot>_symm/__init__.py`.
2. Put robot-specific assets, joint names, body names, contact sensors, default
   joint positions, height ranges, and actuator gains in
   `config/<robot>_symm/flat_env_cfg.py`.
3. Add robot morphology constants or a small adapter in
   `mdp/<robot>_symm.py` only when the shared defaults do not match.
4. Add one `RobotSpec` entry and any aliases in `symm_cli.py`.

Keep shared reward math, gait command logic, time-reversal transforms, wrapper
behavior, and PPO defaults in the `symm_quadruped` modules unless the behavior
is genuinely robot-specific.
