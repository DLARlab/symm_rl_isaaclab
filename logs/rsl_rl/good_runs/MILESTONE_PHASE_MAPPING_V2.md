# 72D Phase Mapping V2 milestone

Date: 2026-08-03
Branch: `jding/symm-72d-trs-milestone`
Runtime audit tag: `same_gait_backward_duty_aware_trs_v2`

## Executive result

Phase Mapping V2 fixes a semantic conflation in the previous 72D environment:
ordinary backward locomotion and the mathematical time-reversal (TR) pairing
used by the auxiliary PPO losses are now two different operations.

- Ordinary positive and negative commands use the same forward-time gait row,
  fixed leg offsets, and footfall order. Only the desired spatial velocity
  changes sign.
- The auxiliary TR pair uses a duty-factor-aware phase reflection. It reverses
  temporal event order while preserving each leg's instantaneous swing or
  stance mode.
- The 72D observation contract, 12D position-target action, reward and
  termination definitions, actor/critic architecture, PPO loss formulas, and
  robot assets are unchanged by the phase-mapping correction.
- PPO transition data augmentation remains disabled. TRS remains an auxiliary
  policy/value inductive bias, not a claim that dissipative contact dynamics
  are exactly reversible.

The matched single-seed studies support the two motivating hypotheses at
selected coefficients:

- For Unitree Go2, TRS `0.20/0.10` substantially reduced several front/hind
  load-concentration proxies relative to no TRS.
- For Dobot X1, the weaker TRS `0.10/0.05` improved descriptive reward sample
  efficiency relative to no TRS.

These results are coefficient-dependent and descriptive. They do not establish
universal TRS gains or a hardware leg-break probability.

## What changed from the previous mapping

Let `wrap(x) = remainder(x, 1)`, `phi` be the common gait clock, `theta_i` the
fixed offset for leg `i`, and `psi_i = wrap(phi + theta_i)`.

The previous environment selected clock orientation from the x-command sign:

```text
vx >= 0: psi_i = wrap( phi + theta_i)
vx <  0: psi_i = wrap(-phi - theta_i)
```

This made a negative spatial command silently remap the gait row and reverse
the event order. A command sign is not a direction of time, so this was the
wrong semantics for ordinary forward-time backward locomotion. The earlier
observation transform also used fixed sine/cosine parity equivalent to
`psi_tr = wrap(-psi)`. That reflection ignores duty factor and does not, in
general, preserve contact mode.

V2 uses the same environment schedule for every command sign:

```text
psi_i = wrap(phi + theta_i)
```

The environment clock is still derived from `episode_length_buf`; no signed or
stateful clock and no clock-direction observation were introduced. Gait period
and duty factor continue to depend on `abs(vx)`. Consequently, equal `phi` and
`theta` give identical phases for `-2`, `-1e-3`, `0`, `+1e-3`, and `+2 m/s`.

## Why the physical TR map is correct

Let `beta` be stance duty factor and `s = 1 - beta` be swing ratio. The task's
mode convention is swing on `[0, s)` and stance on `[s, 1)`. V2 defines

```text
R_beta(psi) = wrap(s - psi)
theta_tr    = wrap(-theta)
phi_tr      = wrap(s - phi)
```

This construction has the required cyclic properties:

1. **Involution.** `R_beta(R_beta(psi)) = psi` modulo one.
2. **Orientation reversal.** Away from the wrap point, the map has slope `-1`,
   so temporal order is reversed.
3. **Mode preservation.** The interior of `[0, s)` maps back into swing, and
   the interior of `[s, 1)` maps back into stance.
4. **Boundary exchange.** `R_beta(0) = s` and `R_beta(s) = 0`. A forward
   touchdown therefore maps to a reversed liftoff, and a forward liftoff maps
   to a reversed touchdown.
5. **One-clock closure.** For every leg,
   `wrap(phi_tr + theta_tr) = wrap(s - phi - theta_i) = R_beta(psi_i)`.
   The four transformed leg phases therefore come from one compatible common
   transformed clock rather than four unrelated phase edits.

Runtime code does not decode phase with `atan2`. With `alpha = 2*pi*s`, it
applies the rotation/reflection directly to unit-circle features:

```text
phase_sin_tr = sin(alpha) * phase_cos - cos(alpha) * phase_sin
phase_cos_tr = cos(alpha) * phase_cos + sin(alpha) * phase_sin
theta_sin_tr = -theta_sin
theta_cos_tr =  theta_cos
```

The pure helper broadcasts `swing_ratio[..., 1]` over four leg channels,
preserves PyTorch device and dtype, supports arbitrary leading batch
dimensions, and rejects incompatible shapes instead of clamping invalid phase
ratios.

## Preserved 72D policy contract

The observation layout is now centralized in a frozen named definition, but
its dimension and concatenation order are unchanged:

| Slice | Width | Quantity | TR treatment |
| --- | ---: | --- | --- |
| `0:6` | 6 | measured base linear/angular velocity | odd |
| `6:9` | 3 | projected gravity | even |
| `9:15` | 6 | desired base twist | odd |
| `15:27` | 12 | relative joint position | even |
| `27:39` | 12 | relative joint velocity | odd |
| `39:51` | 12 | previous position-target action | even |
| `51:55` | 4 | foot-phase sine | duty-aware nonlinear map |
| `55:59` | 4 | foot-phase cosine | duty-aware nonlinear map |
| `59:63` | 4 | foot-theta sine | odd |
| `63:67` | 4 | foot-theta cosine | even |
| `67:68` | 1 | swing ratio | even |
| `68:69` | 1 | stance ratio | even |
| `69:72` | 3 | sagittal-plane state | even |

The action TR map remains an identity clone because the action represents a
joint-position offset. `compute_time_reversal_states()` retains its API and
original-then-transformed tensor pairing for RSL-RL.

## PPO interpretation and invariants

`use_data_augmentation=False` remains configured for Go2 and X1. Transformed
observations therefore do not enter the PPO surrogate or ordinary PPO
value-regression transition batch. They are used by the existing auxiliary
relations

```text
pi(T_obs(s)) ~= T_action(pi(s))
V(T_obs(s))  ~= V(s)
```

The policy mirror/equivariance loss, TR value-consistency loss, actor safety
logic, warm-up, command mask, PPO surrogate, and PPO value loss formulas were
not changed. The standard TRS defaults remain mirror coefficient `0.10`, value
coefficient `0.05`, warm-up `500` iterations, and minimum absolute TR command
velocity `0.0`. A `--no-trs` run disables mirror loss and sets both auxiliary
coefficients to zero, so its TRS enable predicate is false.

This is a structural prior over observations, actions, and values. Friction,
impact losses, actuator losses, and other dissipative contact effects are not
claimed to be physically reversible.

## Gait-library closure

The configured leg order is FL, FR, RL, RR. The closure statements below apply
to the nominal configured `init_foot_thetas` rows before optional independent
theta noise; the finite realized noise support is not exactly closed under
negation.

- Trot is self-reversing.
- Bound is self-reversing.
- Half-bound `H1=(0.13,-0.13,0.5,0.5)` and
  `H2=(-0.13,0.13,0.5,0.5)` are TR partners under `theta -> -theta`.
  Their rear legs are synchronized and the front-leg circular separation is
  `0.26` cycle. At representative swing ratio `0.55`, their cyclic touchdown
  orders anchored at the rear pair are `rear -> FL -> FR` and
  `rear -> FR -> FL`.
- The pure TR partners of the two current gallop rows are not sampled. No new
  gallop rows were added in this milestone.

## Current training preset recorded with this milestone

Several later, separately requested training-preset changes are present in the
current branch and in the analyzed V2 run configurations. They are not part of
the mathematical phase-map fix:

- training and ablation defaults are `20,000` iterations and `512` environments;
- `lin_vel_x` remains `(-2.0, 2.0)`;
- `min_xy_command_norm=0.0`, so sampled low-speed commands are retained rather
  than snapped to zero;
- `rel_standing_envs=0.0`, so exact zero has measure zero in continuous command
  sampling and is not deliberately sampled;
- reset base velocity uses `(-0.5, 0.5)` for x, y, z, roll, pitch, and yaw;
- periodic push velocity uses `(-0.25, 0.25)` for both x and y.

This intentionally differs from the earlier `min_xy_command_norm=0.2`
configuration, where a sampled nonzero command such as `1e-3 m/s` was converted
to exact zero. In this milestone, deployed `+/-1e-3 m/s` is trained as retained
near-zero behavior rather than interpolation from a deadband-generated
standstill command.

No phase-mapping change was made to reward definitions or weights,
terminations, action semantics, task IDs, robot assets, network sizes, or PPO
optimizer settings.

## Matched empirical studies

The detailed, machine-readable reports and plots are archived here:

- [Go2 four-run study](unitree_go2_symm_flat/phase_mapping_v2_go2_trs_run_analysis/REPORT.md)
- [X1 four-run study](dobot_x1_symm_flat/phase_mapping_v2_x1_trs_run_analysis/REPORT.md)

Each comparison uses four seed-42 runs, 20,000 PPO iterations, 512 environments,
the same initial checkpoint within the robot study, and a final-policy 30-second
paired backward/forward rollout. Initial-checkpoint equality was verified before
the archive adopted its latest-checkpoint-only retention policy; the recorded
SHA-256 values remain in the study summaries. The intended TRS coefficients are
the study variable: no TRS, `0.10/0.05`, `0.20/0.10`, and `0.30/0.15`, with
500-iteration warm-up for enabled TRS runs.

### Go2: front/hind load concentration

For direction-equal absolute front/hind imbalance, TRS `0.20/0.10` versus no
TRS changes:

| Proxy | No TRS | TRS 0.20/0.10 | Relative reduction |
| --- | ---: | ---: | ---: |
| Normalized torque utilization | 25.54% | 12.85% | 49.7% |
| Absolute work | 17.82% | 4.51% | 74.7% |
| Rainflow `m=5` sensitivity | 58.06% | 22.29% | 61.6% |
| Vertical GRF impulse | 15.18% | 10.10% | 33.5% |

TRS `0.30/0.15` gives the lowest torque-squared imbalance, changing 22.78% to
10.64% (53.3% reduction). The response is not uniformly better: contact-time
balance worsens, and for `0.20/0.10` total `m=5` sensitivity rises 12.6% even
though the dominant-pair proxy falls 4.7% and the worst-joint proxy falls
41.3%. The supported conclusion is that selected V2 TRS coefficients reduce
concentration between fore and hind pairs on several relevant proxies. The
study does not directly measure fracture, thermal lifetime, gearbox damage, or
leg-break probability.

### X1: reward sample efficiency

TRS `0.10/0.05` is the only setting with a clear useful sample-efficiency
signal in this grid:

| Measure | Change versus no TRS |
| --- | ---: |
| Reward AUC, first 10k iterations | +5.03% |
| Reward AUC, full 20k iterations | +1.84% |
| Samples to sustained reward 30 | -25.65% (`64.28M -> 47.79M`) |

Its final tail reward is slightly lower and its XY/yaw tracking is worse than
the no-TRS policy. TRS `0.20/0.10` is approximately tied on full AUC (`+0.12%`),
while `0.30/0.15` is lower (`-3.59%`). Thus the evidence supports the hypothesis
for the tuned weak regularizer, not a monotonic or universal claim that more
TRS always trains faster.

The previous 60D-to-72D milestone, which used the older semantics and different
archived runs, did not find better fore/hind balance as TRS increased. Phase
Mapping V2 is a corrected formulation and a newly controlled comparison; the
cross-milestone contrast is encouraging but does not by itself prove that the
mapping correction caused every observed performance difference.

## Test coverage and correctness checks

The phase-mapping regression suite covers:

- named 72D layout and unchanged observation order;
- command-sign invariance including zero and near-zero commands;
- phase involution, contact-mode preservation, boundary exchange, and
  unit-circle norm preservation;
- arbitrary leading tensor dimensions, broadcasting, dtype/device retention,
  invalid-shape errors, and no silent ratio clamping;
- full observation involution and one-common-clock closure;
- half-bound geometry, TR partner exchange, and reversed cyclic event order;
- action identity and original/transformed batch ordering;
- configuration invariants, including data augmentation disabled for both
  robots and unchanged auxiliary coefficients/warm-up.

The shared analysis engine additionally validates terminal checkpoints,
resolved environment/agent snapshots after documented normalization, archived
training-source provenance, and matched rollout-array hashes before computing
comparisons. It validates initial-checkpoint equality when a complete initial
set is locally available, records it as unavailable for the published
latest-only archive, and rejects a partially present set.

## Reproducible analysis layout

The two former ~900-line robot-specific reproduction scripts were consolidated
into the maintained manifest-driven utility:

```text
scripts/symm_locomotion/analyze_matched_trs_study.py
```

Each report directory retains a small `reproduce.py` compatibility entry point
and a `study.json` manifest containing only robot/run metadata. Run either:

```powershell
python .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\phase_mapping_v2_go2_trs_run_analysis\reproduce.py
python .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\phase_mapping_v2_x1_trs_run_analysis\reproduce.py
```

or invoke the shared utility with either manifest. The generated `summary.json`
records the analysis-method version and SHA-256 hashes of the shared engine and
manifest for auditability.

The committed `good_runs` tree contains the training event logs, resolved
configuration snapshots, provenance diffs, rollout inputs, plots, recordings,
deployment exports, reports, tables, manifests, and reproduction wrappers.
Each training folder retains exactly one terminal `model_<iteration>.pt`
checkpoint; intermediate and initial training checkpoints remain local and are
excluded from version control. The wrappers therefore reproduce the numerical
study from a fresh clone while explicitly reporting initial-checkpoint
revalidation as unavailable.

## Publication validation record

Validation on 2026-08-03 used the `symm_rl_isaaclab` Conda environment:

- focused phase/PPO/config/CLI/analysis suite: `91 passed`, `8 subtests passed`;
- counterfactual regression check: injecting the former command-sign phase
  branch and fixed `wrap(-psi)` phase transform made the corresponding new
  tests fail, while the V2 implementation passes;
- Go2 analysis wrapper: passed in 50.72 s;
- X1 analysis wrapper: passed in 50.09 s;
- normalized shared-engine hash recorded by both summaries:
  `1f5e5d89e5a063407b2bf2bebd909a98577480ae5b94115180b5db9acb30f0ce`;
- one-iteration Go2 smoke: 72 actor/critic inputs, 12 actions, 24 steps,
  finite losses, checkpoint saved, no shape error or NaN;
- one-iteration X1 smoke: 72 actor/critic inputs, 12 actions, 24 steps,
  finite losses, checkpoint saved, no shape error or NaN.

Both smoke configurations recorded `use_data_augmentation=false`, mirror
coefficient `0.10`, value coefficient `0.05`, and warm-up `500`. At iteration
zero the auxiliary losses were correctly inactive because the run was still in
warm-up. The generated smoke-run directories are validation scratch outputs and
are not part of this milestone artifact.

The repository-wide pre-commit launcher was attempted repeatedly. Ruff, Ruff
format, and all RST hooks passed, but Windows Application Control rejected the
cached hook-specific Python executables with `[WinError 4551]`. The same staged
file set then passed the corresponding trailing-whitespace, symlink,
large-file, merge/case-conflict, EOF, shebang, private-key, debug-statement,
codespell, and license-header checks through the approved Conda interpreter.
This is an execution-policy limitation, not a reported source failure.

## Compatibility boundary

Old 72D checkpoints trained with command-sign-dependent phase reversal have
different negative-command environment semantics. They must be reproduced
with their archived environment source/configuration. The V2 audit tag records
the new semantics but does not alter runtime behavior.

This milestone supplements rather than rewrites the archived
[60D-to-72D milestone](MILESTONE_60D_TO_72D.md).
