# Symmetric-locomotion study manifests

`launch_study.py` resolves a complete, immutable training plan before starting
any simulator process. It requires an explicit Git branch, rejects duplicate
conditions and undeclared condition differences, and never deletes or selects
runs based on their outcome.

Preview a study without writing files or launching Isaac Lab:

```powershell
.\isaaclab.bat -p scripts\symm_locomotion\launch_study.py study.json `
  --dry-run --no-conda-run
```

Launch the resolved conditions sequentially:

```powershell
.\isaaclab.bat -p scripts\symm_locomotion\launch_study.py study.json
```

If that deterministic output directory already contains the same resolved
study, pass `--resume` to retain completed runs, mark an interrupted `running`
entry for retry, and launch only the remaining/retryable conditions. Resume
rejects a directory whose stored manifest or plan identity differs.

The default output is the ignored directory
`logs/rsl_rl/studies/<study-name>-<manifest-hash>/`. `study_plan.json` captures
the initial resolved plan, per-run JSON files under `contexts/` are passed to
the maintained RSL-RL training entrypoint, and `cohort_manifest.json` retains
the status and exit code of every attempted run. A failed run does not prevent
later conditions from launching.

## Four-condition example

This example varies only the canonical policy and value enable switches. The
coefficient of a disabled term remains recorded but its effective contribution
is zero. Policy, value, and augmentation schedules are independent.

```json
{
  "schema_version": 1,
  "study_name": "day1-go2-policy-value",
  "expected_branch": "jding/proprio-history-trs-v5",
  "treatment_variables": [
    "tr_policy.enabled",
    "tr_value.enabled"
  ],
  "common": {
    "robot": "go2",
    "seed": 42,
    "tr_policy": {
      "enabled": false,
      "coefficient": 0.1,
      "schedule": {
        "enabled": true,
        "warmup_iterations": 500,
        "rampup_iterations": 1500,
        "hold_iterations": 0,
        "decay_iterations": 0,
        "final_scale": 1.0,
        "ramp_shape": "half_cosine"
      }
    },
    "tr_value": {
      "enabled": false,
      "coefficient": 0.05,
      "schedule": {
        "enabled": true,
        "warmup_iterations": 500,
        "rampup_iterations": 1500,
        "hold_iterations": 0,
        "decay_iterations": 0,
        "final_scale": 1.0,
        "ramp_shape": "half_cosine"
      }
    },
    "validity_mask": {
      "mode": "command",
      "thresholds": {
        "min_abs_command_velocity": 0.0,
        "tracking_abs_tolerance": 0.25,
        "tracking_rel_tolerance": 0.25,
        "projected_gravity_tolerance": 0.35,
        "phase_boundary_margin": 0.03,
        "command_speed_bin_edges": [0.5, 1.0, 1.5]
      }
    },
    "trajectory_augmentation": {
      "enabled": false,
      "mode": "dynamics_filtered_reverse_action_supervision",
      "action_source": "analytic",
      "coefficient": 0.0,
      "schedule": {
        "enabled": true,
        "warmup_iterations": 0,
        "rampup_iterations": 0,
        "hold_iterations": 0,
        "decay_iterations": 0,
        "final_scale": 1.0,
        "ramp_shape": "linear"
      },
      "filter_settings": {
        "filter_enabled": true,
        "validation_fraction": 0.2,
        "validation_quantile": 0.95,
        "threshold_multiplier": 2.0,
        "minimum_validation_samples": 128,
        "maximum_validation_loss": 1.0,
        "phase_boundary_margin": 0.03,
        "maximum_contact_impulse": 20.0,
        "action_abs_limit": 10.0,
        "max_augmented_to_original_ratio": 0.25,
        "use_confidence_weights": true,
        "rng_seed": 0
      }
    },
    "reward_profile": {
      "name": "default-v1",
      "overrides": []
    },
    "leg_sync_reward_weight": 0.2,
    "training_iterations": 20000,
    "num_envs": 512,
    "evaluation_protocol": "light"
  },
  "conditions": [
    {
      "label": "no-trs"
    },
    {
      "label": "actor-only",
      "tr_policy": {
        "enabled": true
      }
    },
    {
      "label": "value-only",
      "tr_value": {
        "enabled": true
      }
    },
    {
      "label": "actor-value",
      "tr_policy": {
        "enabled": true
      },
      "tr_value": {
        "enabled": true
      }
    }
  ]
}
```

## Exact schedule and resume semantics

Each manifest `coefficient` becomes the resolved schedule `target_coeff` for
that term. Policy, value, and trajectory augmentation remain independent; a
disabled term contributes exactly zero even though its coefficient and
schedule stay in provenance. For absolute PPO update `k>=0`, let
`W,R,H,D,F` be the configured warm-up, ramp-up, hold, decay, and final scale,
with `b=W+R`, `h=b+H`, and `f(p)=p` for `linear` or
`f(p)=0.5*(1-cos(pi*p))` for `half_cosine`:

```text
q(k) = 0                         , k < W
       f((k-W)/R)                , R > 0 and W <= k <= b
       1                         , b <= k <= h
       1 + (F-1) f((k-h)/D)      , D > 0 and h < k <= h+D
       F                         , k > h+D

lambda_eff(k) = coefficient * q(k) when both term and schedule are enabled,
                otherwise 0
```

The ramp and hold share their value-one endpoint. `R=0` gives a hard switch at
`k=W`; `D=0` uses `F` on the first update after the hold. New checkpoints bind
the exact three resolved schedules to consecutive last-completed and
next-absolute update indices. Loading resumes at that next absolute update and
rejects schedule mismatch. Pre-schema checkpoints use the legacy `iter+1`
fallback. Enabled trajectory augmentation also requires its matching semantic
configuration, side models/EMA targets, optimizers, normalizers, calibrated
filter, private RNG, sidecar, and schedule iteration; it never silently starts
those components from random state.

To add an actor-plus-value trajectory-augmentation treatment, declare
`trajectory_augmentation.enabled`, `trajectory_augmentation.action_source`,
and any changed coefficient, schedule, or filter path as treatment variables.
Then use a condition overlay such as:

```json
{
  "label": "actor-value-learned-augmentation",
  "tr_policy": {"enabled": true},
  "tr_value": {"enabled": true},
  "trajectory_augmentation": {
    "enabled": true,
    "action_source": "learned_inverse",
    "coefficient": 0.02,
    "schedule": {
      "warmup_iterations": 2000,
      "rampup_iterations": 2000
    }
  }
}
```

`analytic`, `learned`, and `learned_inverse` are accepted action-source input
values; `learned` is normalized to the canonical `learned_inverse` value.
Enabled augmentation requires `filter_enabled: true`; unfiltered reverse-action
supervision fails manifest validation.

Enabled trajectory augmentation is actor-NLL supervision only. Reversed
samples never enter PPO ratios, clipping, GAE, returns, critic targets,
adaptive KL, or entropy averaging. Its Markov state treats positions,
orientation, controller/history targets, gait ratios/period, masses, and
materials as even; velocities and the velocity command as odd; reflects common
phase about swing ratio; negates foot offsets; and maps the gait row to its
declared time-reversal partner. The analytic action map is identity only for
the maintained position-target controller. Learned inverse dynamics removes
actuator-target and previous-action features from both endpoints before
prediction, then reconstructs those channels from its learned actions.

The mandatory filter requires finite data, bounded action, safe phase margins
at both endpoints, no touchdown/liftoff crossing, bounded control-step contact
impulse, a component-aware forward residual below its held-out quantile
threshold, and a ready validation model. No authentic terminal state is
available after auto-reset, so every done/timeout transition is excluded.
Recurrent policies, multi-GPU augmentation, nonempty hidden actuator state,
missing mass/material state, wrapper action clipping, and controllers outside
the single affine joint-position term are unsupported and fail closed where
detectable. These gates are heuristics, not a proof of physical reversibility;
the full equations, empty-pool rule, resource formulas, and limitations are in
[README.md](README.md#publication-time-reversal-controls).

Reward profiles have a stable name plus sorted, unique Hydra overrides scoped
to `env.rewards.*`. The leg synchronization weight has its own field and
cannot be hidden inside a reward-profile override.

The reward-profile name is provenance, not a hidden preset selector. The
resolved overrides determine training behavior. Conditions that differ only in
that name are therefore rejected as semantic duplicates instead of being
misrepresented as distinct treatments. Augmentation mode is also validated by
the manifest loader; the only maintained value is
`dynamics_filtered_reverse_action_supervision`.

## Initialization and cohort validation

Immediately after runner construction, optional checkpoint loading, and
resolved-config dumping—but before `runner.learn`—a time-reversal run writes:

```text
<run>/provenance/initialization.json
```

The write is exclusive and the record contains a digest of all other fields.
It captures the Python, NumPy, CPU-Torch, and per-device CUDA RNG states;
environment and training seeds; actor, critic, distribution, and optimizer
state hashes; resolved configuration hashes; source-asset hashes; and Git
commit/dirty-tree identity. The dirty-tree digest uses Git raw blob/status
metadata and streaming hashes of changed files, so it does not materialize
large binary patches in memory.

Local robot assets are hashed file by file. The symmetric Go2 and X1 task
configs currently resolve to repository-local asset trees, so their bytes and
dependencies are included in the initialization record. Nucleus or other
external asset URIs remain in the record with `status: unresolved_external`
and produce an explicit risk warning; they are never represented by a
misleading empty asset hash. A record with no asset reference, a missing local
asset, or an external asset without an authoritative content hash fails
publication cohort validation.

For a resume, the new run records the loaded checkpoint identity and validates
the source run's initialization record without modifying it. A checkpoint from
before this feature is marked `missing_legacy_run` rather than receiving a
fabricated source record.

After all launches, the cohort validator requires identical seeds, actor and
critic initializations, distribution parameters, optimizer state and settings,
architecture, reward profile, gait library, command distribution, environment
randomization, training budget, repository/diff identity, and robot assets.
Initialization differences can never be declared away as treatments. Run the
validator directly with:

```powershell
.\isaaclab.bat -p scripts\symm_locomotion\training_provenance.py `
  logs\rsl_rl\studies\<study>\cohort_manifest.json
```

The initialization record embeds the canonical resolved environment and agent
configurations as well as their hashes. Validation first binds the record to
the cohort's study name/hash, expected branch, run name, condition hash, exact
training command, context path, and declared seed. It then compares the full
resolved configurations after masking only two operational paths
(`environment.log_dir` and `agent.run_name`) and exact runtime paths derived
from declared manifest treatments. The treatment mapping is explicit in
`training_provenance.py` and is emitted as `runtime_treatment_paths` in the
validation result. A changed termination, observation, action, simulator, PPO,
or other unmapped field therefore makes `matched` false even when the compact
manifest conditions look identical. Reward override treatments are mapped to
their exact resolved `rewards.*` fields; the reward-profile name maps to no
runtime field because it is provenance only.

The launcher records but does not automatically execute the requested light or
full evaluation. Each run contains an exact `evaluation --protocol
light|full` command template to use after a checkpoint exists.

## Post-training evaluation template

`evaluation_command_template` is an argument-vector list, not an already
formatted shell command. Its `"{run_dir}"` entry is a literal placeholder: the
braces are stored verbatim in `study_plan.json`, each context, and
`cohort_manifest.json`. After a successful launch, replace only that complete
token with the same run's non-null `run_dir` value from `cohort_manifest.json`.
Do not rely on shell variable or brace expansion, and do not evaluate a failed
run whose `run_dir` was never resolved. The resulting `--run` command resolves
the checkpoint and the evaluation `study.json` records its absolute path,
iteration, and SHA-256.

The stored template requests the manifest's exact nominal `light` or `full`
protocol. Light is the fixed eight-cell named/partner screen and rejects custom
velocity or gait-index lists. Full is the immutable ten-row by six-velocity
publication grid and likewise rejects custom lists. Historical/custom v1 work
uses the deprecated `legacy` CLI profile and is not a valid replacement for a
manifest's requested publication protocol. Changing timing, seed, evaluation
thresholds, cell plots, or forwarded runtime overrides changes the publication
evaluation identity. Resume and analyze-only accept only the identical
checkpoint, protocol identity, and ordered runtime overrides.

Completed full artifacts live under
`{run_dir}/evaluations/leg_usage_grid_full_v3/`; light artifacts live under the
distinct `{run_dir}/evaluations/leg_usage_grid_light/`. Each completed cell
contains `sim_data.npz`, `metadata.json`, `status.json`, and
`recording_manifest.json`. Reports and tables live under `metrics/`, including
`coverage.csv`, `stratified_fidelity.csv/.json`, and domain-specific optional
tables; SVGs live under `figures/`. The exact evaluation equations, independent
domain/empty-set rules, output tree, and nominal 8-cell versus 60-cell overhead
are documented in [README.md](README.md#policy-evaluation).
