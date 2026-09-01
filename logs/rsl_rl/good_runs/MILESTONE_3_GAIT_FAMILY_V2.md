# Milestone 3: Gait-family V2

- Date: 2026-08-24
- Branch: `72d-symm-v4-integration`
- Status: implementation and matched evaluation complete; Go2 reward
  optimization remains open.

Timeline:
[Milestone 1](MILESTONE_1_60D_TO_72D.md) ·
[Milestone 2](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md) ·
**Milestone 3** · [Milestone 4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md)

## What changed

This milestone inherits the duty-aware phase map and synchronized-pair
leg-permutation reward from
[Milestone 2](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md). Its own
change is the time-reversal-closed V2 gait family and a reproducible leg-usage
evaluation workflow. It replaces the earlier incidental two-command playback
comparison with a controlled fixed grid and keeps the training learning curves,
raw rollouts, and source/checkpoint provenance in the curated archive.

The implementation adds:

- canonical stable IDs, family labels, and time-reversal partners for all 10
  training gait rows;
- a fixed evaluation scenario that holds velocity and gait constant without
  consuming command-resampling randomness;
- `analyze_leg_usage.py`, `.ps1`, and `.sh` utility entry points beside the
  existing train, play, and record utilities;
- resumable 60-cell collection with a study lock, atomic artifacts, progress
  and ETA reporting, checkpoint/config/source hashes, and strict resume
  validation;
- family-balanced leg-usage summaries and cohort comparisons that retain yaw
  tracking as a separate audit; and
- archived X1 and Go2 reports in the earlier learning-curve and grouped-bar
  plotting style.

## Gait-library change

The training library contains 10 stable rows in FL, FR, RL, RR order:

- self-reversing trot `(0.0, 0.5, 0.5, 0.0)` and bound
  `(0.0, 0.0, 0.5, 0.5)`;
- two front-spread and two hind-spread half-bound rows paired under
  `theta -> -theta`; and
- four gallop rows that close into two time-reversal partner pairs.

Trot and bound receive sampling weight `4` each, while each half-bound and
gallop row receives weight `1`. This gives the four gait families equal total
training weight despite their different row counts. Training adds small
independent phase-offset noise; the nominal table itself is exactly closed.

Velocity-only and gait-row resampling both refresh period and duty factor from
the final command. The piecewise-integrated common phase stays continuous
across timing changes, and coincident changes share one timing-noise draw. This
prevents period and duty factor from remaining stale after a velocity resample.

## Evaluation protocol

Each terminal checkpoint is evaluated on all 10 training gaits at
`vx = ±{0.5, 1.0, 1.5} m/s`. Every cell is reset independently, settles for
5 seconds, and records 10 seconds for cycle-trimmed measurement. The primary
overall value averages rows within each family and then gives trot, bound,
half-bound, and gallop equal weight.

The archived comparison contains 420/420 valid planar-tracking-qualified cells:
180 for X1 and 240 for Go2. Heading success is reported separately because the
strict yaw threshold would otherwise select a sparse and gait-biased subset.

All training runs use seed 42, 512 environments, 24 rollout steps per
environment, and 20,000 iterations. Every retained training directory keeps
the terminal `model_19999.pt` checkpoint. Every checkpoint completed 60/60
fixed-grid cells without evaluation termination.

Milestone 4 later re-evaluates these same checkpoints with the full-V3 metric
and aggregation contract. Small differences between the original evaluation
tables below and Milestone 4's quoted evaluation baselines come from
reprocessing, not different checkpoints.

## Retained cohort

### Dobot X1

| Started | Configuration | Archived run | Tail reward | Full AUC | Grid |
|---|---|---|---:|---:|---:|
| 2026-08-20 21:04 | No TRS | `2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2` | 42.307 | 34.667 | 60/60 |
| 2026-08-21 06:37 | m0.1/v0.05, w500/r0 | `2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2` | 43.040 | 36.124 | 60/60 |
| 2026-08-21 22:50 | m0.2/v0.1, w500/r0 | `2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2` | 41.081 | 32.683 | 60/60 |

### Unitree Go2

| Started | Configuration | Archived run | Tail reward | Full AUC | Grid |
|---|---|---|---:|---:|---:|
| 2026-08-20 21:04 | No TRS | `2026-08-20_21-04-16_go2_72d_no_trs_gait_trclosed_v2` | 39.493 | 32.084 | 60/60 |
| 2026-08-21 10:22 | m0.1/v0.05, w500/r0 | `2026-08-21_10-22-39_go2_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2` | 37.919 | 31.281 | 60/60 |
| 2026-08-21 22:50 | m0.2/v0.1, w500/r0 | `2026-08-21_22-50-01_go2_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2` | 35.678 | 27.502 | 60/60 |
| 2026-08-22 10:35 | m0.2/v0.1, w500/r1000 linear | `2026-08-22_10-35-19_go2_72d_trs_m0p2_v0p1_w500_r1000_linear_gait_trclosed_v2` | 37.092 | 29.953 | 60/60 |

## Dobot X1 result

| Run | Full AUC | Tail reward | Torque² imbalance | Work imbalance | GRF imbalance |
|---|---:|---:|---:|---:|---:|
| No TRS | 34.667 | 42.307 | 17.687% | 23.917% | 6.999% |
| m0.1/v0.05 | **36.124** | **43.040** | **6.642%** | **12.800%** | **3.806%** |
| m0.2/v0.1 | 32.683 | 41.081 | 9.549% | 18.952% | 4.270% |

For this checkpoint cohort, `m0.1/v0.05` is the observed Pareto winner. It
improves early and full learning, final reward, sample efficiency, front/hind
allocation, and all three per-metre exposure measures relative to no TRS.

## Unitree Go2 result and open issue

| Run | Full AUC | Tail reward | Torque² imbalance | Work imbalance | GRF imbalance |
|---|---:|---:|---:|---:|---:|
| No TRS | **32.084** | **39.493** | 20.038% | 17.725% | 18.245% |
| m0.1/v0.05, w500/r0 | 31.281 | 37.919 | **14.088%** | 22.037% | 26.029% |
| m0.2/v0.1, w500/r0 | 27.502 | 35.678 | 19.607% | **9.621%** | **17.241%** |
| m0.2/v0.1, w500/r1000 | 29.953 | 37.092 | 21.106% | 11.668% | 17.664% |

Go2 is not a completed optimization result. No TRS still has the best reward
and full learning-curve AUC. The low TRS dose gives the best torque-allocation
balance, the hard high dose gives the best work and GRF balance, and the high
dose with a ramp recovers learning and improves selected exposure measures.
None improves both learning and leg usage as a whole.

The next coefficient screen should keep the empirically stronger
`w500/r1000 linear` schedule and separate the actor and critic regularizers:

1. `m0.2/v0.05` tests whether high actor mirror pressure can retain the physical
   gains while reducing the critic-consistency learning tax.
2. `m0.15/v0.05` tests a lower-risk actor-strength interpolation from the best
   current TRS learner.

A winning screen must be repeated on matched independent training seeds before
making a method-level claim.

## Known limitations

- The current comparisons use one training seed and one deterministic
  evaluation seed per checkpoint.
- The primary leg-usage comparison is planar-tracking-qualified. Strict yaw plus
  planar coverage is incomplete, especially for Go2, so heading stability
  remains a separate optimization target.
- Recorded effort limits are the PhysX `1e9 N·m` sentinel. Normalized torque is
  therefore intentionally omitted; torque-squared is reported as a raw load
  proxy, work as energy, and vertical GRF impulse as a contact-load proxy.
- The fixed-grid workflow does not fabricate rainflow-fatigue estimates.

## Archived evidence

- [X1 report](dobot_x1_symm_flat/legacy/gait_family_v2_analysis/REPORT.md)
- [Go2 report](unitree_go2_symm_flat/legacy/gait_family_v2_analysis/REPORT.md)
- [X1 curated-run index](dobot_x1_symm_flat/README.md)
- [Go2 curated-run index](unitree_go2_symm_flat/README.md)

Every retained run includes its TensorBoard event, resolved parameters,
training-time Git provenance, terminal checkpoint, exports, playback artifacts,
and raw fixed-grid evaluation data. The analysis folders contain frozen study
manifests, reproducible scripts, CSV/JSON summaries, and SVG/PNG figures.

The per-run TensorBoard event is the authoritative training log. The resolved
environment and agent snapshots, training-source provenance, deployment
exports, and raw evaluation cells retain the inputs needed to audit the report.

## Reproduction

From the repository root, regenerate the archived comparison products with:

```powershell
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\dobot_x1_symm_flat\legacy\gait_family_v2_analysis\reproduce.py
.\isaaclab.bat -p .\logs\rsl_rl\good_runs\unitree_go2_symm_flat\legacy\gait_family_v2_analysis\reproduce.py
```

The Go2 folder vendors the pinned historical V1 analysis engine and its two
helper modules so the original four-run report remains self-contained. The X1
folder regenerates the later full-V3 re-evaluation from the nested curated
runs. Its three `model_0.pt` files are intentionally omitted by the
terminal-checkpoint-only publication policy. Each run instead carries a
`provenance/legacy_initialization.json` record containing the SHA-256 observed
before archival. The comparison engine accepts that record only after the run
registry and resolved configuration match, labels the source as
checksum-only, and does not claim that the omitted bytes were reverified.

The archived studies and their fixed-grid inputs remain historical evidence;
the current parameter screen continues in
[Milestone 4](MILESTONE_4_GAIT_CLOSURE_PARAMETER_V4.md).
