# Leg-permutation fix and gait-family V2 milestone

- Date: 2026-08-24
- Branch: `72d-symm-v4-integration`
- Status: implementation and matched evaluation complete; Go2 reward
  optimization remains open.

## Scope

This milestone integrates the time-reversal-closed V2 gait family and a
reproducible leg-usage evaluation workflow. It replaces the earlier incidental
two-command playback comparison with a controlled fixed grid and keeps the
training learning curves, raw rollouts, and source/checkpoint provenance in the
curated archive.

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

## Evaluation protocol

Each terminal checkpoint is evaluated on all 10 training gaits at
`vx = ±{0.5, 1.0, 1.5} m/s`. Every cell is reset independently, settles for
5 seconds, and records 10 seconds for cycle-trimmed measurement. The primary
overall value averages rows within each family and then gives trot, bound,
half-bound, and gallop equal weight.

The archived comparison contains 420/420 valid planar-tracking-qualified cells:
180 for X1 and 240 for Go2. Heading success is reported separately because the
strict yaw threshold would otherwise select a sparse and gait-biased subset.

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

- [Gait-family V2 run log](RUN_LOG_GAIT_FAMILY_V2.md)
- [X1 report](dobot_x1_symm_flat/gait_famili_v2_analysis/REPORT.md)
- [Go2 report](unitree_go2_symm_flat/gait_famili_v2_analysis/REPORT.md)
- [X1 curated-run index](dobot_x1_symm_flat/README.md)
- [Go2 curated-run index](unitree_go2_symm_flat/README.md)

Every retained run includes its TensorBoard event, resolved parameters,
training-time Git provenance, terminal checkpoint, exports, playback artifacts,
and raw fixed-grid evaluation data. The analysis folders contain frozen study
manifests, reproducible scripts, CSV/JSON summaries, and SVG/PNG figures.
