<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# TRS 0.30/0.15 front/hind leg-usage comparison

Date: 2026-07-22

This report compares four updated 72D policies: no TRS and one TRS setting for
Unitree Go2 and Dobot X1. The TRS setting is the coefficient pair
`mirror_loss_coeff=0.30` and `value_loss_coeff=0.15`; the two numbers are not
separate ablation levels.

The short result is that the data **do not support** the proposed hypothesis.
For Go2, TRS made ground-reaction-force (GRF), absolute mechanical-work, and
normalized motor-heating allocation all less even between the front and hind
pairs. For X1, the result was mixed rather than similar: work became more even,
while GRF and normalized heating became substantially less even.

## Compared runs

| Robot | Condition | Mirror/value | Training run | Evaluation data |
| --- | --- | --- | --- | --- |
| Go2 | No TRS | `0 / 0` | [`2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20`](../unitree_go2_symm_flat/2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20/) | [Curated replay](unitree_go2_symm_flat/2026-07-19_10-32-57_go2_no_trs_pitch0p50_pterm1p20/plots/play/sim_data.npz) |
| Go2 | TRS | `0.30 / 0.15` | [`2026-07-21_22-36-39_go2_trs_m0p30_v0p15_w500`](../unitree_go2_symm_flat/2026-07-21_22-36-39_go2_trs_m0p30_v0p15_w500/) | [Routine-run replay](../unitree_go2_symm_flat/2026-07-21_22-36-39_go2_trs_m0p30_v0p15_w500/plots/play/sim_data.npz) |
| X1 | No TRS | `0 / 0` | [`2026-07-19_10-33-04_x1_no_trs_pitch0p35`](../dobot_x1_symm_flat/2026-07-19_10-33-04_x1_no_trs_pitch0p35/) | [Curated replay](dobot_x1_symm_flat/2026-07-19_10-33-04_x1_no_trs_pitch0p35/plots/play/sim_data.npz) |
| X1 | TRS | `0.30 / 0.15` | [`2026-07-21_22-37-01_x1_trs_m0p30_v0p15_w500`](../dobot_x1_symm_flat/2026-07-21_22-37-01_x1_trs_m0p30_v0p15_w500/) | [Routine-run replay](../dobot_x1_symm_flat/2026-07-21_22-37-01_x1_trs_m0p30_v0p15_w500/plots/play/sim_data.npz) |

All four policies used seed 42, 512 environments, 24 rollout steps per
environment, 20,000 training iterations, `TimeReversalPPO`, an adaptive
`1e-3` learning rate, 500 warm-up iterations, zero minimum TRS command speed,
and no data augmentation. No-TRS policies disabled mirror loss and set both
coefficients to zero. TRS policies enabled mirror loss at `0.30` and value loss
at `0.15`. The resolved configurations otherwise match for this comparison;
the environment snapshot's rename from `morphological_symmetry` to
`leg_permutation_symmetry` retains the same reward implementation through the
deprecated alias.

The curated Go2 no-TRS replay is used because it was regenerated after the
per-foot contact-filter recorder fix described in the
[60D-to-72D milestone](MILESTONE_60D_TO_72D.md#leg-usage-measurement). Its
non-contact arrays match the original replay. The X1 baseline replay is
byte-identical in the routine and curated locations.

## Measurement method

The quantitative comparison uses raw arrays over the common window
`0.5 <= t < 9.5 s`: 450 samples at 50 Hz under the constant
`-0.566558 m/s` sagittal command. None of the four policies reset in this
window. The full 30-second recordings are not a matched comparison because the
Go2 no-TRS policy terminates at about 10.26 seconds and its later command
sequence diverges.

For a front-pair total `F` and hind-pair total `H`, signed imbalance is

```text
100 * (F - H) / (F + H)
```

Positive values mean front-heavy use, negative values mean hind-heavy use, and
zero is perfectly even. Evenness is assessed by the absolute value: a positive
change in absolute imbalance is worse. An imbalance magnitude of 10% is a
55/45 front/hind split.

The measures are:

- **GRF load:** the Euclidean norm of each foot's world-frame GRF, including
  normal and friction components, summed within each pair and averaged over
  the window;
- **absolute mechanical power/work:** per-joint
  `abs(applied_torque * joint_velocity)`, summed within each pair; integrating
  over the common nine-second window gives absolute work;
- **normalized heating proxy:** the mean sum of
  `(applied_torque / effort_limit)^2`, which approximates relative motor-current
  or copper-heating exposure.

Because all conditions use the same duration, mean absolute power and absolute
work have exactly the same front/hind imbalance; they are two units for one
underlying allocation measure, not independent confirmations. The analysis
uses raw data rather than the one-second smoothed plotting arrays. The
[rollout recorder](../../../scripts/reinforcement_learning/rsl_rl/symm_rollout_plotter.py)
stores legs in front-left, front-right, rear-left, rear-right order.

## Results

Signed imbalance values are percentages. Brackets are 95% one-second block
bootstrap intervals describing temporal variation within a rollout only.

| Robot | Condition | Heating imbalance | Absolute power/work imbalance | GRF imbalance |
| --- | --- | ---: | ---: | ---: |
| Go2 | No TRS | +7.1 `[+1.6, +12.3]` | -3.8 `[-6.8, -0.8]` | +17.5 `[+13.5, +21.3]` |
| Go2 | TRS `0.30 / 0.15` | +30.8 `[+29.6, +32.3]` | -15.9 `[-21.0, -10.4]` | +25.8 `[+23.0, +28.4]` |
| X1 | No TRS | +17.2 `[+14.9, +19.6]` | +16.1 `[+14.9, +16.9]` | -2.5 `[-5.8, +1.0]` |
| X1 | TRS `0.30 / 0.15` | +52.9 `[+49.6, +56.2]` | +6.4 `[+4.1, +8.5]` | +12.8 `[+9.6, +16.5]` |

The hypothesis is tested directly by the change in imbalance magnitude:

| Robot | Measure | No-TRS magnitude | TRS magnitude | Change | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Go2 | Heating | 7.1% | 30.8% | **+23.8 pp** | Less even |
| Go2 | Absolute power/work | 3.8% | 15.9% | **+12.1 pp** | Less even |
| Go2 | GRF | 17.5% | 25.8% | **+8.4 pp** | Less even |
| X1 | Heating | 17.2% | 52.9% | **+35.7 pp** | Less even |
| X1 | Absolute power/work | 16.1% | 6.4% | **-9.7 pp** | More even |
| X1 | GRF | 2.5% | 12.8% | **+10.3 pp** | Less even |

Front/hind pair totals provide the underlying scale:

| Robot | Condition | Mean pair GRF, F / H [N] | Mean absolute power, F / H [W] | Absolute work, F / H [J] | Mean normalized heating, F / H |
| --- | --- | ---: | ---: | ---: | ---: |
| Go2 | No TRS | 96.52 / 67.83 | 51.79 / 55.93 | 466.14 / 503.34 | 0.291 / 0.252 |
| Go2 | TRS `0.30 / 0.15` | 103.54 / 61.04 | 58.65 / 80.90 | 527.89 / 728.07 | 0.268 / 0.142 |
| X1 | No TRS | 90.23 / 94.77 | 40.78 / 29.50 | 367.01 / 265.47 | 0.224 / 0.158 |
| X1 | TRS `0.30 / 0.15` | 107.86 / 83.39 | 43.16 / 37.99 | 388.41 / 341.89 | 0.378 / 0.116 |

TRS changed total exposure as well as allocation. Relative to no TRS, Go2's
total pair GRF was essentially unchanged (`+0.1%`), absolute work increased
`29.6%`, and normalized heating decreased `24.6%`; the heating reduction was
concentrated in the hind pair, so balance still worsened. For X1, total pair
GRF increased `3.4%`, absolute work increased `15.5%`, and normalized heating
increased `29.5%`.

The signed work components tell the same allocation story. Go2 positive-work
imbalance changed from `+1.1%` to `-30.1%`, while negative-work-magnitude
imbalance changed from `-9.2%` to `+15.6%`. X1 positive-work imbalance improved
from `+18.0%` to `+3.1%`, and negative-work-magnitude imbalance improved from
`+14.1%` to `+10.0%`.

## Interpretation

The Go2 observation is the opposite of the hypothesis: the TRS policy is less
balanced for every primary leg-usage measure in this matched window. GRF and
heating become more front-heavy, while absolute work becomes more hind-heavy.

The X1 cases are not broadly similar. TRS improves X1's work allocation, but
GRF changes from nearly even/slightly hind-heavy to front-heavy, and the
heating proxy becomes strongly front-heavy. Calling the policies equivalent
would require a predeclared equivalence margin and multiple independently
trained policies.

This result is consistent with the implemented objective: time reversal changes
velocity and odd phase quantities but does not permute the front and hind leg
indices. Its policy/value consistency losses therefore do not directly enforce
equal front/hind GRF, work, or heating.

## Evidence limits and next experiment

Each condition has one trained policy at seed 42 and one matched backward-speed
rollout. The 450 time samples are repeated observations of one policy, not 450
independent training replicates. The block-bootstrap intervals cannot establish
policy-population uncertainty or a causal TRS effect.

For a defensible causal test, train at least eight paired seeds per condition
with all non-TRS fields frozen. Evaluate every checkpoint on the same command
and gait manifest, use training seed as the statistical unit, and preselect
absolute front/hind imbalance as the primary endpoint. Test Go2 with a paired
directional contrast. Test the X1 similarity claim with two one-sided
equivalence tests and a preregistered practical-equivalence margin.
