# Milestone 4: Gait-closure parameter V4

- Evidence collected: 2026-08-25 through 2026-08-31
- Documented: 2026-09-01
- Development branch: `72d-symm-v4-integration`
- Publication branch: `jding/gait-closure-v4-archive`
- Baseline: [Milestone 3: gait-family V2](MILESTONE_3_GAIT_FAMILY_V2.md), completed 2026-08-24
- Status: single-seed development comparison complete; confirmatory multi-seed validation remains open

Timeline:
[Milestone 1](MILESTONE_1_60D_TO_72D.md) ·
[Milestone 2](MILESTONE_2_PHASE_MAPPING_V2_AND_LEG_PERMUTATION_FIX.md) ·
[Milestone 3](MILESTONE_3_GAIT_FAMILY_V2.md) · **Milestone 4**

## Scope

This milestone evaluates coefficient and schedule choices after the gait-family-v2 implementation. It keeps the
time-reversal-closed 10-gait library and the 10-gait-by-6-speed Cartesian evaluation design from Milestone 3,
re-evaluates and reprocesses the Milestone-3 checkpoints under the full-V3 metric and aggregation contract, adds
ramped TRS treatments for both robots, and reports training, velocity tracking, gait fidelity, and front/hind load
balance together.

The X1 study is a schedule-focused descriptive screen that preserves the declared Milestone-3 reward and action
settings. Source-provenance differences within the cohort prevent treating it as a strict single-factor schedule
experiment. The Go2 study is a recovery cohort: it strengthens and changes the joint-target feasibility treatment
as well as screening hard and ramped TRS schedules. Consequently, the Go2 Milestone-3-to-4 comparison is not a
single-coefficient ablation.

## What changed from Milestone 3

### Dobot X1

- Preserved the low and high TRS coefficients: policy/value `0.1/0.05` and `0.2/0.1`.
- Preserved a 500-iteration warmup for TRS runs and added a 1,000-iteration linear ramp at both coefficient levels.
  The matching hard-on treatments retain ramp `0`.
- Preserved foot-phase reward `0.3 * sum`, joint-target-limit weight `0.05` with `legacy_clamped` semantics,
  `legacy_global` actor-mean bounds, and `raw_action_mean` TR policy consistency.
- Preserved training seed 42. The disabled no-TRS run records warmup/ramp as `0/0`; these inactive schedule values do
  not affect its zero TRS coefficients.

Thus, X1 V4 is primarily a schedule/cohort expansion, not reward-coefficient tuning relative to the matching
Milestone-3 hard-on runs.

### Unitree Go2

- Preserved the low and high TRS coefficients `0.1/0.05` and `0.2/0.1`, with a 500-iteration warmup.
- Retained hard-on ramp `0` runs and replaced the earlier high-only 1,000-iteration ramp comparison with low and high
  500-iteration linear ramps. Both ramped runs set TR validity to command mode (`vmcmd`).
- Preserved the effective foot-phase reward at `0.3 * sum` and the existing Go2 foot-clearance profile: reward
  weight `0.15`, target height `0.08 m`, and minimum command speed `0.2 m/s`.
- Increased joint-target-limit weight from `0.05` to `0.2` and changed its meaning from `legacy_clamped` to
  `requested_overflow`.
- Changed actor-mean bounds from `legacy_global` to `per_joint_feasible`, with feasible margin fraction `0.0`.
- Changed TR policy consistency from `raw_action_mean` to `normalized_requested_joint_target`.
- Changed the training seed from 42 to 43.

The run suffix `fp0p3sum_jtlw0p2_amf0_g2fc1_s43` binds these actual choices. An older prospective registry and the
seven-day command matrix describe a planned Go2 profile with `0.4 * mean`, joint-target-limit weight `0.05`, and
seed 42. That planned profile is not the cohort reported here; the archived `provenance/initialization.json` files
are authoritative for this milestone.

## What stayed unchanged

- The 72-dimensional observation, 12-dimensional action, and time-reversal-closed V2 gait definitions from
  Milestone 3.
- Equal-family gait sampling with no gait curriculum.
- Training budget: 20,000 iterations, 512 environments, 24 steps per environment, or 245,760,000 environment
  transitions per run.
- Evaluation design: 10 gait rows crossed with `vx = +/-{0.5, 1.0, 1.5} m/s`, giving 60 cells per run. The full-V3
  implementation applies this inherited Cartesian design at fixed evaluation seed 4242.
- Evaluation aggregation: average cells within trot, bound, half-bound, and gallop, then weight those four family
  means equally. No outcome or tracking-success filter is applied.
- TR minibatch augmentation remains disabled for this cohort; the listed policy/value terms are regularizers.

## Archived five-run cohorts

### Dobot X1, seed 42

| Label | Archived run | Policy/value | Warmup/ramp |
|---|---|---:|---:|
| NoTRS | `2026-08-25_06-08-02_notrs_x1def_s42` | `0/0` | `0/0` |
| Low-r0 | `2026-08-25_16-56-41_trs_m0p1_v0p05_w500_r0_x1def_s42` | `0.1/0.05` | `500/0` |
| Low-r1000 | `2026-08-26_05-12-37_trs_m0p1_v0p05_w500_r1000_x1def_s42` | `0.1/0.05` | `500/1000` |
| High-r0 | `2026-08-27_00-44-04_trs_m0p2_v0p1_w500_r0_x1def_s42` | `0.2/0.1` | `500/0` |
| High-r1000 | `2026-08-27_23-34-38_trs_m0p2_v0p1_w500_r1000_x1def_s42` | `0.2/0.1` | `500/1000` |

### Unitree Go2, seed 43

| Label | Archived run | Policy/value | Warmup/ramp |
|---|---|---:|---:|
| NoTRS | `2026-08-29_11-56-35_notrs_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0/0` | inactive |
| Low-r0 | `2026-08-30_09-45-19_trs_m0p1_v0p05_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.1/0.05` | `500/0` |
| Low-r500 | `2026-08-31_00-14-06_trs_m0p1_v0p05_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.1/0.05` | `500/500` |
| High-r0 | `2026-08-30_09-45-34_trs_m0p2_v0p1_w500_r0_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.2/0.1` | `500/0` |
| High-r500 | `2026-08-31_00-14-42_trs_m0p2_v0p1_w500_r500_vmcmd_fp0p3sum_jtlw0p2_amf0_g2fc1_s43` | `0.2/0.1` | `500/500` |

## Results

`AUC` is the mean-reward learning-curve area over all 20,000 iterations and `Tail` is mean reward over the final
1,000 iterations. Evaluation values are the family-balanced all-velocity results from the 60-cell full-V3 grid.
`vx` and `yaw` are RMSE in `m/s` and `rad/s`; `Gait` is boundary-excluded contact-pattern agreement. The four load
columns are mean absolute front/hind imbalance percentages. Lower is better except for AUC, Tail, and Gait.

Milestone-3 evaluation baselines quoted below use the same archived checkpoints reprocessed with full-V3. Those
evaluation values can differ slightly from the original Milestone-3 tables because the metric and aggregation
version changed; the checkpoints did not.

### Dobot X1

| Run | AUC | Tail | vx | yaw | Gait | Torque^2 | Norm. torque^2 | Work | GRF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoTRS | 34.667 | 42.307 | 0.03136 | 0.03559 | 96.747% | 17.687% | 26.149% | 23.917% | 6.999% |
| Low-r0 | **36.012** | **43.182** | 0.03377 | 0.03571 | 95.586% | **6.053%** | 13.165% | 15.915% | **2.950%** |
| Low-r1000 | 35.614 | 42.436 | 0.03408 | 0.02957 | **95.762%** | 7.297% | 17.505% | **11.989%** | 3.861% |
| High-r0 | 33.281 | 42.392 | **0.03191** | **0.02852** | 95.298% | 6.749% | 12.907% | 17.475% | 3.494% |
| High-r1000 | 34.921 | 42.095 | 0.03252 | 0.03561 | 95.235% | 8.308% | **8.666%** | 18.145% | 3.340% |

Bold evaluation values identify the best TRS treatment for that column, not a statistically confirmed winner. The
low hard-on treatment has the strongest observed learning and lowest torque-squared and GRF imbalance. Ramping the
low treatment improves yaw tracking and work balance, while the high ramp gives the lowest normalized-torque
imbalance. NoTRS retains the best gait agreement and slightly lower forward-velocity RMSE than every TRS run.

Against the matching Milestone-3 runs, Low-r0 tail reward changes from 43.040 to 43.182 and High-r0 from 41.081 to
42.392. High-r0 also improves yaw RMSE from 0.03838 to 0.02852, torque-squared imbalance from 9.549% to 6.749%,
normalized-torque imbalance from 14.945% to 12.907%, work imbalance from 18.952% to 17.475%, and GRF imbalance from
4.271% to 3.494%. Its forward-velocity RMSE worsens from 0.02891 to 0.03191 and gait agreement changes from 95.421%
to 95.298%. The Low-r0 comparison is likewise mixed: torque-squared and GRF balance improve, but forward/yaw
tracking and work balance worsen.

### Unitree Go2

| Run | AUC | Tail | vx | yaw | Gait | Torque^2 | Norm. torque^2 | Work | GRF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoTRS | 32.534 | 40.528 | 0.03439 | 0.05904 | **88.803%** | 26.651% | 28.274% | 13.557% | 19.102% |
| Low-r0 | **33.051** | **41.189** | **0.03326** | 0.06151 | 86.572% | 22.684% | 26.420% | 12.635% | 25.019% |
| Low-r500 | 31.561 | 38.865 | 0.04474 | 0.06232 | 87.438% | 24.321% | 27.356% | 12.956% | 22.049% |
| High-r0 | 31.998 | 39.550 | 0.03581 | 0.06336 | 86.627% | **18.379%** | **21.668%** | **9.278%** | **17.715%** |
| High-r500 | 32.868 | 40.941 | 0.03403 | **0.05530** | 87.824% | 21.340% | 25.810% | 10.992% | 18.875% |

Low-r0 has the highest observed learning AUC and tail reward and the best TRS forward-velocity RMSE. High-r0 has
the best load balance on all four measures but pays a reward and tracking cost. The high ramp recovers reward above
NoTRS, gives the best yaw RMSE, and has the best gait agreement among TRS runs. NoTRS still has the best gait
agreement overall.

Relative to Milestone 3, the matching Go2 Low-r0 run raises tail reward from 37.919 to 41.189 and full AUC from
31.281 to 33.051; forward/yaw RMSE improve from 0.04117/0.08789 to 0.03326/0.06151 and gait agreement from 81.353%
to 86.572%. Work and GRF imbalance improve from 22.230%/26.093% to 12.635%/25.019%, while torque-squared and
normalized-torque imbalance worsen from 13.985%/15.501% to 22.684%/26.420%.

The matching High-r0 run raises tail reward from 35.678 to 39.550 and AUC from 27.502 to 31.998; forward/yaw RMSE
improve from 0.03748/0.08954 to 0.03581/0.06336 and gait agreement from 81.295% to 86.627%. Torque-squared and work
imbalance improve from 19.607%/9.621% to 18.379%/9.278%, while normalized-torque and GRF imbalance worsen from
20.645%/17.241% to 21.668%/17.715%. The old r1000 and new r500-command-validity runs are not schedule-matched.

These results support a narrow conclusion: the archived Go2 recovery profile improves learning, velocity tracking,
and gait closure on several important comparisons, while physical load-balance outcomes remain treatment- and
metric-dependent. They do not support a blanket performance or causal claim.

## Evidence and reproduction

The publication reports, resolved manifests, source hashes, tables, and figure hashes are in:

- [Dobot X1 V4 report](dobot_x1_symm_flat/gait_closure_parameter_v4_analysis/REPORT.md)
- [Unitree Go2 V4 report](unitree_go2_symm_flat/gait_closure_parameter_v4_analysis/REPORT.md)
- `dobot_x1_symm_flat/gait_closure_parameter_v4_analysis/`
- `unitree_go2_symm_flat/gait_closure_parameter_v4_analysis/`

From the repository root, regenerate every comparison table and figure with the archived manifest:

```bash
./isaaclab.sh -p logs/rsl_rl/good_runs_72d/dobot_x1_symm_flat/gait_closure_parameter_v4_analysis/reproduce.py
./isaaclab.sh -p logs/rsl_rl/good_runs_72d/unitree_go2_symm_flat/gait_closure_parameter_v4_analysis/reproduce.py
```

On Windows PowerShell, use the equivalent wrapper:

```powershell
.\isaaclab.bat -p logs\rsl_rl\good_runs_72d\dobot_x1_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
.\isaaclab.bat -p logs\rsl_rl\good_runs_72d\unitree_go2_symm_flat\gait_closure_parameter_v4_analysis\reproduce.py
```

Each retained training directory supplies its TensorBoard event file, `provenance/initialization.json`, resolved
command and source snapshot, terminal checkpoint, and complete `evaluations/leg_usage_grid_full_v3` metrics. The
comparison reads the archived metric tables; it does not require raw NPZ recordings.

The curated checkpoint policy keeps only the latest checkpoint for each completed training run:
`model_19999.pt`. Intermediate `model_*.pt` files are intentionally excluded. Do not remove the terminal checkpoint,
event file, provenance, resolved configuration, or full-V3 metric inputs, because together they bind the reported
learning curve and evaluation to the archived policy.

## Limitations

- Each robot has one development training seed. X1 uses seed 42 and Go2 uses seed 43; the robots are not replicates
  and must not be pooled.
- Go2 changes seed, target-limit coefficient and semantics, actor bounds, policy output space, and ramp validity.
  Milestone-3-to-4 deltas therefore cannot identify one causal mechanism.
- The X1 and Go2 V4 profiles intentionally differ, so cross-robot differences are descriptive.
- The reports note source-provenance differences within the cohorts. Even within a robot, baseline deltas are not a
  strict single-factor experiment.
- The full-V3 grid uses one fixed evaluation seed and simulation-only rollouts. It provides broad gait/velocity
  coverage, not confidence intervals, hardware validation, or robustness to untested environments.
- Metric optima disagree. Reward, tracking, gait fidelity, effort balance, work balance, and GRF balance must remain
  separate endpoints until a prospective, matched, multi-seed study defines and validates a primary trade-off.
