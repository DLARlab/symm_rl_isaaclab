# TRS 3x3 grid: leg allocation, durability exposure, and training efficiency

Date: 2026-07-27

This report compares the full coefficient/warm-up grid against the matched no-TRS
policy for Unitree Go2 and Dobot X1. All comparisons use seed 42, 512
environments, 24 rollout steps per iteration, and 20,000 PPO iterations.

## Executive conclusion

| Hypothesis | Grid evidence | Verdict for this seed and rollout |
| --- | --- | --- |
| TRS makes Go2 front/hind use more even | All three primary measures improved together in 0/9 cells; raw torque-squared, work, and vertical-GRF balance each improved in 0/9. The cyclic-torque proxy was more even in 6/9 (m=3) and 8/9 (m=5). | **Not supported across load mechanisms; contradicted by the three primary allocation measures** |
| TRS changes X1 front/hind distribution only a little | All three primary changes stayed within an illustrative +/-5 pp margin in 1/9 cells. | **Not supported as a general equivalence claim** |
| TRS improves Go2 training efficiency | Reward AUC improved in 1/9 cells; 7/9 TRS cells never achieved the sustained reward-35 criterion reached by no TRS. | **Not robust; supported only by the 0.30/0.15, w=500 cell** |
| TRS improves X1 training efficiency | Reward AUC improved in 0/9 cells and reward 35 took more transitions in every cell. | **Contradicted** |
| TRS reduces durability-oriented exposure | No cell reduced all six separately reported worst-component proxies: Go2 0/9, X1 0/9. This is a mechanism-by-mechanism comparison, not a composite score. | **Not supported as a general claim** |

There is no grid cell that supports all original hypotheses. The component-level
audit also shows why pair evenness cannot be interpreted as lower break risk:
total exposure or one joint/foot can rise while the pair split becomes more even.

These are descriptive grid results, not population-level causal estimates: every
cell contains one policy trained with the same seed. Grid cells are hyperparameter
settings, not independent replicates.

## Metric audit and corrected interpretation

The earlier pair calculation is numerically correct, but pair aggregation alone is
not a durability ranking. A pair can be perfectly balanced while both pairs are
heavily loaded, or one actuator can be overloaded while the two pair sums match.
The revised analysis therefore keeps three orthogonal questions separate:

1. **Allocation:** front/hind imbalance and dominant-pair ratio.
2. **Amount:** total exposure per second and per commanded-direction metre.
3. **Localization:** worst individual joint/foot, peaks, capacity dwell, and cyclic
   load spectrum.

- `sum((torque / effort_limit)^2)` measures **normalized torque-capacity
  utilization**, not motor current or copper heating. A true heating metric needs
  each motor's torque constant, winding resistance, transmission efficiency, and
  thermal model. The configured effort limits are simulator torque clamps, not
  continuous thermal ratings or fatigue strengths.
- Raw **torque-squared exposure**, RMS utilization, p95/p99/peak utilization, and
  time at or above 50%, 75%, 90%, and 99% of the configured limit are computed
  independently for all 12 joints before aggregation.
- Torque histories are cycle-counted per joint with an ASTM E1049-style rainflow
  stack. The reported m=3 and m=5 quantities are
  `sum(count * (cycle_amplitude / effort_limit)^m)` per metre. They are low/high
  peak-sensitivity descriptors, **not** Miner damage, lifetime, or failure
  probability without component S-N curves and a stress/mean-load model.
- Euclidean GRF is retained, but **vertical GRF impulse** is the cleaner support-load
  metric. Per-foot p99/peak vertical force and positive finite-difference loading
  rate expose localized impacts, while their 50 Hz sampling is acknowledged.
- Absolute mechanical work is retained and split into positive and negative work.
  Work measures drivetrain energy throughput, not fatigue, and static torque can be
  damaging even when work is zero.
- Mission-normalized quantities divide by endpoint displacement projected onto the
  mean commanded direction. This avoids making lateral wandering look efficient;
  cumulative path length and lateral drift are retained for validation.
- Pair imbalance remains `100 * (front - hind) / (front + hind)`. Its magnitude is
  exactly twice the deviation of the front share from 50%, so it is already the
  simplest well-scaled two-pair evenness metric. Dominance ratio
  `max(front, hind) / min(front, hind)` gives the same allocation in ratio form.
- Heterogeneous endpoints are not averaged into an arbitrary damage score.

Cycle counting follows the scope of [ASTM E1049](https://store.astm.org/standards/e1049).
The exposure-preserving moment-per-distance treatment is analogous to the load
spectrum workflow used in the [NREL mechanical-loads report]
(https://www.nrel.gov/docs/fy15osti/63679.pdf), but no material calibration is
claimed here.

## Front/hind balance

Every cell below is the change in absolute imbalance relative to no TRS
(percentage points). Negative is more even; positive is less even.

| Robot | Allocation proxy | No-TRS dominant pair (ratio) | TRS front / hind dominant cells | TRS dominance-ratio range |
| --- | --- | --- | ---: | ---: |
| Go2 | Torque-squared exposure | front (1.019x) | 8 / 1 | 1.229x to 2.301x |
| Go2 | Absolute work | hind (1.080x) | 3 / 6 | 1.136x to 3.169x |
| Go2 | Vertical GRF impulse | front (1.428x) | 9 / 0 | 1.453x to 2.309x |
| Go2 | Cyclic-torque severity m=3 | front (2.326x) | 8 / 1 | 1.261x to 4.845x |
| Go2 | Cyclic-torque severity m=5 | front (5.599x) | 8 / 1 | 1.391x to 17.392x |
| X1 | Torque-squared exposure | front (1.057x) | 7 / 2 | 1.018x to 1.673x |
| X1 | Absolute work | front (1.382x) | 8 / 1 | 1.009x to 1.366x |
| X1 | Vertical GRF impulse | hind (1.052x) | 9 / 0 | 1.038x to 1.296x |
| X1 | Cyclic-torque severity m=3 | front (3.267x) | 9 / 0 | 2.834x to 10.239x |
| X1 | Cyclic-torque severity m=5 | front (8.971x) | 9 / 0 | 8.580x to 78.961x |

### Go2

**Raw torque-squared exposure: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +12.5 pp | +19.7 pp | +11.4 pp |
| 0.20 / 0.10 | +25.8 pp | +21.0 pp | +38.5 pp |
| 0.30 / 0.15 | +9.3 pp | +37.9 pp | +26.1 pp |

**Absolute mechanical work: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +43.8 pp | +3.7 pp | +43.2 pp |
| 0.20 / 0.10 | +4.8 pp | +2.5 pp | +9.5 pp |
| 0.30 / 0.15 | +48.2 pp | +14.8 pp | +12.1 pp |

**Vertical GRF impulse: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +7.5 pp | +13.9 pp | +7.6 pp |
| 0.20 / 0.10 | +13.5 pp | +10.6 pp | +10.3 pp |
| 0.30 / 0.15 | +0.8 pp | +21.9 pp | +8.7 pp |

**Normalized torque-capacity utilization: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +11.5 pp | +21.5 pp | +7.3 pp |
| 0.20 / 0.10 | +28.8 pp | +23.6 pp | +35.7 pp |
| 0.30 / 0.15 | -2.9 pp | +40.2 pp | +23.8 pp |

**Cyclic-torque severity m=3: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -18.1 pp | -21.1 pp | -6.3 pp |
| 0.20 / 0.10 | +5.0 pp | +1.6 pp | +25.9 pp |
| 0.30 / 0.15 | -28.3 pp | -3.9 pp | -19.5 pp |

**Cyclic-torque severity m=5: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -45.5 pp | -29.4 pp | -8.8 pp |
| 0.20 / 0.10 | -2.9 pp | -1.8 pp | +19.4 pp |
| 0.30 / 0.15 | -53.3 pp | -5.8 pp | -41.6 pp |

### X1

**Raw torque-squared exposure: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +9.9 pp | +22.4 pp | +6.7 pp |
| 0.20 / 0.10 | +4.1 pp | -1.9 pp | +10.1 pp |
| 0.30 / 0.15 | +1.3 pp | +2.0 pp | +20.5 pp |

**Absolute mechanical work: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -3.9 pp | -0.8 pp | -7.6 pp |
| 0.20 / 0.10 | -15.6 pp | -13.5 pp | -0.6 pp |
| 0.30 / 0.15 | -3.6 pp | -10.0 pp | -9.7 pp |

**Vertical GRF impulse: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +3.9 pp | +9.8 pp | +9.2 pp |
| 0.20 / 0.10 | +6.5 pp | -0.7 pp | +8.4 pp |
| 0.30 / 0.15 | +3.5 pp | +2.8 pp | +10.4 pp |

**Normalized torque-capacity utilization: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +17.1 pp | +35.7 pp | +17.3 pp |
| 0.20 / 0.10 | +10.0 pp | -2.7 pp | +14.0 pp |
| 0.30 / 0.15 | +0.7 pp | +0.6 pp | +35.7 pp |

**Cyclic-torque severity m=3: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +10.3 pp | +29.1 pp | +5.2 pp |
| 0.20 / 0.10 | -1.9 pp | -1.1 pp | +13.3 pp |
| 0.30 / 0.15 | -0.8 pp | -5.3 pp | +20.7 pp |

**Cyclic-torque severity m=5: change in absolute imbalance [pp]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +10.2 pp | +17.6 pp | +9.1 pp |
| 0.20 / 0.10 | -0.4 pp | +2.8 pp | +9.9 pp |
| 0.30 / 0.15 | +0.4 pp | -0.8 pp | +14.7 pp |

## Total load and cost per distance

Balance and total cost are separate outcomes. The tables below normalize each
rollout's exposure by progress projected onto the commanded planar direction and
show percent change from no TRS. Negative is lower total exposure or cost.

### Go2

Torque-squared exposure per metre fell in 6/9 cells; absolute work per metre fell in 2/9.

**Torque-squared exposure per metre: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -4.7% | -2.2% | +8.3% |
| 0.20 / 0.10 | -13.6% | -4.8% | +5.2% |
| 0.30 / 0.15 | +10.1% | -7.6% | -22.8% |

**Absolute mechanical work per metre: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +80.5% | -0.5% | +91.0% |
| 0.20 / 0.10 | +1.6% | +19.3% | +2.2% |
| 0.30 / 0.15 | +249.0% | -3.3% | +32.3% |

### X1

Torque-squared exposure per metre fell in 0/9 cells; absolute work per metre fell in 2/9.

**Torque-squared exposure per metre: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +23.4% | +1.5% | +4.5% |
| 0.20 / 0.10 | +5.5% | +10.5% | +29.4% |
| 0.30 / 0.15 | +11.2% | +6.6% | +15.6% |

**Absolute mechanical work per metre: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +27.3% | -6.5% | -5.0% |
| 0.20 / 0.10 | +3.9% | +21.3% | +16.9% |
| 0.30 / 0.15 | +4.0% | +16.4% | +21.4% |

## Durability-oriented component results

Negative percentages below mean lower observed exposure than no TRS. A lower value
is evidence only for that mechanism; agreement across unrelated mechanisms is
reported rather than forced through a composite score.

| Robot | Pair-additive proxy | Lower total / directed m | Lower worst pair / directed m | More even pair split |
| --- | --- | ---: | ---: | ---: |
| Go2 | Torque-squared exposure | 6/9 | 1/9 | 0/9 |
| Go2 | Cyclic-torque severity m=3 | 6/9 | 6/9 | 6/9 |
| Go2 | Cyclic-torque severity m=5 | 6/9 | 6/9 | 8/9 |
| Go2 | Absolute work | 2/9 | 0/9 | 0/9 |
| Go2 | Vertical GRF impulse | 3/9 | 0/9 | 0/9 |
| X1 | Torque-squared exposure | 0/9 | 0/9 | 1/9 |
| X1 | Cyclic-torque severity m=3 | 1/9 | 1/9 | 4/9 |
| X1 | Cyclic-torque severity m=5 | 2/9 | 2/9 | 2/9 |
| X1 | Absolute work | 2/9 | 3/9 | 9/9 |
| X1 | Vertical GRF impulse | 2/9 | 1/9 | 1/9 |

| Robot | Worst-component proxy | Lower worst component | Median change in worst component |
| --- | --- | ---: | ---: |
| Go2 | Joint RMS torque-limit utilization | 1/9 | +10.2% |
| Go2 | Joint p99 torque-limit utilization | 3/9 | +0.9% |
| Go2 | Joint cyclic-torque severity m=3 / m | 6/9 | -15.3% |
| Go2 | Joint cyclic-torque severity m=5 / m | 6/9 | -21.3% |
| Go2 | Joint absolute work / m | 0/9 | +77.0% |
| Go2 | Foot p99 vertical force | 3/9 | +2.6% |
| Go2 | Foot p99 vertical loading rate | 0/9 | +10.0% |
| X1 | Joint RMS torque-limit utilization | 2/9 | +11.4% |
| X1 | Joint p99 torque-limit utilization | 1/9 | +28.3% |
| X1 | Joint cyclic-torque severity m=3 / m | 2/9 | +11.4% |
| X1 | Joint cyclic-torque severity m=5 / m | 2/9 | +17.3% |
| X1 | Joint absolute work / m | 3/9 | +4.1% |
| X1 | Foot p99 vertical force | 0/9 | +77.8% |
| X1 | Foot p99 vertical loading rate | 0/9 | +96.8% |

The pair and component conclusions can differ. For example, X1 absolute-work
allocation became more even in every cell, yet the worst-pair work fell in only
three cells and the worst-joint work fell in three. Go2 cyclic-torque severity
improved in six cells for both m=3 and m=5, but worst-joint RMS utilization fell
in only one cell, worst-joint work in none, and worst-foot p99 loading rate in
none.

### Go2 durability grids

**Cyclic-torque severity m=3, total per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -15.0% | -14.5% | +35.8% |
| 0.20 / 0.10 | -10.3% | -2.4% | +20.3% |
| 0.30 / 0.15 | +126.8% | -14.6% | -25.6% |

**Cyclic-torque severity m=5, total per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -25.1% | -34.1% | +73.5% |
| 0.20 / 0.10 | -11.2% | -7.5% | +62.6% |
| 0.30 / 0.15 | +261.4% | -20.7% | -45.5% |

**Worst-joint cyclic-torque severity m=5 per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -60.3% | -47.2% | +91.6% |
| 0.20 / 0.10 | -21.3% | -21.6% | +104.4% |
| 0.30 / 0.15 | +97.9% | -11.9% | -72.5% |

**Worst-joint RMS torque-limit utilization: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +10.2% | +9.0% | +13.8% |
| 0.20 / 0.10 | +9.4% | +12.0% | +19.6% |
| 0.30 / 0.15 | -9.4% | +19.9% | +0.5% |

**Maximum any-joint dwell at >=99% of configured effort limit [% of window]; no-TRS=0.0%**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | 0.0% | 0.0% | 0.0% |
| 0.20 / 0.10 | 0.0% | 0.0% | 0.0% |
| 0.30 / 0.15 | 0.0% | 0.0% | 0.0% |

**Worst-joint absolute work per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +299.3% | +7.2% | +310.0% |
| 0.20 / 0.10 | +35.0% | +77.0% | +27.4% |
| 0.30 / 0.15 | +507.9% | +1.5% | +104.9% |

**Worst-foot p99 vertical force: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -6.6% | +65.2% | +5.7% |
| 0.20 / 0.10 | +3.2% | +1.8% | +2.6% |
| 0.30 / 0.15 | +9.7% | -1.7% | -7.7% |

**Worst-foot p99 positive vertical loading rate: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +4.9% | +89.7% | +14.8% |
| 0.20 / 0.10 | +16.1% | +6.4% | +10.0% |
| 0.30 / 0.15 | +26.0% | +2.8% | +0.0% |

### X1 durability grids

**Cyclic-torque severity m=3, total per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +46.5% | +57.4% | +20.4% |
| 0.20 / 0.10 | +16.7% | +17.8% | +27.1% |
| 0.30 / 0.15 | +5.9% | -27.3% | +38.2% |

**Cyclic-torque severity m=5, total per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +97.8% | +156.5% | +44.7% |
| 0.20 / 0.10 | +19.3% | +15.7% | +25.4% |
| 0.30 / 0.15 | -6.9% | -57.2% | +74.9% |

**Worst-joint cyclic-torque severity m=5 per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +116.3% | +157.6% | +54.2% |
| 0.20 / 0.10 | +17.3% | +11.9% | +9.4% |
| 0.30 / 0.15 | -31.1% | -64.5% | +79.7% |

**Worst-joint RMS torque-limit utilization: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +27.9% | +32.4% | +11.4% |
| 0.20 / 0.10 | +18.6% | -7.7% | +10.6% |
| 0.30 / 0.15 | +6.7% | -21.2% | +34.5% |

**Maximum any-joint dwell at >=99% of configured effort limit [% of window]; no-TRS=0.0%**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | 8.2% | 5.3% | 1.1% |
| 0.20 / 0.10 | 0.0% | 0.0% | 0.0% |
| 0.30 / 0.15 | 0.0% | 0.0% | 9.3% |

**Worst-joint absolute work per directed metre: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +27.9% | -23.7% | -16.6% |
| 0.20 / 0.10 | +1.4% | +4.1% | +16.5% |
| 0.30 / 0.15 | -8.4% | +8.7% | +26.7% |

**Worst-foot p99 vertical force: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +149.2% | +158.5% | +152.0% |
| 0.20 / 0.10 | +34.5% | +77.8% | +33.7% |
| 0.30 / 0.15 | +81.3% | +20.5% | +44.5% |

**Worst-foot p99 positive vertical loading rate: change [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +175.6% | +184.1% | +176.9% |
| 0.20 / 0.10 | +47.8% | +96.8% | +46.9% |
| 0.30 / 0.15 | +100.4% | +32.5% | +59.1% |

### Worst-component localization by grid cell

The component name is as important as the pair sum. Parentheses show the absolute
RMS utilization or the change in the current global maximum versus the no-TRS
global maximum.

| Robot | Mirror/value, warm-up | Worst RMS-utilization joint | Worst m=5 cyclic-severity joint | Worst p99-impact foot |
| --- | --- | --- | --- | --- |
| Go2 | 0.10/0.05, w=10 | `FR_thigh_joint` (0.348) | `RL_thigh_joint` (-60.3%) | `front_left` (-6.6%) |
| Go2 | 0.10/0.05, w=100 | `FR_thigh_joint` (0.344) | `FR_thigh_joint` (-47.2%) | `front_right` (+65.2%) |
| Go2 | 0.10/0.05, w=500 | `FL_calf_joint` (0.360) | `FR_thigh_joint` (+91.6%) | `front_right` (+5.7%) |
| Go2 | 0.20/0.10, w=10 | `FR_thigh_joint` (0.346) | `FR_thigh_joint` (-21.3%) | `front_left` (+3.2%) |
| Go2 | 0.20/0.10, w=100 | `FR_thigh_joint` (0.354) | `FR_thigh_joint` (-21.6%) | `front_right` (+1.8%) |
| Go2 | 0.20/0.10, w=500 | `FL_calf_joint` (0.378) | `FR_thigh_joint` (+104.4%) | `front_left` (+2.6%) |
| Go2 | 0.30/0.15, w=10 | `FR_thigh_joint` (0.286) | `RR_thigh_joint` (+97.9%) | `rear_left` (+9.7%) |
| Go2 | 0.30/0.15, w=100 | `FR_thigh_joint` (0.379) | `FR_thigh_joint` (-11.9%) | `front_right` (-1.7%) |
| Go2 | 0.30/0.15, w=500 | `FR_thigh_joint` (0.318) | `FR_thigh_joint` (-72.5%) | `front_right` (-7.7%) |
| X1 | 0.10/0.05, w=10 | `joint_front_left_thigh_pitch` (0.429) | `joint_front_left_thigh_pitch` (+116.3%) | `front_left` (+149.2%) |
| X1 | 0.10/0.05, w=100 | `joint_front_left_thigh_pitch` (0.444) | `joint_front_left_thigh_pitch` (+157.6%) | `front_left` (+158.5%) |
| X1 | 0.10/0.05, w=500 | `joint_front_left_thigh_pitch` (0.374) | `joint_front_left_thigh_pitch` (+54.2%) | `front_left` (+152.0%) |
| X1 | 0.20/0.10, w=10 | `joint_front_left_thigh_pitch` (0.398) | `joint_front_left_thigh_pitch` (+17.3%) | `front_left` (+34.5%) |
| X1 | 0.20/0.10, w=100 | `joint_front_left_thigh_pitch` (0.310) | `joint_front_left_thigh_pitch` (+11.9%) | `front_left` (+77.8%) |
| X1 | 0.20/0.10, w=500 | `joint_front_left_thigh_pitch` (0.371) | `joint_front_left_thigh_pitch` (+9.4%) | `front_left` (+33.7%) |
| X1 | 0.30/0.15, w=10 | `joint_front_left_thigh_pitch` (0.358) | `joint_front_left_thigh_pitch` (-31.1%) | `front_left` (+81.3%) |
| X1 | 0.30/0.15, w=100 | `joint_front_left_thigh_pitch` (0.264) | `joint_front_left_thigh_pitch` (-64.5%) | `front_left` (+20.5%) |
| X1 | 0.30/0.15, w=500 | `joint_front_left_thigh_pitch` (0.451) | `joint_front_left_thigh_pitch` (+79.7%) | `rear_left` (+44.5%) |

## Training efficiency

Reward AUC is the mean return over the fixed 245.76-million-transition
budget; it is the primary threshold-free sample-efficiency measure. The
reward-35 metric uses a trailing 200-iteration mean and requires the next 500
iterations to remain above threshold. PPO learning time isolates the optimization
phase, but all wall-clock and throughput results remain secondary because machine
load and paired Go2/X1 GPU contention differed between scan dates.

The TensorBoard exports show the same 200-iteration trailing-mean reward
curves against environment transitions (left, primary evidence) and observed
wall time (right, secondary evidence). Color identifies the mirror/value pair;
line style identifies the warm-up. The black curve is the matched no-TRS
baseline.

### Go2

![Go2 TensorBoard training-efficiency curves](tensorboard_reward_efficiency_go2.svg)

Final reward improved in 1/9 cells. The sustained reward-35 target was reached in 2/9 cells and missed in 7/9. The PPO learning phase was faster in 0/9 cells.

**Reward AUC change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -3.0% | -10.3% | -4.1% |
| 0.20 / 0.10 | -8.2% | -8.8% | -13.2% |
| 0.30 / 0.15 | -20.4% | -14.2% | +1.2% |

**Transitions to sustained reward 35: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -6.3% | n/a | n/a |
| 0.20 / 0.10 | n/a | n/a | n/a |
| 0.30 / 0.15 | n/a | n/a | -21.7% |

**PPO learning time per iteration: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +55.4% | +48.4% | +46.2% |
| 0.20 / 0.10 | +74.0% | +83.5% | +65.7% |
| 0.30 / 0.15 | +96.2% | +48.1% | +80.2% |

### X1

![X1 TensorBoard training-efficiency curves](tensorboard_reward_efficiency_x1.svg)

Final reward improved in 0/9 cells. The sustained reward-35 target was reached in 9/9 cells and missed in 0/9. The PPO learning phase was faster in 1/9 cells.

**Reward AUC change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | -0.6% | -5.0% | -1.1% |
| 0.20 / 0.10 | -9.0% | -1.5% | -6.0% |
| 0.30 / 0.15 | -12.3% | -16.0% | -9.5% |

**Transitions to sustained reward 35: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +11.5% | +51.4% | +21.2% |
| 0.20 / 0.10 | +63.4% | +5.9% | +47.4% |
| 0.30 / 0.15 | +61.5% | +92.4% | +62.3% |

**PPO learning time per iteration: change vs no TRS [%]**

| Mirror / value | Warm-up 10 | Warm-up 100 | Warm-up 500 |
| --- | ---: | ---: | ---: |
| 0.10 / 0.05 | +16.0% | +16.4% | -4.1% |
| 0.20 / 0.10 | +45.2% | +19.1% | +48.0% |
| 0.30 / 0.15 | +11.3% | +17.8% | +39.9% |

## Baseline scales and validation

| Robot | Metric | No-TRS signed imbalance | No-TRS front share |
| --- | --- | ---: | ---: |
| Go2 | Raw torque-squared exposure | +0.9% | 50.5% |
| Go2 | Absolute mechanical work | -3.8% | 48.1% |
| Go2 | Vertical GRF impulse | +17.6% | 58.8% |
| Go2 | Normalized torque-capacity utilization | +7.1% | 53.5% |
| X1 | Raw torque-squared exposure | +2.8% | 51.4% |
| X1 | Absolute mechanical work | +16.1% | 58.0% |
| X1 | Vertical GRF impulse | -2.5% | 48.7% |
| X1 | Normalized torque-capacity utilization | +17.2% | 58.6% |

| Robot | Directed progress [m] | Path length [m] | Lateral drift [m] | Tracking RMSE [m/s] | Absolute work [J/m] | Torque-squared exposure [N^2 m^2 s/m] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Go2 | 5.057 | 5.066 | 0.001 | 0.0433 | 191.70 | 893.57 |
| X1 | 5.035 | 5.036 | 0.034 | 0.0330 | 125.62 | 395.75 |

| Robot | No-TRS worst component metric | Component | Value |
| --- | --- | --- | ---: |
| Go2 | RMS torque-limit utilization | `RL_calf_joint` | 0.3161 |
| Go2 | p99 torque-limit utilization | `FR_thigh_joint` | 0.6365 |
| Go2 | Cyclic severity m=3 / directed m | `FR_thigh_joint` | 0.3649 1/m |
| Go2 | Cyclic severity m=5 / directed m | `FR_thigh_joint` | 0.0649 1/m |
| Go2 | Absolute work / directed m | `FL_calf_joint` | 40.1522 J/m |
| Go2 | Foot p99 vertical force | `front_left` | 116.3440 N |
| Go2 | Foot p99 vertical loading rate | `front_left` | 5069.7067 N/s |
| X1 | RMS torque-limit utilization | `joint_front_left_thigh_pitch` | 0.3354 |
| X1 | p99 torque-limit utilization | `joint_front_left_thigh_pitch` | 0.7210 |
| X1 | Cyclic severity m=3 / directed m | `joint_front_left_thigh_pitch` | 0.7045 1/m |
| X1 | Cyclic severity m=5 / directed m | `joint_front_left_thigh_pitch` | 0.2250 1/m |
| X1 | Absolute work / directed m | `joint_front_left_calf_pitch` | 30.6382 J/m |
| X1 | Foot p99 vertical force | `front_left` | 176.7470 N |
| X1 | Foot p99 vertical loading rate | `front_left` | 8044.7702 N/s |

- All 20 evaluations contain 450 matched
  samples over `0.5 <= t < 9.5 s`, with no reset in
  the window and a common backward sagittal command of -0.566558 m/s.
- Joint names validate the stored front-left, front-right, rear-left, rear-right
  leg-major ordering in every archive.
- In 20/20 archives that retain joint velocity, recomputed
  `torque * joint_velocity` agrees with recorded power to at most 1.525e-05 W;
  recomputed GRF norms agree in all archives to at most 3.479e-05 N.
- All selected GRF samples include friction; vertical impulse is recomputed from
  the raw world-frame z component.

## Evidence limits and next experiment

The scan answers whether this seed is robust to the chosen TRS hyperparameters,
but it cannot estimate the expected TRS effect. The within-rollout block bootstrap
describes temporal variability in one trace; it does not make the 450 samples
independent policies and does not cover training-seed uncertainty.

The saved evaluations contain one reset-free nine-second backward-command window
at 50 Hz. They do not resolve physics-substep impact peaks, long thermal time
constants, rare events, turning, forward gait, terrain, or payload effects.
Rainflow residue is retained as half cycles, but mean-load correction is omitted
because material ultimate strength is unavailable.

A follow-up durability study should use paired independent training seeds and
repeated matched rollouts over forward/backward motion, turning, terrain, payload,
and longer duty cycles. Predeclare component-level endpoints and the X1
equivalence margin. Actual break probability or lifetime additionally requires
motor electrical/thermal parameters, gearbox and bearing ratings, geometry and
materials, stress conversion, component S-N/load-life curves, and failure labels.

Machine-readable details are in `summary.json`, `leg_usage.csv`,
`durability_pair_metrics.csv`, `joint_durability.csv`, `foot_impact.csv`,
`rainflow_cycles.csv`, and `training_efficiency.csv` beside this report. The
two `tensorboard_reward_efficiency_*.svg` files are generated directly from
the raw TensorBoard event logs with:

```powershell
.\isaaclab.bat -p scripts\symm_locomotion\plot_trs_tensorboard.py
```
