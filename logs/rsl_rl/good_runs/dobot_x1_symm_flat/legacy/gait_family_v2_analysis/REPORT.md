# Dobot X1 gait-closure comparison

Method: `gait_closure_comparison_v1`

This analysis compares the ordered run cohort in `study.json`. The no-TRS baseline is always black; other runs use the gait-family-v2 palette.

## Metric and aggregation contract

- Velocity plots use literal forward-velocity RMSE [m/s] and yaw-velocity RMSE [rad/s].
- Leg plots use per-cell absolute front/hind imbalance for torque², normalized torque², absolute work, and vertical GRF impulse.
- Normalized torque² is the archived `normalized_torque_utilization` metric, computed from physical joint effort limits.
- Gait fidelity uses boundary-excluded contact-pattern agreement.
- All, negative, and positive scopes contain 60, 30, and 30 cells per run. No outcome or tracking-success filter is applied.
- Family rows average selected cells within one family. Overall rows first compute each family mean and then weight the four gait families equally.

## Run cohort

| Abbreviation | Run | Color | Treatment |
|---|---|---|---|
| No TRS | `2026-08-20_21-04-26_x1_72d_no_trs_gait_trclosed_v2` | `#202124` | No TRS |
| TRS m0.1/v0.05 | `2026-08-21_06-37-31_x1_72d_trs_m0p1_v0p05_w500_r0_gait_trclosed_v2` | `#0072B2` | policy=0.1 (warmup=500, ramp=0); value=0.05 (warmup=500, ramp=0) |
| TRS m0.2/v0.1 | `2026-08-21_22-50-05_x1_72d_trs_m0p2_v0p1_w500_r0_gait_trclosed_v2` | `#56B4E9` | policy=0.2 (warmup=500, ramp=0); value=0.1 (warmup=500, ramp=0) |

## Comparability notes

- Legacy checksum-only runs lack pre-first-rollout initialization provenance, and their archived `model_0.pt` bytes are unavailable under the terminal-checkpoint-only publication policy. The registry-bound checksum record is auditable, but the checkpoint bytes were not reverified and the pre-rollout model, optimizer, RNG, and runtime identity cannot be audited: No TRS, TRS m0.1/v0.05, TRS m0.2/v0.1.
- Training source provenance differs across runs; baseline deltas are descriptive rather than a strict single-factor causal contrast.

## Figures

Figure IDs, filenames, source tables, and hashes are recorded in `figure_manifest.json`.

- **fig01** Training efficiency
- **fig02** Velocity tracking: overall and by direction
- **fig03_01** Velocity tracking by gait family: all velocities
- **fig03_02** Velocity tracking by gait family: negative velocities
- **fig03_03** Velocity tracking by gait family: positive velocities
- **fig03** Velocity tracking by gait family: combined
- **fig04** Leg usage: overall and by direction
- **fig05_01** Leg usage by gait family: all velocities
- **fig05_02** Leg usage by gait family: negative velocities
- **fig05_03** Leg usage by gait family: positive velocities
- **fig05** Leg usage by gait family: combined
- **fig06** Gait fidelity: overall and by direction
- **fig07_01** Gait fidelity by family: all velocities
- **fig07_02** Gait fidelity by family: negative velocities
- **fig07_03** Gait fidelity by family: positive velocities
- **fig07** Gait fidelity by family: combined

## Reproduction requirements

Each run needs its terminal checkpoint, one `events.out.tfevents.*` file containing `Train/mean_reward`, training metadata, and a complete `evaluations/leg_usage_grid_full_v3` directory containing `study.json`, `progress.json`, `metrics/cell_metrics.csv`, `metrics/overall_metrics.json`, and `metrics/analysis_provenance.json`. Raw NPZ recordings are not reread.

Training metadata normally comes from `provenance/initialization.json`. Legacy runs may instead use registered `params/agent.yaml` and `params/env.yaml` snapshots plus `model_0.pt`. A terminal-checkpoint-only archive may replace the omitted checkpoint bytes with the strictly validated `provenance/legacy_initialization.json` checksum record. These weaker, distinct provenance classes are identified in `study.json`, `analysis_provenance.json`, and the comparability notes above.

Run `reproduce.py` with the Isaac Lab Python wrapper to regenerate the datasets and figures.
`source_manifest_snapshot.json` preserves the exact manifest-mode input; in direct mode it mirrors the generated canonical `study.json`.
