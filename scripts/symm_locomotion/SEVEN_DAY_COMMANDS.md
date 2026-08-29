<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Seven-Day Symmetric-Locomotion Command Matrix

This command matrix holds the training budget at 20,000 iterations and 512
environments. Days 1-4 are seed-42 development runs, not confirmatory evidence.
The eight registered treatment roles are crossed with both robots to produce
exactly sixteen runs.

The Go2 `g2fc1` profile uses `0.4 * mean` foot-phase reduction,
requested-target overflow, per-joint feasible actor bounds, and normalized
requested-target TR policy consistency. The X1 `x1def` profile explicitly
preserves the current `0.3 * sum`, legacy target-limit, legacy actor-bound, and
raw-action defaults. Both profiles use the equal-family gait distribution with
no gait curriculum. Analyze the robots separately because these profiles are
not the same intervention.

## Day 1: no TRS and weak hard TRS

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d1_go2_dev_main_notrs_g2fc1_s42 --seed 42 --no-trs --tr-policy-coef 0.0 --tr-value-coef 0.0 --tr-warmup-iterations 0 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d1_x1_dev_main_notrs_x1def_s42 --seed 42 --no-trs --tr-policy-coef 0.0 --tr-value-coef 0.0 --tr-warmup-iterations 0 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d1_go2_dev_main_m0p1_v0p05_w500_r0_g2fc1_s42 --seed 42 --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d1_x1_dev_main_m0p1_v0p05_w500_r0_x1def_s42 --seed 42 --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

## Day 2: weak ramped and high hard TRS

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d2_go2_dev_main_m0p1_v0p05_w500_r1000_g2fc1_s42 --seed 42 --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d2_x1_dev_main_m0p1_v0p05_w500_r1000_x1def_s42 --seed 42 --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d2_go2_dev_main_m0p2_v0p1_w500_r0_g2fc1_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 500 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d2_x1_dev_main_m0p2_v0p1_w500_r0_x1def_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 500 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

## Day 3: high ramped and delayed high TRS

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d3_go2_dev_main_m0p2_v0p1_w500_r1000_g2fc1_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d3_x1_dev_main_m0p2_v0p1_w500_r1000_x1def_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d3_go2_dev_supp_m0p2_v0p1_w1000_r1000_g2fc1_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 1000 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d3_x1_dev_supp_m0p2_v0p1_w1000_r1000_x1def_s42 --seed 42 --tr-policy-coef 0.2 --tr-value-coef 0.1 --tr-warmup-iterations 1000 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

## Day 4: supplementary coefficient separation

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d4_go2_dev_supp_m0p15_v0p05_w500_r1000_g2fc1_s42 --seed 42 --tr-policy-coef 0.15 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d4_x1_dev_supp_m0p15_v0p05_w500_r1000_x1def_s42 --seed 42 --tr-policy-coef 0.15 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name d4_go2_dev_supp_m0p15_v0p025_w500_r1000_g2fc1_s42 --seed 42 --tr-policy-coef 0.15 --tr-value-coef 0.025 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name d4_x1_dev_supp_m0p15_v0p025_w500_r1000_x1def_s42 --seed 42 --tr-policy-coef 0.15 --tr-value-coef 0.025 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

## Days 5-7: prospective paired-seed template

Use the primary no-TRS versus weak-ramped-TRS contrast for both robots. Set
`Day` and `Seed` before each block: Day 5 uses seed 101, Day 6 uses seed 202,
and Day 7 uses seed 303. The same seed blocks the two treatments within each
robot; do not pool the two robots as replicate observations.

```powershell
$Day = 5
$Seed = 101

.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name "d${Day}_go2_pros_primary_notrs_g2fc1_s${Seed}" --seed $Seed --no-trs --tr-policy-coef 0.0 --tr-value-coef 0.0 --tr-warmup-iterations 0 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot go2 --num-envs 512 --iterations 20000 --run-name "d${Day}_go2_pros_primary_m0p1_v0p05_w500_r1000_g2fc1_s${Seed}" --seed $Seed --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.4 --foot-phase-reduction mean --joint-target-limit-mode requested_overflow --joint-target-limit-weight 0.05 --actor-mean-bound-mode per_joint_feasible --tr-policy-output-space normalized_requested_joint_target --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name "d${Day}_x1_pros_primary_notrs_x1def_s${Seed}" --seed $Seed --no-trs --tr-policy-coef 0.0 --tr-value-coef 0.0 --tr-warmup-iterations 0 --tr-rampup-iterations 0 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
.\scripts\symm_locomotion\train.ps1 --robot x1 --num-envs 512 --iterations 20000 --run-name "d${Day}_x1_pros_primary_m0p1_v0p05_w500_r1000_x1def_s${Seed}" --seed $Seed --tr-policy-coef 0.1 --tr-value-coef 0.05 --tr-warmup-iterations 500 --tr-rampup-iterations 1000 --tr-ramp-shape linear --tr-min-abs-cmd-vel 0.0 --foot-phase-weight 0.3 --foot-phase-reduction sum --joint-target-limit-mode legacy_clamped --joint-target-limit-weight 0.05 --actor-mean-bound-mode legacy_global --tr-policy-output-space raw_action_mean --gait-sampling-profile trclosed_v2_equal_family --gait-curriculum-iterations 0 --expected-branch 72d-symm-v4-integration --no-conda-run -- agent.algorithm.symmetry_cfg.tr_augmentation.enabled=false
```

The seed-42 runs remain development evidence. Classify Days 5-7 as a
prospective replication only after freezing the code hash, primary endpoints,
failure policy, and analysis before Day 5. Three fresh seeds do not alone
justify a performance-confirmation claim.

## Light and full evaluation

Assign each completed final-checkpoint path before running the matching robot
commands. Evaluation seed 4242 is fixed across all checkpoints and distinct
from the training seeds. The full protocol retains its default ten gait rows
and six commanded velocities.

```powershell
$Go2Checkpoint = 'D:\symm_rl_isaaclab\logs\rsl_rl\unitree_go2_symm_flat\YYYY-MM-DD_HH-MM-SS_RUN_NAME\model_19999.pt'
$X1Checkpoint = 'D:\symm_rl_isaaclab\logs\rsl_rl\dobot_x1_symm_flat\YYYY-MM-DD_HH-MM-SS_RUN_NAME\model_19999.pt'

.\scripts\symm_locomotion\evaluation.ps1 --robot go2 --checkpoint $Go2Checkpoint --protocol light --evaluation-seed 4242 --expected-branch 72d-symm-v4-integration --no-conda-run
.\scripts\symm_locomotion\evaluation.ps1 --robot go2 --checkpoint $Go2Checkpoint --protocol full --evaluation-seed 4242 --expected-branch 72d-symm-v4-integration --no-conda-run
.\scripts\symm_locomotion\evaluation.ps1 --robot x1 --checkpoint $X1Checkpoint --protocol light --evaluation-seed 4242 --expected-branch 72d-symm-v4-integration --no-conda-run
.\scripts\symm_locomotion\evaluation.ps1 --robot x1 --checkpoint $X1Checkpoint --protocol full --evaluation-seed 4242 --expected-branch 72d-symm-v4-integration --no-conda-run
```
