# DLARlab Symmetric RL Isaac Lab Backup

## Overview

This repository is an Isaac Lab 3.0.0 beta 2 workspace with the DLARlab
quadruped symmetric locomotion work backed up on top of the upstream Isaac Lab
tree. It contains the code, robot assets, convenience launchers, and selected
training artifacts for:

- Unitree Go2 symmetric gait training migrated from the old IsaacGym project.
- Dobot X1 symmetric flat locomotion training built from the same Go2 task
  structure.
- Legacy time-reversal symmetry regularization for RSL-RL PPO.
- Curated checkpoints, exported TorchScript/ONNX policies, TensorBoard event
  files, and frozen training parameter YAMLs for the good runs.

## Requirements/Environment Setup

Known working local setup:

- Windows PowerShell or Ubuntu bash.
- NVIDIA GPU and driver compatible with Isaac Sim 6.0.0/6.0.1.
- Conda environment named `go2_symm_rl_lab`.
- Isaac Lab checkout at the repository root.
- Isaac Sim binary available through the local `_isaac_sim` folder or through
  the Isaac Sim environment expected by this Isaac Lab branch.

Fresh clone setup:

```powershell
git clone https://github.com/DLARlab/symm_rl_isaaclab.git
cd symm_rl_isaaclab
conda env create -n go2_symm_rl_lab -f environment.yml
conda activate go2_symm_rl_lab
```

If the conda environment already exists, activate it instead:

```powershell
cd D:\go2_symm_rl_lab
conda activate go2_symm_rl_lab
```

Check that the wrapper is using the expected Python:

```powershell
.\isaaclab.bat -p -c "import sys; print(sys.executable)"
```

On Linux, use:

```bash
./isaaclab.sh -p -c "import sys; print(sys.executable)"
```

Verify the custom tasks are registered:

```powershell
.\isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print([s.id for s in gym.registry.values() if 'Go2-Symm' in s.id or 'Dobot-X1-Symm' in s.id])"
```

Expected task IDs:

```text
Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
Isaac-Velocity-Flat-Dobot-X1-Symm-v0
Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0
```

## What We Added

### Go2 Symmetric Task

- Added the `Isaac-Velocity-Flat-Unitree-Go2-Symm-v0` and play task
  registrations.
- Migrated the Go2 URDF/mesh asset into `source/isaaclab_assets`.
- Built a manager-based flat velocity task with 12 joint position actions,
  flat terrain, Go2 PD gains, domain randomization, reset logic, and 30 second
  episodes.
- Recreated the old 60D policy observation layout:
  projected gravity, command, joint position, joint velocity, last action,
  gait phase sine/cosine, gait theta sine/cosine, and swing/stance ratios.
- Added `GaitVelocityCommand` to sample command-conditioned gait phase, period,
  and duty factor.
- Added migrated rewards for command tracking, foot periodicity, base height,
  foot clearance, hip action, morphology symmetry, torque/action smoothness,
  and alive bonus.
- Added nested URDF contact-sensor support and task-local spawners so the foot
  sensors work on the imported URDF hierarchy.

### Dobot X1 Symmetric Task

- Added the `Isaac-Velocity-Flat-Dobot-X1-Symm-v0` and play task
  registrations.
- Added Dobot X1/`dobot_quad_v2` URDF and mesh assets.
- Reused the Go2 gait-command and time-reversal observation/action layout while
  remapping Dobot joint names, default positions, foot links, calf links, and
  morphology symmetry signs.
- Configured Dobot-specific PD gains, action scale, contact sensors, base
  height ranges, termination thresholds, and domain randomization.

### RSL-RL / Symmetry

- Added legacy time-reversal fields to `RslRlSymmetryCfg`.
- Added `LegacyTimeReversalPPO`, which restores warmup-gated policy mirror loss
  and value consistency loss semantics from the old project.
- Added Go2 and Dobot PPO configs that point to the legacy TRS PPO class.
- Added `--print_gait_info` and `--print_gait_info_interval` to RSL-RL play for
  live gait diagnostics when the environment exposes gait command data.

## Code Layout

Main task configs:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/
```

MDP functions:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/go2_symm.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/dobot_x1_symm.py
```

Legacy TRS PPO:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/legacy_trs_ppo.py
source/isaaclab_rl/isaaclab_rl/rsl_rl/symmetry_cfg.py
```

Robot assets:

```text
source/isaaclab_assets/isaaclab_assets/robots/go2_symm/
source/isaaclab_assets/isaaclab_assets/robots/dobot_quad_v2/
```

Convenience wrappers:

```text
scripts/go2_symm/
```

## Good Runs Included

The repository intentionally tracks only selected run artifacts. Routine logs
remain ignored by `.gitignore`.

Go2 no-TRS baseline:

```text
logs/rsl_rl/unitree_go2_symm_flat/Good_Runs/2026-07-07_00-11-14_no_trs/
```

Go2 TRS auxiliary run:

```text
logs/rsl_rl/unitree_go2_symm_flat/Good_Runs/2026-07-10_02-17-16_trs_aux_only_lr1e4_fixed/
```

Dobot X1 no-TRS runs:

```text
logs/rsl_rl/dobot_x1_symm_flat/2026-07-09_14-34-53_dobot_no_trs/
logs/rsl_rl/dobot_x1_symm_flat/2026-07-10_14-30-46_no_trs/
```

Each backed-up run includes:

- `model_0.pt` through `model_9999.pt`.
- `exported/policy.pt`.
- `exported/policy.onnx` and `exported/policy.onnx.data`.
- `params/agent.yaml`.
- `params/env.yaml`.
- TensorBoard `events.out.tfevents...` file.
- A `git/go2_symm_rl_lab.diff` snapshot when the training run captured one.

Use `model_9999.pt` or the exported policy files as the default artifact for
playback unless you are comparing intermediate checkpoints.

## Training

Direct Go2 training:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 256 --max_iterations 5000
```

Direct Dobot X1 training:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 256 --max_iterations 5000
```

Convenience Go2 wrappers:

```powershell
.\scripts\go2_symm\train.bat --iterations 10000 --mirror 0.1
.\scripts\go2_symm\train.bat --no-trs --iterations 10000
.\scripts\go2_symm\ablation.bat --iterations 10000 --seeds 1
```

Linux examples:

```bash
bash scripts/go2_symm/train.sh --iterations 10000 --mirror 0.1
bash scripts/go2_symm/train.sh --no-trs --iterations 10000
bash scripts/go2_symm/ablation.sh --iterations 10000 --seeds 1
```

Resume a specific run:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 256 --resume --load_run Good_Runs/2026-07-10_02-17-16_trs_aux_only_lr1e4_fixed --checkpoint model_9999.pt
```

For Dobot X1, replace the task and run folder:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 256 --resume --load_run 2026-07-10_14-30-46_no_trs --checkpoint model_9999.pt
```

## Playing Policies

Go2 GUI play from a backed-up run:

```powershell
.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0 --num_envs 1 --load_run Good_Runs/2026-07-10_02-17-16_trs_aux_only_lr1e4_fixed --checkpoint model_9999.pt --real-time --viz kit --rendering_mode balanced --kit_args="--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false" --print_gait_info
```

Dobot X1 GUI play:

```powershell
.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0 --num_envs 1 --load_run 2026-07-10_14-30-46_no_trs --checkpoint model_9999.pt --real-time --viz kit --rendering_mode balanced --kit_args="--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"
```

Headless Go2 video:

```powershell
.\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0 --num_envs 1 --video --video_length 400 --load_run Good_Runs/2026-07-10_02-17-16_trs_aux_only_lr1e4_fixed --checkpoint model_9999.pt
```

## How To Modify

Change velocity ranges, terrain, reset thresholds, reward weights, and domain
randomization in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/flat_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/flat_env_cfg.py
```

Change gait command sampling, observation terms, time reversal transforms, and
reward math in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/go2_symm.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/dobot_x1_symm.py
```

Change PPO hyperparameters and TRS coefficients in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/agents/rsl_rl_ppo_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/agents/rsl_rl_ppo_cfg.py
```

Change URDF conversion or nested contact setup in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/spawners.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dobot_x1_symm/spawners.py
source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py
```

Generated USD cache folders under the robot asset directories are intentionally
ignored. They can be regenerated from the tracked URDF and mesh assets.

## Verification

Compile the custom files:

```powershell
.\isaaclab.bat -p -m py_compile source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\flat_env_cfg.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\spawners.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\go2_symm.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\legacy_trs_ppo.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\dobot_x1_symm\flat_env_cfg.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\dobot_x1_symm\spawners.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\dobot_x1_symm.py
```

Run a one-iteration smoke train:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 1 --headless --max_iterations 1
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 1 --headless --max_iterations 1
```

Run formatting and lint checks before committing or pushing:

```powershell
.\isaaclab.bat -f
```

## Notes

- Keep the old IsaacGym project separate from this Isaac Lab migration.
- Add future curated artifacts by naming the run in `.gitignore`; do not commit
  every experiment under `logs/`.
- Do not edit generated changelog outputs directly. Add changelog fragments
  under `source/<package>/changelog.d/`.
