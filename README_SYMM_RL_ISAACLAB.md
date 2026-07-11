# DLARlab Symmetric RL Isaac Lab Workspace

## Overview

This repository is an Isaac Lab 3.0.0 beta 2 workspace with DLARlab symmetric
quadruped locomotion tasks and selected good runs. The current implementation
uses one shared symmetric-quadruped task layer for multiple robots:

- Unitree Go2 symmetric flat locomotion.
- Dobot X1 symmetric flat locomotion.
- Legacy time-reversal symmetry regularization for RSL-RL PPO.
- Shared train, play, record, ablation, compare, and TensorBoard launchers.

The goal of the structure is to make Go2 and X1 consistent now, while keeping
the path open for more quadruped robots later.

## Requirements

Known working setup:

- Windows PowerShell or Ubuntu bash.
- NVIDIA GPU and driver compatible with Isaac Sim 6.0.0/6.0.1.
- Conda environment named `symm_rl_isaaclab`.
- Isaac Sim available through this Isaac Lab checkout's `_isaac_sim` folder or
  the Isaac Sim environment expected by this branch.

Fresh clone:

```powershell
git clone https://github.com/DLARlab/symm_rl_isaaclab.git
cd symm_rl_isaaclab
conda env create -n symm_rl_isaaclab -f environment.yml
conda activate symm_rl_isaaclab
```

Existing checkout:

```powershell
cd D:\symm_rl_isaaclab
conda activate symm_rl_isaaclab
```

Check the IsaacLab Python:

```powershell
.\isaaclab.bat -p -c "import sys; print(sys.executable)"
```

Linux equivalent:

```bash
./isaaclab.sh -p -c "import sys; print(sys.executable)"
```

Verify the custom task registrations:

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

## Current Structure

Shared MDP and task logic:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/
  symm_quadruped.py      Shared gait command, observation/action transforms,
                         rewards, terminations, and symmetry helpers.
  go2_symm.py            Go2 compatibility adapter over shared MDP logic.
  dobot_x1_symm.py       X1 adapter and X1 morphology constants.
```

Shared config layer:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
  symm_quadruped/
    env.py               Shared ManagerBasedRLEnv subclass.
    flat_env_cfg.py      Shared scene, terrain, observation, reward,
                         termination, and randomization builders.
    spawners.py          Shared nested URDF/contact-sensor spawning helper.
    time_reversal_ppo.py Shared time-reversal PPO implementation.
    agents/
      rsl_rl_ppo_cfg.py  Shared PPO/TRS runner configuration helper.
```

Robot-specific config layers:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
  go2_symm/
    __init__.py
    flat_env_cfg.py
    env.py
    spawners.py
    agents/rsl_rl_ppo_cfg.py

  dobot_x1_symm/
    __init__.py
    flat_env_cfg.py
    env.py
    spawners.py
    agents/rsl_rl_ppo_cfg.py
```

Robot-specific files should mainly contain assets, joint order, default joint
positions, body names, contact sensor names, actuator gains, base height ranges,
task registrations, and experiment names.

Assets:

```text
source/isaaclab_assets/isaaclab_assets/robots/go2_symm/
source/isaaclab_assets/isaaclab_assets/robots/dobot_quad_v2/
```

Shared utility scripts:

```text
scripts/symm_locomotion/
  symm_cli.py
  symm_locomotion.sh/ps1
  train.py/sh/ps1
  play.py/sh/ps1
  record.py/sh/ps1
  ablation.py/sh/ps1
  compare.py/sh/ps1
  tensorboard.py/sh/ps1
```

## Script Usage

Use `scripts/symm_locomotion` for normal work.

Windows PowerShell:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --num-envs 4096 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --num-envs 4096 --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --video-length 400 --gif
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1
```

Ubuntu/bash:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --iterations 30000 --num-envs 4096 --no-trs
bash scripts/symm_locomotion/train.sh --robot x1 --iterations 30000 --num-envs 4096 --no-trs
bash scripts/symm_locomotion/play.sh --robot go2 --checkpoint latest
bash scripts/symm_locomotion/play.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/record.sh --robot x1 --checkpoint latest --video-length 400 --gif
bash scripts/symm_locomotion/compare.sh --robots go2 x1
bash scripts/symm_locomotion/tensorboard.sh --robots go2 x1
```

Direct Python style:

```bash
python scripts/symm_locomotion/train.py --robot go2 --iterations 30000 --no-trs
python scripts/symm_locomotion/play.py --robot x1 --checkpoint latest
```

Generic launcher style:

```powershell
.\scripts\symm_locomotion\symm_locomotion.ps1 train --robot go2 --smoke --dry-run
```

```bash
bash scripts/symm_locomotion/symm_locomotion.sh train --robot x1 --smoke --dry-run
```

All commands accept `--dry-run` to print the resolved Isaac Lab command without
running it. Extra Isaac Lab/Hydra overrides can be passed after `--`.

## Training

Good no-TRS baselines:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --num-envs 4096 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --num-envs 4096 --no-trs
```

TRS/mirror-loss runs:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --mirror 0.1
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --mirror 0.1
```

One-iteration smoke runs:

```powershell
.\scripts\symm_locomotion\train.ps1 --robot go2 --smoke --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --smoke --no-trs
```

Linux background runs:

```bash
bash scripts/symm_locomotion/train.sh --nohup --robot go2 --iterations 30000 --no-trs
bash scripts/symm_locomotion/ablation.sh --nohup --robot x1 --iterations 10000 --seeds 1 2 3
```

`--no-trs` forwards:

```text
agent.algorithm.symmetry_cfg.use_data_augmentation=False
agent.algorithm.symmetry_cfg.use_mirror_loss=False
agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0
agent.algorithm.symmetry_cfg.value_loss_coeff=0.0
```

Direct IsaacLab commands still work:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 4096 --max_iterations 30000
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 4096 --max_iterations 30000
```

## Playing and Recording

Latest checkpoints:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
```

Specific run and model:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --run good_runs/unitree_go2_symm_flat/2026-07-07_00-11-14_no_trs --model 9999
.\scripts\symm_locomotion\play.ps1 --robot x1 --run good_runs/dobot_x1_symm_flat/2026-07-11_02-59-13_no_trs --model 9999
```

Record videos:

```powershell
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --video-length 400 --gif
.\scripts\symm_locomotion\record.ps1 --robot x1 --checkpoint latest --video-length 400 --gif
```

## Logs and Good Runs

Routine experiment directories:

```text
logs/rsl_rl/unitree_go2_symm_flat/
logs/rsl_rl/dobot_x1_symm_flat/
```

Selected backed-up runs are copied under:

```text
logs/rsl_rl/good_runs/unitree_go2_symm_flat/2026-07-07_00-11-14_no_trs/
logs/rsl_rl/good_runs/unitree_go2_symm_flat/2026-07-10_02-17-16_trs_aux_only_lr1e4_fixed/
logs/rsl_rl/good_runs/dobot_x1_symm_flat/2026-07-11_02-59-13_no_trs/
```

Only files under `logs/rsl_rl/good_runs/` are tracked by Git. Leave routine
training outputs in the robot-specific experiment directories unless a run is
curated and copied into `good_runs`.

Compare recent runs:

```powershell
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1 --limit 5
```

Open TensorBoard:

```powershell
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1 --port 6006
```

## How To Modify

Shared behavior belongs in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symm_quadruped.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/flat_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/symm_quadruped/agents/rsl_rl_ppo_cfg.py
```

Use shared files for:

- Gait command behavior and sampling.
- Time-reversal observation/action transforms.
- Reward and termination math that should apply to all symmetric quadrupeds.
- Observation layout.
- Domain randomization shared by all robots.
- PPO/TRS defaults shared by all robots.
- Script behavior that can be selected by `--robot`.

Robot-specific behavior belongs in:

```text
config/go2_symm/
config/dobot_x1_symm/
mdp/go2_symm.py
mdp/dobot_x1_symm.py
```

Use robot files for:

- URDF/USD asset paths.
- Joint order and default positions.
- Foot, calf, base, and contact sensor names.
- Actuator gains and action scale.
- Robot-specific base height ranges and termination thresholds.
- Morphology signs/ranges when the shared symmetry defaults do not match.
- Task IDs and experiment names.

Avoid copying a whole robot folder when adding a new robot. Start with a small
robot-specific config that calls the shared `symm_quadruped` builders.

## Adding a New Robot

1. Add robot assets under `source/isaaclab_assets/isaaclab_assets/robots/`.
2. Create `config/<robot>_symm/__init__.py` and register train/play Gym tasks.
3. Create `config/<robot>_symm/flat_env_cfg.py` with only robot-specific asset,
   joint, actuator, contact, and height-range definitions.
4. Create `config/<robot>_symm/agents/rsl_rl_ppo_cfg.py` by calling
   `configure_symm_quadruped_ppo(...)`.
5. Create `mdp/<robot>_symm.py` only for morphology constants/adapters that
   differ from the shared defaults.
6. Add a `RobotSpec` entry in `scripts/symm_locomotion/symm_cli.py`.
7. Add a dry-run CLI test for the new robot.

Before training, run:

```powershell
python scripts/symm_locomotion/train.py --robot <robot> --smoke --dry-run --no-conda-run
python scripts/symm_locomotion/play.py --robot <robot> --dry-run --no-conda-run
```

Then run a one-iteration IsaacLab smoke train.

## Improving Safely

When changing shared logic:

- First verify Go2 and X1 still resolve the same task IDs and log directories.
- Keep public adapters such as `mdp/go2_symm.py` unless there is a planned
  deprecation.
- Keep observation/action order stable unless retraining all policies is
  intentional.
- Add parameters to shared helpers instead of hard-coding one robot's names.
- Run the CLI dry-run tests before launching expensive simulation.
- Run at least one Go2 and one X1 smoke train after reward, command, or config
  changes.

## Verification

Lightweight checks:

```powershell
python -m py_compile scripts\symm_locomotion\symm_cli.py scripts\symm_locomotion\train.py scripts\symm_locomotion\play.py scripts\symm_locomotion\record.py scripts\symm_locomotion\ablation.py scripts\symm_locomotion\compare.py scripts\symm_locomotion\tensorboard.py
python scripts\symm_locomotion\train.py --robot go2 --smoke --dry-run --no-conda-run
python scripts\symm_locomotion\train.py --robot x1 --smoke --dry-run --no-conda-run
python scripts\symm_locomotion\compare.py --robots go2 x1 --limit 1 --dry-run --no-conda-run
```

Conda/IsaacLab tests:

```powershell
conda run --no-capture-output -n symm_rl_isaaclab .\isaaclab.bat -p -m pytest source\isaaclab_tasks\test\test_go2_symm_time_reversal.py source\isaaclab_tasks\test\test_symm_quadruped_time_reversal_ppo.py source\isaaclab_tasks\test\test_symm_locomotion_cli.py
```

Full pre-commit before committing or pushing:

```powershell
.\isaaclab.bat -f
```

## Notes

- Keep the old IsaacGym project separate from this Isaac Lab migration.
- Keep routine logs ignored; add only curated runs under
  `logs/rsl_rl/good_runs/` intentionally.
- Do not edit generated changelog outputs directly. Add changelog fragments
  under `source/<package>/changelog.d/` when needed.
