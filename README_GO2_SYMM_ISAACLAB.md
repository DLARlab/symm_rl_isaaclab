# Go2 Symmetric RL for Isaac Lab

This document describes the Isaac Lab migration of the old IsaacGym Go2
symmetric gait project. The migrated task keeps the useful IsaacGym pieces:
the Go2 asset, 60D gait-clock observation, command-conditioned gait phase
generator, foot periodicity reward, reset and termination logic, domain
randomization, RSL-RL PPO config, and time-reversal symmetry augmentation.

The original IsaacGym project is kept separate at:

```powershell
C:\Users\jding\Documents\Github\go2_symm_rl_2
```

Do not edit the old project while working on this Isaac Lab version. The Isaac
Lab migration lives in:

```powershell
D:\go2_symm_rl_lab
```

## Current Status

The migrated task is registered as:

```text
Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
```

The minimum Isaac Lab environment has been verified with one environment. It
loads the copied Go2 URDF asset, resets, steps, exposes the 60D policy
observation, runs the migrated reward terms, and completes a one-iteration
RSL-RL headless training smoke test.

## Requirements

- Windows PowerShell.
- Conda environment named `go2_symm_rl_lab`.
- Isaac Sim and Isaac Lab installed in `D:\go2_symm_rl_lab`.
- NVIDIA GPU and drivers compatible with the installed Isaac Sim version.
- No global Python packages are required for this migrated task.

Use the Isaac Lab wrapper and the existing conda environment. Do not install
packages into system Python.

## Code Layout

Main task registration:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/__init__.py
```

Environment config:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/flat_env_cfg.py
```

Task-local URDF spawner and nested contact-report activation:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/spawners.py
```

RSL-RL PPO and symmetry config:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_symm/agents/rsl_rl_ppo_cfg.py
```

Migrated MDP terms:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/go2_symm.py
```

Copied Go2 asset:

```text
source/isaaclab_assets/isaaclab_assets/robots/go2_symm/urdf/go2.urdf
source/isaaclab_assets/isaaclab_assets/robots/go2_symm/dae/
```

Generated USD converter caches are intentionally ignored under the copied asset
directory.

## Migrated Features

Asset:

- Copied Go2 URDF and mesh assets from the old IsaacGym project.
- Uses a task-local URDF spawner so nested URDF links get PhysX contact
  reporting.
- Keeps the old project untouched.

Environment:

- Manager-based Isaac Lab velocity task.
- Flat generated terrain.
- 12 joint position actions.
- Default Go2 pose with hip, thigh, and calf initial angles.
- Go2 PD gains and action scale migrated from the old setup.

Observations:

- 60D policy observation.
- Projected gravity.
- Velocity command.
- Relative joint positions.
- Relative joint velocities.
- Last action.
- Foot phase sine/cosine.
- Foot theta sine/cosine.
- Swing and stance phase ratios.

Commands:

- Custom `GaitVelocityCommand`.
- Forward velocity range `[-2.0, 2.0]` m/s.
- Lateral and yaw commands currently fixed at zero.
- Gait period and duty factor are derived from commanded forward speed.
- Foot theta offsets are sampled from the old gait list.

Rewards:

- Alive bonus.
- Command tracking penalty.
- Foot periodicity penalty with both:
  - swing contact-force penalty from foot contact sensors
  - stance foot-speed penalty from foot body velocities
- Base height range penalty.
- Foot clearance penalty.
- Hip action penalty.
- Morphological symmetry penalty.
- Torque smoothness penalty.

Terminations:

- Time out.
- Base contact.
- Base height out of range.
- Base roll or pitch out of range.
- Calf body height below threshold.

Domain randomization:

- Static and dynamic friction range `(0.3, 2.0)`.
- Base mass additive randomization `(-1.5, 1.5)` kg.
- Periodic base push with x/y velocity range `(-0.25, 0.25)` m/s.

Symmetry:

- RSL-RL symmetry config is enabled.
- Uses data augmentation and mirror loss.
- Time-reversal transform is implemented for the 60D observation layout.
- Isaac Lab RSL-RL does not directly expose the old separate TR value loss,
  warmup, and minimum command velocity knobs in the same form.

## Environment Setup

Open PowerShell:

```powershell
cd D:\go2_symm_rl_lab
conda activate go2_symm_rl_lab
```

Check that the Isaac Lab wrapper uses the expected Python:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat -p -c "import sys; print(sys.executable)"
```

Expected Python path should be under:

```text
C:\Users\jding\.conda\envs\go2_symm_rl_lab
```

## Verification

Run these checks after changing the task.

### 1. Import And Task Registration

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print([s.id for s in gym.registry.values() if 'Go2-Symm' in s.id])"
```

Expected output includes:

```text
Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
```

### 2. Python Compile Check

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat -p -m py_compile source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\flat_env_cfg.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\spawners.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\mdp\go2_symm.py source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_symm\agents\rsl_rl_ppo_cfg.py
```

### 3. One-Environment Headless Smoke Test

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 1 --headless --max_iterations 1
```

This should create a short run under:

```text
logs/rsl_rl/unitree_go2_symm_flat/
```

The training output should show:

- actor input dimension `60`
- action dimension `12`
- active command term `GaitVelocityCommand`
- active reward term `foot_periodicity`
- `Mean symmetry loss`

### 4. Contact Sensor Inspection

Use this when changing the Go2 asset, spawner, or contact rewards:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat -p -c "from isaaclab.app import AppLauncher; app=AppLauncher(headless=True).app; import gymnasium as gym; import isaaclab_tasks; from isaaclab_tasks.utils import parse_env_cfg; task='Isaac-Velocity-Flat-Unitree-Go2-Symm-v0'; cfg=parse_env_cfg(task, device='cuda:0', num_envs=1); env=gym.make(task, cfg=cfg); env.reset(); scene=env.unwrapped.scene; print('sensors:', sorted(scene.sensors.keys())); print('contact_forces bodies:', scene.sensors['contact_forces'].body_names); [print(name, scene.sensors[name].body_names, tuple(scene.sensors[name].data.net_forces_w_history.torch.shape)) for name in ('contact_FL_foot','contact_FR_foot','contact_RL_foot','contact_RR_foot')]; env.close(); app.close()"
```

Expected foot sensors:

```text
contact_FL_foot ['FL_foot'] (1, 3, 1, 3)
contact_FR_foot ['FR_foot'] (1, 3, 1, 3)
contact_RL_foot ['RL_foot'] (1, 3, 1, 3)
contact_RR_foot ['RR_foot'] (1, 3, 1, 3)
```

## Training

### Convenience Scripts

The migrated Isaac Lab task includes short wrappers under:

```text
scripts/go2_symm/
```

Windows PowerShell examples:

```powershell
.\scripts\go2_symm\train.ps1 --num-envs 256 --iterations 30000 --mirror 0.2
.\scripts\go2_symm\play.ps1 --checkpoint latest
.\scripts\go2_symm\record.ps1 --checkpoint latest --video-length 400 --gif
.\scripts\go2_symm\ablation.ps1 --num-envs 256 --iterations 10000 --seeds 1
```

Ubuntu / bash examples:

```bash
bash scripts/go2_symm/train.sh --num-envs 256 --iterations 30000 --mirror 0.2
bash scripts/go2_symm/play.sh --checkpoint latest
bash scripts/go2_symm/record.sh --checkpoint latest --video-length 400 --gif
bash scripts/go2_symm/ablation.sh --num-envs 256 --iterations 10000 --seeds 1
```

Direct Python-style examples:

```bash
python scripts/go2_symm/train.py --num-envs 256 --iterations 30000 --mirror 0.2
python scripts/go2_symm/play.py --checkpoint latest
python scripts/go2_symm/record.py --checkpoint latest --gif
python scripts/go2_symm/ablation.py --only both --seeds 1
```

These wrappers still call `isaaclab.bat` or `isaaclab.sh` underneath. If the
active conda environment is not `go2_symm_rl_lab`, they wrap the command with
`conda run --no-capture-output -n go2_symm_rl_lab`.

Default short training:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 256 --max_iterations 5000
```

Small debug training:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 8 --max_iterations 10
```

Resume a run:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 256 --resume --load_run <RUN_FOLDER> --checkpoint <CHECKPOINT_FILE>
```

Example checkpoint file names usually look like:

```text
model_1000.pt
model_2000.pt
```

Training logs are written to:

```text
logs/rsl_rl/unitree_go2_symm_flat/<RUN_FOLDER>/
```

## Playing A Policy

For GUI play on this machine, use the D3D12-friendly Kit arguments:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0 --num_envs 1 --real-time --viz kit --rendering_mode balanced --kit_args="--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"
```

Play a specific run and checkpoint:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0 --num_envs 1 --load_run <RUN_FOLDER> --checkpoint <CHECKPOINT_FILE> --real-time --viz kit --rendering_mode balanced --kit_args="--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"
```

Headless play with video recording:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat play --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0 --num_envs 1 --video --video_length 400 --load_run <RUN_FOLDER> --checkpoint <CHECKPOINT_FILE>
```

## Useful Configuration Edits

Change command ranges in:

```text
flat_env_cfg.py
```

Look for:

```python
self.commands.base_velocity = go2_symm_mdp.GaitVelocityCommandCfg(...)
```

Change reward weights in:

```text
flat_env_cfg.py
```

Look for:

```python
self.rewards.foot_periodicity = RewTerm(...)
self.rewards.morphological_symmetry = RewTerm(...)
```

Change PPO and symmetry settings in:

```text
agents/rsl_rl_ppo_cfg.py
```

Look for:

```python
self.algorithm.symmetry_cfg = RslRlSymmetryCfg(...)
```

Change the time-reversal observation transform in:

```text
mdp/go2_symm.py
```

Look for:

```python
time_reverse_observations(...)
compute_time_reversal_states(...)
```

## Troubleshooting

### Task Not Found

If this returns `[]`:

```powershell
conda run -n go2_symm_rl_lab .\isaaclab.bat -p -c "import gymnasium as gym; import isaaclab_tasks; print([s.id for s in gym.registry.values() if 'Go2-Symm' in s.id])"
```

check:

- You are running from `D:\go2_symm_rl_lab`.
- You are using conda env `go2_symm_rl_lab`.
- `source/isaaclab_tasks/.../config/go2_symm/__init__.py` exists and is not
  ignored by Git.

### Early `pxr` Import Or SimulationApp Startup Errors

Isaac Sim is sensitive to importing `pxr` modules before `SimulationApp`
starts. The Go2 task-local spawner intentionally imports `pxr` lazily, inside
runtime helper functions. Avoid moving `pxr` imports to module top level in
the migrated task package.

### Contact Sensor Warnings

The imported URDF has a nested link tree under:

```text
/World/envs/env_0/Robot/Geometry/base/...
```

The task uses:

- a broad `contact_forces` sensor for the body list
- one single-body sensor per foot for reliable foot force histories

If foot periodicity rewards become zero or contact bodies disappear, rerun the
contact sensor inspection command above.

### Viewer Does Not Open Or Uses The Wrong Renderer

Use:

```powershell
--viz kit --rendering_mode balanced --kit_args="--/app/vulkan=false --/rtx/hydra/mdlMaterialWarmup=false"
```

This avoids the Vulkan path on systems that need D3D12.

The convenience play script uses this Windows-friendly viewer path by default.
On Ubuntu, the convenience scripts use `isaaclab.sh` and do not add the Windows
D3D12 Kit arguments unless you pass `--kit-args` explicitly.

## Migration Notes

The Isaac Lab version intentionally follows Isaac Lab conventions instead of
copying the old IsaacGym project layout. The old direct environment class is
split into:

- config classes in `config/go2_symm`
- MDP functions and command terms in `mdp/go2_symm.py`
- RSL-RL config in `agents/rsl_rl_ppo_cfg.py`
- assets in `source/isaaclab_assets`

Hardware deployment from the old project has not been migrated yet. Treat this
README as the simulation and training guide for the Isaac Lab version.
