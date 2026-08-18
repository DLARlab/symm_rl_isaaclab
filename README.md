# DLARlab Symmetric RL Isaac Lab Workspace

## Overview

This repository is an Isaac Lab 3.0.0 workspace with DLARlab symmetric
quadruped locomotion tasks and selected good runs. The current implementation
uses one shared symmetric-quadruped task layer for multiple robots:

- Unitree Go2 symmetric flat locomotion.
- Dobot X1 symmetric flat locomotion.
- Time-reversal symmetry regularization for RSL-RL PPO.
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

Install the core Isaac Lab packages after cloning. Rerun this command whenever
the checkout is copied or moved so the editable package paths point at the
current workspace:

```powershell
.\isaaclab.bat -i core
```

Linux equivalent:

```bash
./isaaclab.sh -i core
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
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --num-envs 256 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --num-envs 256 --no-trs
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1
```

Ubuntu/bash:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --iterations 30000 --num-envs 256 --no-trs
bash scripts/symm_locomotion/train.sh --robot x1 --iterations 30000 --num-envs 256 --no-trs
bash scripts/symm_locomotion/play.sh --robot go2 --checkpoint latest
bash scripts/symm_locomotion/play.sh --robot x1 --checkpoint latest
bash scripts/symm_locomotion/record.sh --robot x1 --checkpoint latest --gif
bash scripts/symm_locomotion/compare.sh --robots go2 x1
bash scripts/symm_locomotion/tensorboard.sh --robots go2 x1
```

Direct Python style from an activated environment:

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
.\scripts\symm_locomotion\train.ps1 --robot go2 --iterations 30000 --num-envs 256 --no-trs
.\scripts\symm_locomotion\train.ps1 --robot x1 --iterations 30000 --num-envs 256 --no-trs
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
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Unitree-Go2-Symm-v0 --num_envs 256 --max_iterations 30000
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Velocity-Flat-Dobot-X1-Symm-v0 --num_envs 256 --max_iterations 30000
```

## Playing and Recording

Latest checkpoints:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint latest
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint latest
```

Specific routine run and model:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --run 2026-07-11_20-53-43_more_trs_lr1e4_fixed --model 9999
.\scripts\symm_locomotion\play.ps1 --robot x1 --run 2026-07-11_20-53-48_more_trs_lr1e4_fixed --model 9999
```

`--run` is resolved under the selected robot's routine experiment directory,
for example `logs/rsl_rl/unitree_go2_symm_flat/`. To play a curated
`good_runs` checkpoint, pass the checkpoint path directly:

```powershell
.\scripts\symm_locomotion\play.ps1 --robot go2 --checkpoint logs\rsl_rl\good_runs\unitree_go2_symm_flat\2026-07-07_00-11-14_no_trs\model_9999.pt
.\scripts\symm_locomotion\play.ps1 --robot x1 --checkpoint logs\rsl_rl\good_runs\dobot_x1_symm_flat\2026-07-11_02-59-13_no_trs\model_9999.pt
```

Record videos:

```powershell
.\scripts\symm_locomotion\record.ps1 --robot go2 --checkpoint latest --gif
.\scripts\symm_locomotion\record.ps1 --robot x1 --checkpoint latest --gif
```

Recordings are 30 seconds by default (1,500 environment steps at 50 Hz).
Pass `--video-length` to override the length in environment steps.

Play and record save videos, GIFs, tracking errors, and symmetric rollout
diagnostics together under the selected checkpoint run's `eval/<checkpoint>/`
directory by default, for example `eval/model_9999/`:

```text
sim_data.npz
tracking_errors.txt
figure1_linear_velocities_and_position.png
figure2_E_C_frc_and_contact_forces.png
figure3_E_C_spd_and_foot_velocities.png
figure4_agg_E_C_frc_vs_contact.png
figure5_policy_actions_and_joint_limits.png
figure6_straight_line_reward_diagnostics.png
figure7_foot_clearance.png
figure8_leg_motor_torques.png
figure9_leg_motor_powers.png
figure10_leg_ground_reaction_forces.png
```

The plots compare measured and commanded base velocity/position, `E_C_frc`
and foot contact force, `E_C_spd` and foot speed, policy outputs and joint-limit
use, and the straight-line reward components. `sim_data.npz` retains the
per-joint action, target, position, velocity, applied-torque, signed mechanical
power, soft-limit, and utilization arrays together with signed and magnitude
per-leg torque/power sums, world-frame ground-normal and friction forces, and
measured and commanded swing-foot heights. Standard Go2/X1 playback filters
these forces to the flat ground, so the saved ground-reaction force contains
both normal contact and tangential friction. The leg-use figures plot absolute
motor torques, absolute motor mechanical powers, and absolute world-frame force
components. Their black aggregate curves are L1 sums (`sum(abs(component))`).
Every raw magnitude is paired with a thicker dashed, centered 1-second moving
mean; partial edge windows are correctly normalized and smoothing stops at
episode boundaries. Signed source arrays remain in `sim_data.npz`. The four-row
leg plots use front-left, front-right, rear-left, rear-right order. Pass
`--no-plots` to disable them or `--plots_dir PATH` to select another output
directory. Plot collection is limited to 30 seconds by default; use
`--plot_duration SECONDS` to change the window.

## Logs and Good Runs

Routine experiment directories:

```text
logs/rsl_rl/unitree_go2_symm_flat/
logs/rsl_rl/dobot_x1_symm_flat/
```

Selected backed-up runs are copied under `logs/rsl_rl/good_runs/`. See the
[curated-run index](logs/rsl_rl/good_runs/README.md) and the
[60D-to-72D milestone](logs/rsl_rl/good_runs/MILESTONE_60D_TO_72D.md) for the
complete Go2/X1 run inventory, configuration changes, TRS measurements, and
safe restoration procedure.

Leave routine training outputs in the robot-specific experiment directories
unless a run is intentionally curated and copied into `good_runs`.

Compare recent runs:

```powershell
.\scripts\symm_locomotion\compare.ps1 --robots go2 x1 --limit 5
```

Open TensorBoard:

```powershell
.\scripts\symm_locomotion\tensorboard.ps1 --robots go2 x1 --port 6006
```

On Windows, TensorBoard is launched through `python -m tensorboard.main` and
uses the shared `logs/rsl_rl/` root when multiple robots are selected. On
Linux/macOS, the launcher uses named robot log directories.

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
.\isaaclab.bat -p scripts\symm_locomotion\train.py --robot <robot> --smoke --dry-run --no-conda-run
.\isaaclab.bat -p scripts\symm_locomotion\play.py --robot <robot> --dry-run --no-conda-run
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
.\isaaclab.bat -p -m py_compile scripts\symm_locomotion\symm_cli.py scripts\symm_locomotion\train.py scripts\symm_locomotion\play.py scripts\symm_locomotion\record.py scripts\symm_locomotion\ablation.py scripts\symm_locomotion\compare.py scripts\symm_locomotion\tensorboard.py
.\isaaclab.bat -p scripts\symm_locomotion\train.py --robot go2 --smoke --dry-run --no-conda-run
.\isaaclab.bat -p scripts\symm_locomotion\train.py --robot x1 --smoke --dry-run --no-conda-run
.\isaaclab.bat -p scripts\symm_locomotion\compare.py --robots go2 x1 --limit 1 --dry-run --no-conda-run
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

## Upstream Isaac Lab

This workspace is based on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab),
a GPU-accelerated framework for robot learning built on NVIDIA Isaac Sim. This
checkout tracks the Isaac Lab 3.0.0 and Isaac Sim 6.0.0/6.0.1 generation; use
the project-specific environment and commands above instead of assuming that
instructions for another Isaac Lab release are compatible.

For upstream framework documentation, see:

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)
- [Reinforcement learning workflows](https://isaac-sim.github.io/IsaacLab/source/overview/reinforcement-learning/index.html)
- [Isaac Lab troubleshooting](https://isaac-sim.github.io/IsaacLab/source/refs/troubleshooting.html)

## License

Isaac Lab is released under the [BSD 3-Clause License](LICENSE). The
`isaaclab_mimic` extension and its standalone scripts are released under the
[Apache License 2.0](LICENSE-mimic). Dependency and asset licenses are stored
under [`docs/licenses`](docs/licenses). Isaac Sim also contains components
under proprietary licensing terms; see the
[Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt).

## Citation and Acknowledgement

If this workspace contributes to published research, cite the upstream Isaac
Lab technical report described in the
[Isaac Lab repository](https://github.com/isaac-sim/IsaacLab#citation), along
with the appropriate DLARlab project or experiment artifacts. Isaac Lab grew
from the [Orbit](https://isaac-orbit.github.io/) framework; we acknowledge both
projects and their contributors.
