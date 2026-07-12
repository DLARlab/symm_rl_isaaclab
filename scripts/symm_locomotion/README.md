# Symmetric Locomotion Scripts

This folder is the shared launcher surface for the symmetric quadruped tasks.
Use it for both Unitree Go2 and Dobot X1, and add future robots here instead
of creating another robot-specific script tree.

Supported robot keys:

```text
go2    Unitree Go2
x1     Dobot X1
```

Aliases are also accepted: `unitree-go2`, `unitree_go2`, `dobot`,
`dobot-x1`, and `dobot_x1`.

## Quick Commands

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

Direct Python style still works from an activated environment:

```bash
python scripts/symm_locomotion/train.py --robot go2 --iterations 30000 --no-trs
python scripts/symm_locomotion/play.py --robot x1 --checkpoint latest
```

The generic launcher accepts the command as its first argument:

```bash
bash scripts/symm_locomotion/symm_locomotion.sh train --robot go2 --smoke --dry-run
```

```powershell
.\scripts\symm_locomotion\symm_locomotion.ps1 train --robot x1 --smoke --dry-run
```

## Common Options

Every command accepts:

```text
--robot go2|x1
--conda-env symm_rl_isaaclab
--use-conda-run
--no-conda-run
--dry-run
```

Training options:

```text
--iterations / --max-iterations
--num-envs
--run-name
--seed
--mirror
--tr-value-coef
--tr-warmup-iterations
--tr-min-abs-cmd-vel
--no-trs
--smoke
```

`--no-trs` disables symmetry data augmentation, mirror loss, and TRS value
loss by forwarding these Hydra overrides:

```text
agent.algorithm.symmetry_cfg.use_data_augmentation=False
agent.algorithm.symmetry_cfg.use_mirror_loss=False
agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.0
agent.algorithm.symmetry_cfg.value_loss_coeff=0.0
```

`--smoke` uses one environment and one training iteration.

Play and record resolve `--checkpoint latest` from the newest run under the
selected robot experiment directory. You can also use:

```text
--run RUN_FOLDER_OR_PATH
--model 9999
--checkpoint PATH_TO_MODEL_PT
```

Relative `--run` values are resolved under the selected robot's routine log
directory, such as `logs/rsl_rl/unitree_go2_symm_flat/`. For curated
`logs/rsl_rl/good_runs/` checkpoints, pass the checkpoint path directly with
`--checkpoint`.

Extra Isaac Lab or Hydra overrides can be passed after `--`:

```bash
bash scripts/symm_locomotion/train.sh --robot go2 --no-trs -- \
  env.commands.base_velocity.ranges.lin_vel_x='(-1.0, 2.0)'
```

## Logs

The script maps each robot to its task and experiment directory:

```text
go2 train: Isaac-Velocity-Flat-Unitree-Go2-Symm-v0
go2 play:  Isaac-Velocity-Flat-Unitree-Go2-Symm-Play-v0
go2 logs:  logs/rsl_rl/unitree_go2_symm_flat/

x1 train:  Isaac-Velocity-Flat-Dobot-X1-Symm-v0
x1 play:   Isaac-Velocity-Flat-Dobot-X1-Symm-Play-v0
x1 logs:   logs/rsl_rl/dobot_x1_symm_flat/
```

`compare` prints recent run folders and latest checkpoints across robots.
`tensorboard` starts TensorBoard through `python -m tensorboard.main`. On
Windows, multiple selected robots use the shared `logs/rsl_rl/` root because
TensorBoard's named logdir grammar conflicts with drive letters. On Linux and
macOS, the launcher uses named log directories for the selected robots.

## Background Runs

On Linux, `train.sh` and `ablation.sh` support `--nohup`:

```bash
bash scripts/symm_locomotion/train.sh --nohup --robot go2 --iterations 30000 --no-trs
bash scripts/symm_locomotion/ablation.sh --nohup --robot x1 --iterations 10000 --seeds 1 2 3
```

Logs are written under:

```text
logs/symm_locomotion/
```

## Adding Another Robot

Add a new robot in four places:

1. Register the Isaac Lab tasks under
   `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/<robot>_symm/__init__.py`.
2. Put robot-specific assets, joint names, body names, contact sensors, default
   joint positions, height ranges, and actuator gains in
   `config/<robot>_symm/flat_env_cfg.py`.
3. Add robot morphology constants or a small adapter in
   `mdp/<robot>_symm.py` only when the shared defaults do not match.
4. Add one `RobotSpec` entry and any aliases in `symm_cli.py`.

Keep shared reward math, gait command logic, time-reversal transforms, wrapper
behavior, and PPO defaults in the `symm_quadruped` modules unless the behavior
is genuinely robot-specific.
