# Go2 Symm Convenience Scripts

These scripts shorten the Isaac Lab commands for the migrated Go2 symmetric RL
task while still using the official Isaac Lab launcher underneath.

## Windows PowerShell

From `D:\go2_symm_rl_lab`:

```powershell
.\scripts\go2_symm\train.bat --iterations 30000 --mirror 0.2
.\scripts\go2_symm\train.bat --no-trs --iterations 10000
.\scripts\go2_symm\play.bat --checkpoint latest
.\scripts\go2_symm\record.bat --checkpoint latest --video-length 400 --gif
.\scripts\go2_symm\ablation.bat --iterations 10000 --seeds 1
```

The `.bat` launchers discover `C:\Users\<you>\.conda\envs\go2_symm_rl_lab`
automatically and set `CONDA_PREFIX` for Isaac Lab. The `.ps1` launchers are
kept for users who allow local PowerShell scripts.

## Ubuntu / Bash

From the Isaac Lab repository root:

```bash
bash scripts/go2_symm/train.sh --iterations 30000 --mirror 0.2
bash scripts/go2_symm/play.sh --checkpoint latest
bash scripts/go2_symm/record.sh --checkpoint latest --video-length 400 --gif
bash scripts/go2_symm/ablation.sh --iterations 10000 --seeds 1
```

The bash ablation wrapper also supports background execution:

```bash
bash scripts/go2_symm/ablation.sh --nohup --iterations 10000
```

## Direct Python Style

You can also call the Python wrappers directly:

```bash
python scripts/go2_symm/train.py --iterations 30000 --mirror 0.2
python scripts/go2_symm/play.py --checkpoint latest
python scripts/go2_symm/record.py --checkpoint latest --gif
python scripts/go2_symm/ablation.py --only both --seeds 1
```

The wrapper detects the platform and calls either `isaaclab.bat` or
`isaaclab.sh`. If the active conda environment is not `go2_symm_rl_lab`, it
uses:

```text
conda run --no-capture-output -n go2_symm_rl_lab ...
```

Use `--dry-run` on any command to print the full Isaac Lab command without
running it.

Training and ablation default to 256 environments for fast iteration. Pass
`--num-envs 2048` when you want the old IsaacGym-scale Go2 setup.

## Notes

- `--mirror`, `--tr-value-coef`, `--tr-warmup-iterations`, and
  `--tr-min-abs-cmd-vel` map to the legacy TRS policy/value/warmup gates.
- `--no-trs` disables symmetry data augmentation, mirror loss, and TRS value
  loss.
- `play.py` automatically resolves the newest checkpoint when
  `--checkpoint latest` is used.
- On Windows, viewer play defaults to the known-good D3D12 Kit args and
  `--rendering_mode balanced`.
- On Ubuntu, no Windows-specific Kit args are added by default.
