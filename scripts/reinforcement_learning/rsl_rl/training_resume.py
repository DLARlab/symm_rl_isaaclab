# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint command curricula and continue RSL-RL logging without modifying RSL-RL."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger


def _command_terms(env) -> dict:
    manager = getattr(env.unwrapped, "command_manager", None)
    if manager is None:
        return {}
    return {name: manager.get_term(name) for name in manager.active_terms}


def _logging_metadata(log_dir: str | Path) -> dict:
    path = Path(log_dir) / "training_resume.json"
    return json.loads(path.read_text()) if path.is_file() else {}


class _LoggerResume(Logger):
    """Keep cumulative counters and optionally reuse a TensorBoard or W&B run."""

    def __init__(self, *args, resume_logging: bool = False, wandb_run_id: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resume_logging = resume_logging
        self.wandb_run_id = wandb_run_id
        self.resume_state: dict = {}
        self.next_iteration = 0
        self.prior_time_s = 0.0
        self.curriculum_restarted = False
        self.resume_segment_dir = (
            str(Path(self.log_dir) / "resume_segments" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
            if resume_logging and self.log_dir is not None
            else None
        )

    def _store_code_state(self) -> list[str]:
        if self.resume_segment_dir is None:
            return super()._store_code_state()
        original_dir = self.log_dir
        try:
            self.log_dir = self.resume_segment_dir
            return super()._store_code_state()
        finally:
            self.log_dir = original_dir

    def log(self, *args, **kwargs) -> None:
        # RSL-RL estimates ETA from total accumulated time divided by completed iterations.
        if self.prior_time_s > 0.0:
            kwargs["start_it"] = 0
        super().log(*args, **kwargs)

    def init_logging_writer(self) -> None:
        logger_type = self.cfg.get("logger", "tensorboard").lower()
        if self.resume_logging and self.resume_state.get("logger", logger_type) != logger_type:
            raise ValueError("The logging backend differs from the original run; use a new logging run instead.")
        if not self.resume_logging or self.disable_logs:
            super().init_logging_writer()
        else:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "wandb":
                self.writer = self._wandb_writer()
            elif self.logger_type == "tensorboard":
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                from torch.utils.tensorboard import SummaryWriter

                events = EventAccumulator(self.log_dir, size_guidance={"scalars": 0}).Reload()
                if "Perf/total_fps" in events.Tags()["scalars"]:
                    last_step = max(event.step for event in events.Scalars("Perf/total_fps"))
                    if last_step >= self.next_iteration:
                        raise ValueError("Logging extends past this checkpoint; resume into a new logging run instead.")
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("--resume_logging supports TensorBoard and W&B loggers only.")

            files = self._store_code_state()
            if self.logger_type == "wandb":
                self.writer.store_config(self.env_cfg, self.cfg)
                for path in files:
                    import wandb

                    wandb.save(path, base_path=self.log_dir)

        if self.writer is not None and self.log_dir is not None:
            if self.curriculum_restarted:
                self.writer.add_scalar("Resume/curriculum_restarted", 1.0, self.next_iteration)
            metadata = {"logger": self.logger_type, "curriculum_restarted": self.curriculum_restarted}
            if self.logger_type == "wandb":
                import wandb

                metadata["wandb"] = {
                    "id": wandb.run.id,
                    "project": wandb.run.project,
                    "entity": wandb.run.entity,
                }
            self.resume_state = metadata
            path = Path(self.log_dir) / "training_resume.json"
            path.write_text(json.dumps(metadata, indent=2) + "\n")

    def _wandb_writer(self):
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter
        from torch.utils.tensorboard import SummaryWriter

        import wandb

        metadata = self.resume_state.get("wandb", {})
        run_id = self.wandb_run_id or metadata.get("id")
        if not run_id:
            raise ValueError("No saved W&B run ID; supply --wandb_run_id when continuing an older W&B run.")
        if self.wandb_run_id and metadata.get("id") not in (None, self.wandb_run_id):
            raise ValueError("--wandb_run_id does not match the checkpoint's W&B run.")
        project = self.cfg["wandb_project"]
        entity = os.environ.get("WANDB_USERNAME") or metadata.get("entity")
        if metadata.get("project", project) != project or metadata.get("entity", entity) != entity:
            raise ValueError("W&B project/entity differ from the original run; use a new logging run instead.")
        next_iteration = self.next_iteration

        class _Writer(WandbSummaryWriter):
            """Reuse the existing writer's scalar and checkpoint methods with explicit W&B resumption."""

            def __init__(self, log_dir: str, cfg: dict) -> None:
                run = wandb.init(
                    project=project,
                    entity=entity,
                    id=run_id,
                    resume="must",
                    allow_val_change=True,
                )
                if run.step > next_iteration:
                    run.finish()
                    raise ValueError("W&B history extends past this checkpoint; use a new logging run instead.")
                SummaryWriter.__init__(self, log_dir=log_dir, flush_secs=10)
                self.logged_videos = set()

            def store_config(self, env_cfg, train_cfg: dict) -> None:
                config = env_cfg if isinstance(env_cfg, dict) else env_cfg.to_dict()
                wandb.config.update({"env_cfg": config, "train_cfg": train_cfg}, allow_val_change=True)

        return _Writer(self.log_dir, self.cfg)


class _OnPolicyRunnerResume(OnPolicyRunner):
    """Persist curriculum weights, cumulative logging counters, and the next update index."""

    _STATE_KEY = "training_resume"

    def __init__(self, *args, resume_logging: bool = False, wandb_run_id: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger = _LoggerResume(
            log_dir=self.logger.log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            resume_logging=resume_logging,
            wandb_run_id=wandb_run_id,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        curricula = {}
        for name, term in _command_terms(self.env).items():
            if hasattr(term, "curriculum_state_dict"):
                state = term.curriculum_state_dict()
                if state is not None:
                    curricula[name] = state
        state = {
            "version": 1,
            "next_iteration": self.current_learning_iteration + 1,
            "curricula": curricula,
            "total_timesteps": self.logger.tot_timesteps,
            "total_time_s": self.logger.tot_time,
            "logging": self.logger.resume_state,
        }
        infos = dict(infos or {})
        infos[self._STATE_KEY] = state
        super().save(path, infos=infos)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        if load_cfg is not None and not load_cfg.get("iteration", False):
            return infos
        state = (infos or {}).get(self._STATE_KEY)
        terms = _command_terms(self.env)
        if state is None:
            if any(
                hasattr(term, "curriculum_state_dict") and term.curriculum_state_dict() is not None
                for term in terms.values()
            ):
                self.logger.curriculum_restarted = True
                warnings.warn(
                    "This checkpoint has no curriculum state; command curriculum and cumulative logging counters"
                    " restart from their configured initial values.",
                    UserWarning,
                    stacklevel=2,
                )
            self.current_learning_iteration += 1
        else:
            if state.get("version") != 1:
                raise ValueError("Unsupported training resume checkpoint version.")
            next_iteration = int(state["next_iteration"])
            if next_iteration != self.current_learning_iteration + 1:
                raise ValueError("Checkpoint iteration and training resume state are inconsistent.")
            curricula = state["curricula"]
            for name, curriculum in curricula.items():
                if name not in terms or not hasattr(terms[name], "curriculum_load_state_dict"):
                    raise ValueError(f"Checkpoint curriculum command {name!r} is absent in the current environment.")
                terms[name].curriculum_load_state_dict(curriculum)
            for name, term in terms.items():
                if hasattr(term, "curriculum_state_dict") and term.curriculum_state_dict() is not None:
                    if name not in curricula:
                        self.logger.curriculum_restarted = True
                        warnings.warn(f"No saved curriculum for {name!r}; using its initial weights.", stacklevel=2)
            self.current_learning_iteration = next_iteration
            self.logger.tot_timesteps = int(state["total_timesteps"])
            self.logger.tot_time = float(state["total_time_s"])
            self.logger.prior_time_s = self.logger.tot_time
            if self.logger.resume_logging:
                self.logger.resume_state = state.get("logging", {})
            if curricula:
                # Initial commands were sampled before loading. Start fresh episodes with the restored sampler.
                self.env.reset()
        if self.logger.resume_logging and not self.logger.resume_state:
            self.logger.resume_state = _logging_metadata(Path(path).parent)
        self.logger.next_iteration = self.current_learning_iteration
        if hasattr(self.alg, "current_learning_iteration"):
            self.alg.current_learning_iteration = self.current_learning_iteration
        if hasattr(self.alg, "_time_reversal_update_count"):
            self.alg._time_reversal_update_count = self.current_learning_iteration
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        return infos
