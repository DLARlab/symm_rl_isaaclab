# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inference compatibility for archived 56D symmetric quadruped checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

from .observation_history import PolicyObservationHistoryCfg

_LEGACY_TERMS = (
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "foot_phase_sin",
    "foot_phase_cos",
    "foot_theta",
    "phase_ratios",
)


def _read_config(path: Path) -> dict[str, Any]:
    # BaseLoader never executes the Python tags found in archived YAML.
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration mapping in {path}.")
    return config


def _numeric(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numeric(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_numeric(item) for item in value)
    return None if value in (None, "null", "None") else float(value)


def _load_legacy_config(checkpoint: Path) -> dict[str, Any] | None:
    """Recognize the saved 56D layout without changing native 72D evaluation."""
    config_path = checkpoint.parent / "params" / "env.yaml"
    if not config_path.is_file():
        return None
    config = _read_config(config_path)
    policy = config.get("observations", {}).get("policy", {})
    if not isinstance(policy.get("foot_theta"), dict):
        return None
    terms = tuple(name for name, term in policy.items() if isinstance(term, dict) and "func" in term)
    if terms != _LEGACY_TERMS:
        raise ValueError(f"Unsupported archived observation order in {config_path}: {terms}.")
    history = config.get("policy_observation_history")
    if history in (None, "null"):
        history = {}
    length = int(history.get("history_length", 1))
    if length < 1 or history.get("group_name", "policy") != "policy":
        raise ValueError(f"Unsupported policy history in {config_path}: {history}.")
    if policy.get("history_length") not in (None, "null", "0") or any(
        policy[name].get("history_length", "0") != "0" for name in terms
    ):
        raise ValueError("Archived 56D policies must use complete-frame history without per-term history.")
    config["_history_length"] = length
    return config


def _legacy_profile(checkpoint: Path) -> dict[str, Any] | None:
    """Describe the compatibility profile for evaluation provenance."""
    config = _load_legacy_config(checkpoint)
    if config is None:
        return None
    return {
        "name": "legacy_56d",
        "frame_dim": 56,
        "history_length": config["_history_length"],
        "input_dim": 56 * config["_history_length"],
        "history_order": "oldest_first_zero_padded",
        "robot_settings": "saved_env_yaml",
    }


def _foot_theta(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return archived raw foot offsets in [-0.25, 0.75) cycles."""
    offsets = env.command_manager.get_term(command_name).foot_thetas
    return torch.remainder(offsets + 0.25, 1.0) - 0.25


def _configure_observations(env_cfg: ManagerBasedRLEnvCfg, saved: dict[str, Any]) -> None:
    """Rebuild the group so offsets precede phase ratios exactly as in training."""
    previous = env_cfg.observations.policy
    policy = ObservationGroupCfg(concatenate_terms=True, enable_corruption=previous.enable_corruption)
    functions = {
        "projected_gravity": mdp.projected_gravity,
        "velocity_commands": mdp.generated_commands,
        "joint_pos": mdp.joint_pos_rel,
        "joint_vel": mdp.joint_vel_rel,
        "actions": mdp.last_action,
        "foot_phase_sin": symm_quadruped.foot_phase_sin,
        "foot_phase_cos": symm_quadruped.foot_phase_cos,
        "foot_theta": _foot_theta,
        "phase_ratios": symm_quadruped.phase_ratios,
    }
    for name in _LEGACY_TERMS:
        archived = saved["observations"]["policy"][name]
        if name == "foot_theta":
            term = ObservationTermCfg(func=_foot_theta, params={"command_name": "base_velocity"})
        else:
            term = copy.deepcopy(getattr(previous, name))
        term.func = functions[name]
        term.scale = _numeric(archived.get("scale"))
        term.clip = _numeric(archived.get("clip"))
        if name in ("joint_pos", "joint_vel"):
            joint_names = archived["params"]["asset_cfg"]["joint_names"]
            if len(joint_names) != 12:
                raise ValueError("A legacy policy must specify its 12 observation joints in order.")
            term.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True)}
        elif name == "velocity_commands":
            term.params = {"command_name": "base_velocity"}
        setattr(policy, name, term)
    env_cfg.observations.policy = policy
    env_cfg.policy_observation_history = PolicyObservationHistoryCfg(history_length=saved["_history_length"])


def _configure_robot(env_cfg: ManagerBasedRLEnvCfg, saved: dict[str, Any]) -> None:
    """Restore archived actuator, action, and deterministic joint-limit settings."""
    robot = saved["scene"]["robot"]
    for name, actuator in robot["actuators"].items():
        if name not in env_cfg.scene.robot.actuators:
            raise ValueError(f"Archived actuator {name!r} does not exist in the selected robot task.")
        for field in ("stiffness", "damping", "effort_limit", "velocity_limit", "friction"):
            if field in actuator:
                setattr(env_cfg.scene.robot.actuators[name], field, _numeric(actuator[field]))
    env_cfg.scene.robot.init_state.joint_pos = _numeric(robot["init_state"]["joint_pos"])
    action = saved["actions"]["joint_pos"]
    env_cfg.actions.joint_pos.joint_names = list(action["joint_names"])
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.actions.joint_pos.scale = _numeric(action["scale"])
    env_cfg.actions.joint_pos.offset = _numeric(action["offset"])
    env_cfg.actions.joint_pos.use_default_offset = action["use_default_offset"] == "true"
    env_cfg.actions.joint_pos.clip = _numeric(action["clip"])
    for name in ("front_calf_joint_limits", "rear_calf_joint_limits"):
        archived = saved.get("events", {}).get(name)
        if not isinstance(archived, dict):
            continue
        params = archived["params"]
        if archived["func"] != "isaaclab.envs.mdp.events:randomize_joint_parameters" or archived["mode"] != "startup":
            raise ValueError(f"Unsupported archived joint-limit event {name!r}.")
        setattr(
            env_cfg.events,
            name,
            EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=params["asset_cfg"]["joint_names"]),
                    "lower_limit_distribution_params": _numeric(params["lower_limit_distribution_params"]),
                    "upper_limit_distribution_params": _numeric(params["upper_limit_distribution_params"]),
                    "operation": params["operation"],
                    "distribution": params["distribution"],
                },
            ),
        )


def _configure_checkpoint_compatibility(
    env_cfg: ManagerBasedRLEnvCfg, agent_cfg, checkpoint: Path
) -> dict[str, Any] | None:
    """Configure inference from archived metadata before constructing the environment."""
    command = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if not isinstance(command, symm_quadruped.GaitVelocityCommandCfg):
        return None
    saved = _load_legacy_config(checkpoint)
    if saved is None:
        return None
    profile = _legacy_profile(checkpoint)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for network in ("actor", "critic"):
        state = weights.get(f"{network}_state_dict")
        first_layer = (
            state.get("mlp.0.weight")
            if state is not None
            else weights.get("model_state_dict", {}).get(f"{network}.0.weight")
        )
        if first_layer is None or first_layer.ndim != 2 or first_layer.shape[1] != profile["input_dim"]:
            raise ValueError(f"Checkpoint {network} does not match the saved {profile['input_dim']}D policy input.")
    saved_dt = float(saved["sim"]["dt"]) * int(saved["decimation"])
    if abs(saved_dt - env_cfg.sim.dt * env_cfg.decimation) > 1.0e-10:
        raise ValueError("The archived policy control timestep differs from the evaluation timestep.")
    archived_agent = _read_config(checkpoint.parent / "params" / "agent.yaml")
    for network in ("actor", "critic"):
        cfg = getattr(agent_cfg, network)
        source = archived_agent[network]
        if source["class_name"] != "MLPModel":
            raise ValueError("Legacy checkpoint compatibility supports MLPModel actor and critic networks.")
        cfg.hidden_dims = [int(width) for width in source["hidden_dims"]]
        cfg.activation = source["activation"]
        cfg.obs_normalization = source["obs_normalization"] == "true"
    distribution = archived_agent["actor"]["distribution_cfg"]
    agent_cfg.actor.distribution_cfg.std_type = distribution["std_type"]
    agent_cfg.clip_actions = _numeric(archived_agent["clip_actions"])
    # The 72D training transforms do not apply to legacy inference. Evaluation performs no PPO updates.
    agent_cfg.algorithm.symmetry_cfg = None
    _configure_observations(env_cfg, saved)
    _configure_robot(env_cfg, saved)
    return profile
