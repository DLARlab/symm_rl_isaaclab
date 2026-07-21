# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared flat-environment helpers for symmetric quadruped locomotion tasks."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from isaaclab_tasks.utils import PresetCfg

SYMM_QUADRUPED_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(160.0, 160.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)
"""Shared flat terrain generator used by symmetric quadruped tasks."""

SYMM_QUADRUPED_GROUND_COLLISION_PATH = "/World/ground/terrain/mesh"
"""Collision-mesh path used to filter playback ground-reaction forces."""


def _warn_morphological_symmetry_deprecation() -> None:
    """Warn when the deprecated reward configuration name is used."""
    warnings.warn(
        "The 'morphological_symmetry' reward is deprecated; use 'leg_permutation_symmetry'.",
        DeprecationWarning,
        stacklevel=3,
    )


@configclass
class SymmQuadrupedRewardsCfg(RewardsCfg):
    """Reward configuration with a deprecated leg-symmetry name alias."""

    leg_permutation_symmetry: RewTerm | None = None
    """Phase-weighted joint symmetry under configured leg permutations."""

    def __getattr__(self, name: str):
        """Resolve the deprecated reward name without exposing it as a config field."""
        if name == "morphological_symmetry":
            _warn_morphological_symmetry_deprecation()
            return self.leg_permutation_symmetry
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        """Redirect assignments through the deprecated reward name."""
        if name == "morphological_symmetry":
            _warn_morphological_symmetry_deprecation()
            name = "leg_permutation_symmetry"
        super().__setattr__(name, value)


@configclass
class SymmQuadrupedPhysicsCfg(PresetCfg):
    """Physics preset for symmetric quadruped flat locomotion runs."""

    default = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
        gpu_found_lost_pairs_capacity=2**22,
        gpu_found_lost_aggregate_pairs_capacity=2**27,
        gpu_total_aggregate_pairs_capacity=2**22,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=35,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


def configure_flat_scene(env_cfg) -> None:
    """Apply the shared flat terrain and lighting setup."""
    env_cfg.scene.num_envs = 256
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = SYMM_QUADRUPED_FLAT_TERRAIN_CFG
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
    env_cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0))


def make_single_body_contact_sensor(prim_path: str) -> ContactSensorCfg:
    """Create a single-body contact sensor for nested URDF link paths."""
    return ContactSensorCfg(
        prim_path=prim_path,
        history_length=3,
        track_air_time=False,
    )


def configure_play_ground_reaction_force_sensors(env_cfg, sensor_names: Sequence[str]) -> None:
    """Enable ground-filtered normal and friction forces on playback foot sensors."""
    for sensor_name in sensor_names:
        sensor_cfg = getattr(env_cfg.scene, sensor_name)
        sensor_cfg.filter_prim_paths_expr = [SYMM_QUADRUPED_GROUND_COLLISION_PATH]
        sensor_cfg.track_friction_forces = True


def make_gait_velocity_command(
    mdp_module,
    *,
    base_height_range: tuple[float, float] = (0.35, 0.45),
):
    """Create the shared forward-velocity gait command config."""
    return mdp_module.GaitVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        min_xy_command_norm=0.2,
        resample_once_after_reset=True,
        resample_gait_once_after_reset=True,
        vel_xy_success_threshold=0.05,
        vel_xy_success_rel_threshold=0.25,
        vel_yaw_success_threshold=0.05,
        vel_yaw_success_rel_threshold=0.25,
        base_height_range=base_height_range,
        ranges=mdp_module.GaitVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


def configure_policy_observations(env_cfg, mdp_module, joint_names: Sequence[str]) -> None:
    """Configure the shared 72D symmetric quadruped policy observation."""
    policy = env_cfg.observations.policy
    ordered_joint_cfg = SceneEntityCfg("robot", joint_names=list(joint_names), preserve_order=True)

    policy.base_lin_vel = ObsTerm(
        func=base_mdp.base_lin_vel,
        noise=Unoise(n_min=-0.1, n_max=0.1),
        scale=(2.0, 2.0, 2.0),
    )
    policy.base_ang_vel = ObsTerm(
        func=base_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=(0.25, 0.25, 0.25),
    )
    policy.projected_gravity = ObsTerm(
        func=base_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    policy.velocity_commands = ObsTerm(
        func=mdp_module.desired_base_twist,
        params={"command_name": "base_velocity"},
        scale=(2.0, 2.0, 2.0, 0.25, 0.25, 0.25),
    )
    policy.joint_pos = ObsTerm(
        func=base_mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": ordered_joint_cfg},
    )
    policy.joint_vel = ObsTerm(
        func=base_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.05,
        params={"asset_cfg": ordered_joint_cfg},
    )
    policy.actions = ObsTerm(func=base_mdp.last_action)
    policy.height_scan = None
    policy.foot_phase_sin = ObsTerm(func=mdp_module.foot_phase_sin, params={"command_name": "base_velocity"})
    policy.foot_phase_cos = ObsTerm(func=mdp_module.foot_phase_cos, params={"command_name": "base_velocity"})
    policy.foot_theta_sin = ObsTerm(func=mdp_module.foot_theta_sin, params={"command_name": "base_velocity"})
    policy.foot_theta_cos = ObsTerm(func=mdp_module.foot_theta_cos, params={"command_name": "base_velocity"})
    policy.phase_ratios = ObsTerm(func=mdp_module.phase_ratios, params={"command_name": "base_velocity"})
    policy.sagittal_plane_state = ObsTerm(
        func=mdp_module.sagittal_plane_state,
        params={"lateral_position_scale": 0.5},
    )


def configure_rewards(
    env_cfg,
    mdp_module,
    *,
    joint_names: Sequence[str],
    foot_body_names: Sequence[str],
    foot_sensor_names: Sequence[str],
    foot_sensor_body_names: Sequence[str],
    base_height_range: tuple[float, float],
    foot_clearance_height: float = 0.08,
    foot_clearance_height_scale: float = 0.05,
    foot_clearance_mode: str = "phase_penalty",
    foot_clearance_weight: float = 0.10,
    pitch_scale: float = 0.50,
) -> None:
    """Configure the shared symmetric quadruped reward layout."""
    env_cfg.rewards.track_lin_vel_xy_exp = None
    env_cfg.rewards.track_ang_vel_z_exp = None
    env_cfg.rewards.lin_vel_z_l2 = None
    env_cfg.rewards.ang_vel_xy_l2 = None
    env_cfg.rewards.dof_torques_l2 = None
    env_cfg.rewards.dof_acc_l2 = None
    env_cfg.rewards.action_rate_l2 = RewTerm(
        func=mdp_module.action_rate_exp_penalty,
        weight=0.05,
        params={"scale": 0.1},
    )
    env_cfg.rewards.feet_air_time = None
    env_cfg.rewards.flat_orientation_l2 = None
    env_cfg.rewards.dof_pos_limits = None

    feet_cfg = SceneEntityCfg("robot", body_names=list(foot_body_names), preserve_order=True)
    joint_cfg = SceneEntityCfg("robot", joint_names=list(joint_names), preserve_order=True)
    env_cfg.rewards.alive_bonus = RewTerm(func=mdp_module.alive_bonus, weight=0.20)
    env_cfg.rewards.termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
    env_cfg.rewards.cmd = None
    env_cfg.rewards.foot_periodicity = RewTerm(
        func=mdp_module.foot_periodicity_penalty,
        weight=0.30,
        params={
            "command_name": "base_velocity",
            "feet_cfg": feet_cfg,
            "foot_sensor_names": tuple(foot_sensor_names),
            "foot_sensor_body_names": tuple(foot_sensor_body_names),
            "force_scale": 0.005,
        },
    )
    env_cfg.rewards.base_height = RewTerm(
        func=mdp_module.base_height_range_penalty,
        weight=0.30,
        params={"height_range": base_height_range},
    )
    if foot_clearance_mode == "phase_penalty":
        foot_clearance_func = mdp_module.foot_clearance_penalty
        foot_clearance_params = {
            "command_name": "base_velocity",
            "feet_cfg": feet_cfg,
            "min_height": foot_clearance_height,
            "height_scale": foot_clearance_height_scale,
            "min_command_speed": 0.20,
        }
    elif foot_clearance_mode == "current_speed_penalty":
        foot_clearance_func = mdp_module.foot_clearance_current_speed_penalty
        foot_clearance_params = {
            "command_name": "base_velocity",
            "feet_cfg": feet_cfg,
            "min_height": foot_clearance_height,
        }
    elif foot_clearance_mode == "tracking_reward":
        foot_clearance_func = mdp_module.foot_clearance_tracking_reward
        foot_clearance_params = {
            "command_name": "base_velocity",
            "feet_cfg": feet_cfg,
            "foot_sensor_names": tuple(foot_sensor_names),
            "foot_sensor_body_names": tuple(foot_sensor_body_names),
            "target_height": foot_clearance_height,
            "height_scale": foot_clearance_height_scale,
            "excess_height_margin": 0.03,
            "excess_height_scale": foot_clearance_height_scale,
            "min_command_speed": 0.20,
        }
    else:
        raise ValueError(f"Unsupported foot-clearance mode: {foot_clearance_mode!r}.")
    env_cfg.rewards.foot_clearance = RewTerm(
        func=foot_clearance_func,
        weight=foot_clearance_weight,
        params=foot_clearance_params,
    )
    env_cfg.rewards.hip_action_penalty = RewTerm(func=mdp_module.hip_action_penalty, weight=0.15)
    env_cfg.rewards.joint_target_limits = RewTerm(
        func=mdp_module.joint_position_target_limit_penalty,
        weight=0.05,
        params={"action_term_name": "joint_pos", "margin_fraction": 0.05},
    )
    env_cfg.rewards.sagittal_plane = None
    env_cfg.rewards.straight_line_motion = RewTerm(
        func=mdp_module.straight_line_motion_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "forward_velocity_scale": 0.35,
            "lateral_position_scale": 0.35,
            "heading_scale": 0.35,
            "lateral_velocity_scale": 0.20,
            "yaw_rate_scale": 0.20,
            "pose_weight": 0.30,
            "roll_scale": 0.25,
            "pitch_scale": pitch_scale,
            "min_base_height": base_height_range[0],
            "height_scale": 0.10,
            "support_loss_weight": 0.25,
            "forward_weight": 1.0,
            "straight_weight": 0.30,
            "posture_weight": 0.15,
            "lateral_position_deadband": 0.05,
            "heading_deadband": 0.05,
        },
    )
    env_cfg.rewards.leg_permutation_symmetry = RewTerm(
        func=mdp_module.leg_permutation_symmetry_penalty,
        weight=0.30,
        params={"command_name": "base_velocity", "joint_cfg": joint_cfg},
    )
    env_cfg.rewards.smoothness = RewTerm(func=mdp_module.SmoothnessPenalty, weight=0.10)


def configure_terminations(
    env_cfg,
    mdp_module,
    *,
    base_sensor_names: Sequence[str],
    base_height_range: tuple[float, float],
    calf_body_names: Sequence[str],
    additional_contact_terms: dict[str, Sequence[str]] | None = None,
    additional_body_point_terms: dict[str, tuple[tuple[float, float, float], float]] | None = None,
    max_roll: float = 0.8,
    max_pitch: float = 1.2,
) -> None:
    """Configure the shared symmetric quadruped termination layout."""
    env_cfg.terminations.base_contact = DoneTerm(
        func=mdp_module.illegal_contact_any_sensor,
        params={"sensor_names": tuple(base_sensor_names), "threshold": 1.0},
    )
    env_cfg.terminations.base_height = DoneTerm(
        func=mdp_module.base_height_out_of_range,
        params={"height_range": base_height_range},
    )
    env_cfg.terminations.base_orientation = DoneTerm(
        func=mdp_module.base_roll_pitch_out_of_range,
        params={"max_roll": max_roll, "max_pitch": max_pitch},
    )
    for term_name, sensor_names in (additional_contact_terms or {}).items():
        setattr(
            env_cfg.terminations,
            term_name,
            DoneTerm(
                func=mdp_module.illegal_contact_any_sensor,
                params={"sensor_names": tuple(sensor_names), "threshold": 1.0},
            ),
        )
    for term_name, (point_b, min_height) in (additional_body_point_terms or {}).items():
        setattr(
            env_cfg.terminations,
            term_name,
            DoneTerm(
                func=mdp_module.body_local_point_height_below,
                params={"point_b": point_b, "min_height": min_height},
            ),
        )
    env_cfg.terminations.calf_height = DoneTerm(
        func=mdp_module.body_height_below,
        params={
            "min_height": 0.08,
            "body_cfg": SceneEntityCfg("robot", body_names=list(calf_body_names), preserve_order=True),
        },
    )


def configure_domain_randomization(env_cfg, *, base_body_name: str | None = None) -> None:
    """Configure shared symmetric quadruped domain randomization values."""
    env_cfg.events.physics_material.params["static_friction_range"] = (0.3, 2.0)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 2.0)
    env_cfg.events.physics_material.params["make_consistent"] = True

    if base_body_name is not None:
        env_cfg.events.add_base_mass.params["asset_cfg"].body_names = base_body_name
        env_cfg.events.base_external_force_torque.params["asset_cfg"].body_names = base_body_name
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 1.5)
    env_cfg.events.add_base_mass.params["operation"] = "add"
    env_cfg.events.add_base_mass.params["distribution"] = "uniform"

    env_cfg.events.base_com = None
    env_cfg.events.reset_base.params["pose_range"] = {}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (0.0, 0.0),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (0.0, 0.0),
    }

    env_cfg.events.push_robot.interval_range_s = (15.0, 15.0)
    env_cfg.events.push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (0.0, 0.0)}


def make_play_physics_cfg() -> PhysxCfg:
    """Create the reduced-capacity PhysX config used by one-env play tasks."""
    return PhysxCfg(
        gpu_max_rigid_patch_count=2**15,
        gpu_found_lost_pairs_capacity=2**20,
        gpu_found_lost_aggregate_pairs_capacity=2**22,
        gpu_total_aggregate_pairs_capacity=2**20,
    )
