# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.spawners import (
    spawn_dobot_x1_symm_from_urdf,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import dobot_x1_symm as dobot_mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import ISAACLAB_ASSETS_EXT_DIR

# The vendor URDF package is still named dobot_quad_v2; task-facing names use Dobot X1 Symm.
_DOBOT_X1_ASSET_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "isaaclab_assets", "robots", "dobot_quad_v2")
_DOBOT_X1_JOINT_ORDER = [
    "joint_front_left_abad",
    "joint_front_left_thigh_pitch",
    "joint_front_left_calf_pitch",
    "joint_front_right_abad",
    "joint_front_right_thigh_pitch",
    "joint_front_right_calf_pitch",
    "joint_rear_left_abad",
    "joint_rear_left_thigh_pitch",
    "joint_rear_left_calf_pitch",
    "joint_rear_right_abad",
    "joint_rear_right_thigh_pitch",
    "joint_rear_right_calf_pitch",
]
_DOBOT_X1_DEFAULT_JOINT_POSITIONS = {
    "joint_front_left_abad": 0.1,
    "joint_front_left_thigh_pitch": 0.7,
    "joint_front_left_calf_pitch": -1.5,
    "joint_front_right_abad": -0.1,
    "joint_front_right_thigh_pitch": 0.7,
    "joint_front_right_calf_pitch": -1.5,
    "joint_rear_left_abad": 0.1,
    "joint_rear_left_thigh_pitch": -0.7,
    "joint_rear_left_calf_pitch": 1.5,
    "joint_rear_right_abad": -0.1,
    "joint_rear_right_thigh_pitch": -0.7,
    "joint_rear_right_calf_pitch": 1.5,
}
_DOBOT_X1_FOOT_LINK_ORDER = [
    "link_front_left_foot",
    "link_front_right_foot",
    "link_rear_left_foot",
    "link_rear_right_foot",
]
_DOBOT_X1_CALF_LINK_ORDER = [
    "link_front_left_calf",
    "link_front_right_calf",
    "link_rear_left_calf",
    "link_rear_right_calf",
]
_DOBOT_X1_SYMM_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(160.0, 160.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class PhysicsCfg(PresetCfg):
    """Physics preset for the Dobot X1 symmetric flat task."""

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


@configclass
class DobotX1SymmFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat velocity task for the Dobot X1 symmetric robot."""

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 30.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _DOBOT_X1_SYMM_FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
        self.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0))

        self.scene.robot.spawn = sim_utils.UrdfFileCfg(
            func=spawn_dobot_x1_symm_from_urdf,
            asset_path=os.path.join(_DOBOT_X1_ASSET_DIR, "urdf", "dobot_quad.urdf"),
            usd_dir=os.path.join(_DOBOT_X1_ASSET_DIR, "usd_x1_symm"),
            usd_file_name="dobot_x1_symm.usd",
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=True,
            activate_contact_sensors=True,
            ros_package_paths=[{"name": "dobot_quad_v2", "path": _DOBOT_X1_ASSET_DIR}],
            robot_type="Quadruped",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.4,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        )
        self.scene.contact_forces = None
        self._configure_dobot_x1_symm_contact_sensors()

        self.scene.robot.init_state.joint_pos = _DOBOT_X1_DEFAULT_JOINT_POSITIONS
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.scene.robot.actuators = {
            "hip_legs": IdealPDActuatorCfg(
                joint_names_expr=["joint_.*_abad"],
                effort_limit=17.0,
                velocity_limit=30.0,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
            "thigh_legs": IdealPDActuatorCfg(
                joint_names_expr=["joint_.*_thigh_pitch"],
                effort_limit={
                    "joint_front_.*_thigh_pitch": 17.0,
                    "joint_rear_.*_thigh_pitch": 20.0,
                },
                velocity_limit=30.0,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
            "calf_legs": IdealPDActuatorCfg(
                joint_names_expr=["joint_.*_calf_pitch"],
                effort_limit=37.0,
                velocity_limit=26.0,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
        }
        self.actions.joint_pos.joint_names = _DOBOT_X1_JOINT_ORDER
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25

        self.commands.base_velocity = dobot_mdp.GaitVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            heading_control_stiffness=0.5,
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            min_xy_command_norm=0.2,
            resample_once_after_reset=True,
            resample_gait_once_after_reset=True,
            vel_xy_success_threshold=0.05,
            vel_xy_success_rel_threshold=0.25,
            vel_yaw_success_threshold=0.05,
            vel_yaw_success_rel_threshold=0.25,
            base_height_range=(0.35, 0.55),
            ranges=dobot_mdp.GaitVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
        )

        self._configure_dobot_x1_symm_observations()
        self._configure_dobot_x1_symm_rewards()
        self._configure_dobot_x1_symm_terminations()
        self._configure_dobot_x1_symm_domain_randomization()

    def _configure_dobot_x1_symm_contact_sensors(self) -> None:
        """Configure contact sensors for the nested Dobot URDF link tree."""
        contact_sensor_kwargs = {
            "history_length": 3,
            "track_air_time": True,
        }
        self.scene.contact_trunk = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/link_trunk",
            **contact_sensor_kwargs,
        )
        self.scene.contact_FL_foot = ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_front_left_hip/"
                "link_front_left_thigh/link_front_left_calf/link_front_left_foot"
            ),
            **contact_sensor_kwargs,
        )
        self.scene.contact_FR_foot = ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_front_right_hip/"
                "link_front_right_thigh/link_front_right_calf/link_front_right_foot"
            ),
            **contact_sensor_kwargs,
        )
        self.scene.contact_RL_foot = ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_rear_left_hip/"
                "link_rear_left_thigh/link_rear_left_calf/link_rear_left_foot"
            ),
            **contact_sensor_kwargs,
        )
        self.scene.contact_RR_foot = ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_rear_right_hip/"
                "link_rear_right_thigh/link_rear_right_calf/link_rear_right_foot"
            ),
            **contact_sensor_kwargs,
        )

    def _configure_dobot_x1_symm_observations(self) -> None:
        """Configure the 60D Dobot policy observation to match Go2 ordering."""
        policy = self.observations.policy
        legacy_joint_cfg = SceneEntityCfg("robot", joint_names=_DOBOT_X1_JOINT_ORDER, preserve_order=True)

        policy.base_lin_vel = None
        policy.base_ang_vel = None
        policy.projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        policy.velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        policy.joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": legacy_joint_cfg},
        )
        policy.joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
            params={"asset_cfg": legacy_joint_cfg},
        )
        policy.actions = ObsTerm(func=base_mdp.last_action)
        policy.height_scan = None
        policy.foot_phase_sin = ObsTerm(func=dobot_mdp.foot_phase_sin, params={"command_name": "base_velocity"})
        policy.foot_phase_cos = ObsTerm(func=dobot_mdp.foot_phase_cos, params={"command_name": "base_velocity"})
        policy.foot_theta_sin = ObsTerm(func=dobot_mdp.foot_theta_sin, params={"command_name": "base_velocity"})
        policy.foot_theta_cos = ObsTerm(func=dobot_mdp.foot_theta_cos, params={"command_name": "base_velocity"})
        policy.phase_ratios = ObsTerm(func=dobot_mdp.phase_ratios, params={"command_name": "base_velocity"})

    def _configure_dobot_x1_symm_rewards(self) -> None:
        """Configure Dobot rewards with the Go2 symmetric task layout."""
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = RewTerm(
            func=dobot_mdp.action_rate_exp_penalty,
            weight=0.05,
            params={"scale": 0.1},
        )
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_pos_limits = None

        self.rewards.alive_bonus = RewTerm(func=dobot_mdp.alive_bonus, weight=1.0)
        self.rewards.cmd = RewTerm(
            func=dobot_mdp.command_tracking_penalty,
            weight=0.40,
            params={"command_name": "base_velocity"},
        )
        self.rewards.foot_periodicity = RewTerm(
            func=dobot_mdp.foot_periodicity_penalty,
            weight=0.30,
            params={
                "command_name": "base_velocity",
                "feet_cfg": SceneEntityCfg("robot", body_names=_DOBOT_X1_FOOT_LINK_ORDER, preserve_order=True),
                "foot_sensor_names": ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot"),
                "foot_sensor_body_names": tuple(_DOBOT_X1_FOOT_LINK_ORDER),
            },
        )
        self.rewards.base_height = RewTerm(
            func=dobot_mdp.base_height_range_penalty,
            weight=0.30,
            params={"height_range": (0.35, 0.55)},
        )
        self.rewards.foot_clearance = RewTerm(
            func=dobot_mdp.foot_clearance_penalty,
            weight=0.10,
            params={
                "command_name": "base_velocity",
                "feet_cfg": SceneEntityCfg("robot", body_names=_DOBOT_X1_FOOT_LINK_ORDER, preserve_order=True),
            },
        )
        self.rewards.hip_action_penalty = RewTerm(func=dobot_mdp.hip_action_penalty, weight=0.15)
        self.rewards.morphological_symmetry = RewTerm(
            func=dobot_mdp.morphological_symmetry_penalty,
            weight=0.30,
            params={
                "command_name": "base_velocity",
                "joint_cfg": SceneEntityCfg("robot", joint_names=_DOBOT_X1_JOINT_ORDER, preserve_order=True),
            },
        )
        self.rewards.smoothness = RewTerm(func=dobot_mdp.SmoothnessPenalty, weight=0.10)

    def _configure_dobot_x1_symm_terminations(self) -> None:
        """Configure Dobot reset and termination terms."""
        self.terminations.base_contact = DoneTerm(
            func=dobot_mdp.illegal_contact_any_sensor,
            params={"sensor_names": ("contact_trunk",), "threshold": 1.0},
        )
        self.terminations.base_height = DoneTerm(
            func=dobot_mdp.base_height_out_of_range,
            params={"height_range": (0.15, 0.65)},
        )
        self.terminations.base_orientation = DoneTerm(
            func=dobot_mdp.base_roll_pitch_out_of_range,
            params={"max_roll": 0.8, "max_pitch": 1.0},
        )
        self.terminations.calf_height = DoneTerm(
            func=dobot_mdp.body_height_below,
            params={
                "min_height": 0.08,
                "body_cfg": SceneEntityCfg("robot", body_names=_DOBOT_X1_CALF_LINK_ORDER, preserve_order=True),
            },
        )

    def _configure_dobot_x1_symm_domain_randomization(self) -> None:
        """Configure Dobot domain randomization values."""
        self.events.physics_material.params["static_friction_range"] = (0.3, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 2.0)
        self.events.physics_material.params["make_consistent"] = True

        self.events.add_base_mass.params["asset_cfg"].body_names = "link_trunk"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 1.5)
        self.events.add_base_mass.params["operation"] = "add"
        self.events.add_base_mass.params["distribution"] = "uniform"

        self.events.base_external_force_torque.params["asset_cfg"].body_names = "link_trunk"
        self.events.base_com = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }

        self.events.push_robot.interval_range_s = (15.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}


@configclass
class DobotX1SymmFlatEnvCfg_PLAY(DobotX1SymmFlatEnvCfg):
    """Play configuration for the flat Dobot X1 symmetric task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.sim.physics = PhysxCfg(
            gpu_max_rigid_patch_count=2**15,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**20,
        )
        self.viewer.eye = (0.0, -4.0, 1.5)
        self.viewer.lookat = (0.3, 0.0, 0.45)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
