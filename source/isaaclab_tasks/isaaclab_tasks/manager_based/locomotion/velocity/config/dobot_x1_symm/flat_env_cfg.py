# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.spawners import (
    spawn_dobot_x1_symm_from_urdf,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.flat_env_cfg import (
    SymmQuadrupedPhysicsCfg,
    configure_domain_randomization,
    configure_flat_scene,
    configure_policy_observations,
    configure_rewards,
    configure_terminations,
    make_gait_velocity_command,
    make_play_physics_cfg,
    make_single_body_contact_sensor,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import dobot_x1_symm as dobot_mdp

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
_DOBOT_X1_FOOT_SENSOR_NAMES = ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot")

PhysicsCfg = SymmQuadrupedPhysicsCfg


@configclass
class DobotX1SymmFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat velocity task for the Dobot X1 symmetric robot."""

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 30.0
        configure_flat_scene(self)

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

        self.commands.base_velocity = make_gait_velocity_command(dobot_mdp, base_height_range=(0.35, 0.55))

        self._configure_dobot_x1_symm_observations()
        self._configure_dobot_x1_symm_rewards()
        self._configure_dobot_x1_symm_terminations()
        self._configure_dobot_x1_symm_domain_randomization()

    def _configure_dobot_x1_symm_contact_sensors(self) -> None:
        """Configure contact sensors for the nested Dobot URDF link tree."""
        self.scene.contact_trunk = make_single_body_contact_sensor("{ENV_REGEX_NS}/Robot/Geometry/link_trunk")
        self.scene.contact_FL_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_front_left_hip/"
            "link_front_left_thigh/link_front_left_calf/link_front_left_foot"
        )
        self.scene.contact_FR_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_front_right_hip/"
            "link_front_right_thigh/link_front_right_calf/link_front_right_foot"
        )
        self.scene.contact_RL_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_rear_left_hip/"
            "link_rear_left_thigh/link_rear_left_calf/link_rear_left_foot"
        )
        self.scene.contact_RR_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/link_trunk/link_rear_right_hip/"
            "link_rear_right_thigh/link_rear_right_calf/link_rear_right_foot"
        )

    def _configure_dobot_x1_symm_observations(self) -> None:
        """Configure the 60D Dobot policy observation to match Go2 ordering."""
        configure_policy_observations(self, dobot_mdp, _DOBOT_X1_JOINT_ORDER)

    def _configure_dobot_x1_symm_rewards(self) -> None:
        """Configure Dobot rewards with the Go2 symmetric task layout."""
        configure_rewards(
            self,
            dobot_mdp,
            joint_names=_DOBOT_X1_JOINT_ORDER,
            foot_body_names=_DOBOT_X1_FOOT_LINK_ORDER,
            foot_sensor_names=_DOBOT_X1_FOOT_SENSOR_NAMES,
            foot_sensor_body_names=_DOBOT_X1_FOOT_LINK_ORDER,
            base_height_range=(0.35, 0.55),
        )

    def _configure_dobot_x1_symm_terminations(self) -> None:
        """Configure Dobot reset and termination terms."""
        configure_terminations(
            self,
            dobot_mdp,
            base_sensor_names=("contact_trunk",),
            base_height_range=(0.15, 0.65),
            calf_body_names=_DOBOT_X1_CALF_LINK_ORDER,
        )

    def _configure_dobot_x1_symm_domain_randomization(self) -> None:
        """Configure Dobot domain randomization values."""
        configure_domain_randomization(self, base_body_name="link_trunk")


@configclass
class DobotX1SymmFlatEnvCfg_PLAY(DobotX1SymmFlatEnvCfg):
    """Play configuration for the flat Dobot X1 symmetric task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.sim.physics = make_play_physics_cfg()
        self.viewer.eye = (0.0, -4.0, 1.5)
        self.viewer.lookat = (0.3, 0.0, 0.45)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
