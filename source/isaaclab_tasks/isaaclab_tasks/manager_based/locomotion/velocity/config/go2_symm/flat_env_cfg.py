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

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.spawners import spawn_go2_symm_from_urdf
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.flat_env_cfg import (
    SymmQuadrupedPhysicsCfg,
    SymmQuadrupedRewardsCfg,
    configure_domain_randomization,
    configure_flat_scene,
    configure_play_ground_reaction_force_sensors,
    configure_policy_observations,
    configure_rewards,
    configure_terminations,
    make_gait_velocity_command,
    make_play_physics_cfg,
    make_single_body_contact_sensor,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import go2_symm as go2_symm_mdp

from isaaclab_assets import ISAACLAB_ASSETS_EXT_DIR

_GO2_SYMM_ASSET_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "isaaclab_assets", "robots", "go2_symm")
_GO2_LEGGED_GYM_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
_GO2_FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
_GO2_CALF_BODY_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
_GO2_FOOT_SENSOR_NAMES = ("contact_FL_foot", "contact_FR_foot", "contact_RL_foot", "contact_RR_foot")

PhysicsCfg = SymmQuadrupedPhysicsCfg


@configclass
class UnitreeGo2SymmFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat Go2 velocity task used as the first Isaac Lab migration milestone."""

    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    rewards: SymmQuadrupedRewardsCfg = SymmQuadrupedRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 30.0
        configure_flat_scene(self)

        self.scene.robot.spawn = sim_utils.UrdfFileCfg(
            func=spawn_go2_symm_from_urdf,
            asset_path=os.path.join(_GO2_SYMM_ASSET_DIR, "urdf", "go2_calf_collapsed.urdf"),
            usd_dir=os.path.join(_GO2_SYMM_ASSET_DIR, "usd_calf_collapsed_damped"),
            usd_file_name="go2_symm_calf_collapsed_damped.usd",
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=True,
            activate_contact_sensors=True,
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
        self._configure_go2_symm_contact_sensors()

        self.scene.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.scene.robot.actuators = {
            "hip_legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_hip_joint"],
                effort_limit=23.7,
                velocity_limit=30.1,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
            "thigh_legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_thigh_joint"],
                effort_limit=23.7,
                velocity_limit=30.1,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
            "calf_legs": IdealPDActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=35.55,
                velocity_limit=20.07,
                stiffness=30.0,
                damping=0.65,
                friction=0.0,
            ),
        }
        self.actions.joint_pos.joint_names = _GO2_LEGGED_GYM_JOINT_ORDER
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25

        self.commands.base_velocity = make_gait_velocity_command(go2_symm_mdp, base_height_range=(0.35, 0.45))

        self._configure_go2_symm_observations()
        self._configure_go2_symm_rewards()
        self._configure_go2_symm_terminations()
        self._configure_go2_symm_domain_randomization()

    def _configure_go2_symm_contact_sensors(self) -> None:
        """Configure contact sensors for the nested URDF link tree."""
        self.scene.contact_base = make_single_body_contact_sensor("{ENV_REGEX_NS}/Robot/Geometry/base")
        self.scene.contact_head_upper = make_single_body_contact_sensor("{ENV_REGEX_NS}/Robot/Geometry/base/Head_upper")
        self.scene.contact_head_lower = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/base/Head_upper/Head_lower"
        )
        self.scene.contact_FL_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/base/FL_hip/FL_thigh/FL_calf/FL_foot"
        )
        self.scene.contact_FR_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/base/FR_hip/FR_thigh/FR_calf/FR_foot"
        )
        self.scene.contact_RL_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/base/RL_hip/RL_thigh/RL_calf/RL_foot"
        )
        self.scene.contact_RR_foot = make_single_body_contact_sensor(
            "{ENV_REGEX_NS}/Robot/Geometry/base/RR_hip/RR_thigh/RR_calf/RR_foot"
        )

    def _configure_go2_symm_observations(self) -> None:
        """Configure the shared 72D Go2 policy observation."""
        configure_policy_observations(self, go2_symm_mdp, _GO2_LEGGED_GYM_JOINT_ORDER)

    def _configure_go2_symm_rewards(self) -> None:
        """Configure the first migrated Go2 reward terms."""
        configure_rewards(
            self,
            go2_symm_mdp,
            joint_names=_GO2_LEGGED_GYM_JOINT_ORDER,
            foot_body_names=_GO2_FOOT_BODY_NAMES,
            foot_sensor_names=_GO2_FOOT_SENSOR_NAMES,
            foot_sensor_body_names=_GO2_FOOT_BODY_NAMES,
            base_height_range=(0.35, 0.45),
            foot_clearance_height=0.08,
            foot_clearance_height_scale=0.03,
            foot_clearance_mode="tracking_reward",
            foot_clearance_weight=0.15,
        )

    def _configure_go2_symm_terminations(self) -> None:
        """Configure migrated Go2 reset and termination terms."""
        configure_terminations(
            self,
            go2_symm_mdp,
            base_sensor_names=("contact_base",),
            base_height_range=(0.15, 0.45),
            calf_body_names=_GO2_CALF_BODY_NAMES,
            additional_contact_terms={"head_contact": ("contact_head_upper", "contact_head_lower")},
        )

    def _configure_go2_symm_domain_randomization(self) -> None:
        """Configure migrated Go2 domain randomization values."""
        configure_domain_randomization(self)


@configclass
class UnitreeGo2SymmFlatEnvCfg_PLAY(UnitreeGo2SymmFlatEnvCfg):
    """Play configuration for the flat Go2 symmetric migration task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.gait_library_version = go2_symm_mdp.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_VERSION
        self.commands.base_velocity.init_foot_thetas = go2_symm_mdp.SYMM_QUADRUPED_GAIT_LIBRARY_PLAY_ROWS
        self.commands.base_velocity.init_foot_theta_weights = None
        self.commands.base_velocity.resample_once_after_reset = True
        self.commands.base_velocity.resample_gait_once_after_reset = True
        self.commands.base_velocity.gait_sequence_enabled = True
        self.commands.base_velocity.gait_sequence_duration_s = 5.0
        self.commands.base_velocity.add_noise_period = True
        self.commands.base_velocity.add_noise_theta = False
        self.episode_length_s = (
            len(self.commands.base_velocity.init_foot_thetas) * self.commands.base_velocity.gait_sequence_duration_s
            + self.decimation * self.sim.dt
        )
        self.sim.physics = make_play_physics_cfg()
        configure_play_ground_reaction_force_sensors(self, _GO2_FOOT_SENSOR_NAMES)
        self.viewer.eye = (0.0, -4.0, 1.4)
        self.viewer.lookat = (0.3, 0.0, 0.35)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
