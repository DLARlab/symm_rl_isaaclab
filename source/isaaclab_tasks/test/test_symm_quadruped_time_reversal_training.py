# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for independent TR consistency and trajectory augmentation."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.agents.rsl_rl_ppo_cfg import (
    DobotX1SymmFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.dobot_x1_symm.flat_env_cfg import (
    DobotX1SymmFlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2SymmFlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_symm.flat_env_cfg import (
    UnitreeGo2SymmFlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_augmentation import (
    TimeReversalAugmentation,
    TRAugmentationPool,
    TRDynamicsState,
    TRReplayCandidatePool,
    TRSidecarBuffer,
    _capture_contact_diagnostics,
    assert_time_reverse_action_transform,
    build_reversed_sequence_segment,
    quaternion_to_rotation_6d,
    time_reversal_augmentation_nll,
    time_reversal_dynamics_residual,
    time_reversal_filter_mask,
    time_reversal_phase_event_crossing,
    time_reverse_dynamics_state,
    validate_filter_with_one_step_replay,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_cfg import (
    TimeReversalAugmentationCfg,
    TimeReversalScheduleCfg,
    TimeReversalSymmetryCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_core import (
    resolve_time_reversal_schedule,
    time_reversal_gradient_diagnostics,
    time_reversal_mask_diagnostics,
    time_reversal_schedule_scale,
    time_reversal_validity_mask,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo import (
    TimeReversalPPO,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped import (
    SYMM_QUADRUPED_POLICY_OBS_LAYOUT,
)

from scripts.symm_locomotion.training_provenance import _config_dict, sha256_value


def _dynamics_state(
    batch_size: int = 1,
    *,
    phase: float = 0.2,
    command: float = 1.0,
    previous_action: float = 0.0,
) -> TRDynamicsState:
    zeros3 = torch.zeros(batch_size, 3)
    zeros12 = torch.zeros(batch_size, 12)
    identity_rotation = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0)).repeat(batch_size, 1)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0)).repeat(batch_size, 1)
    velocity_command = torch.zeros(batch_size, 3)
    velocity_command[:, 0] = command
    return TRDynamicsState(
        root_position=zeros3.clone(),
        root_rotation_6d=identity_rotation,
        root_linear_velocity=zeros3.clone(),
        root_angular_velocity=zeros3.clone(),
        joint_position=zeros12.clone(),
        joint_velocity=zeros12.clone(),
        actuator_target=zeros12.clone(),
        previous_action=torch.full((batch_size, 12), previous_action),
        velocity_command=velocity_command,
        common_gait_phase=torch.full((batch_size,), phase),
        foot_phase_offsets=offsets,
        swing_ratio=torch.full((batch_size,), 0.55),
        stance_ratio=torch.full((batch_size,), 0.45),
        gait_period=torch.full((batch_size,), 0.4),
        gait_row=torch.zeros(batch_size, dtype=torch.long),
        body_mass=torch.ones(batch_size, 2),
        material_properties=torch.tensor((0.8, 0.6, 0.0)).reshape(1, 1, 3).repeat(batch_size, 2, 1),
        additional_actuator_state=torch.empty(batch_size, 0),
    )


class _FakeScene(dict):
    def __init__(self, robot, num_envs):
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(num_envs, 3)
        self.sensors = {}


class _FakeGaitCommand:
    def __init__(self, num_envs):
        self.command = torch.tensor((1.0, 0.0, 0.0)).repeat(num_envs, 1)
        self.foot_thetas = torch.tensor((0.0, 0.5, 0.5, 0.0)).repeat(num_envs, 1)
        self.duty_factors = torch.full((num_envs,), 0.45)
        self.gait_periods = torch.full((num_envs,), 0.4)
        self.gait_row_indices = torch.zeros(num_envs, dtype=torch.long)
        self.phase = torch.full((num_envs,), 0.2)

    def common_gait_phases(self):
        return self.phase


class _FakeManager:
    def __init__(self, name, term, *, action=None):
        self.active_terms = [name]
        self._term = term
        self.action = action
        if action is not None:
            self.total_action_dim = action.shape[-1]

    def get_term(self, _name):
        return self._term


class _FakeRootView:
    def __init__(self, num_envs):
        self.material_properties = torch.tensor((0.8, 0.6, 0.0)).reshape(1, 1, 3).repeat(num_envs, 2, 1)

    def get_material_properties(self):
        return self.material_properties


def _augmentation_wrapper(num_envs=4, *, include_material_properties=True):
    data = SimpleNamespace(
        root_pos_w=torch.zeros(num_envs, 3),
        root_quat_w=torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(num_envs, 1),
        root_lin_vel_b=torch.zeros(num_envs, 3),
        root_ang_vel_b=torch.zeros(num_envs, 3),
        joint_pos=torch.zeros(num_envs, 12),
        default_joint_pos=torch.zeros(num_envs, 12),
        joint_vel=torch.zeros(num_envs, 12),
        joint_pos_target=torch.zeros(num_envs, 12),
        body_mass=torch.ones(num_envs, 13),
    )
    robot = SimpleNamespace(data=data)
    if include_material_properties:
        robot.root_view = _FakeRootView(num_envs)
    action_term = SimpleNamespace(
        _joint_ids=slice(None),
        _scale=0.5,
        _offset=torch.full((num_envs, 12), 0.1),
        cfg=SimpleNamespace(clip=None),
        processed_actions=torch.zeros(num_envs, 12),
    )
    gait_command = _FakeGaitCommand(num_envs)
    action_manager = _FakeManager("joint_pos", action_term, action=torch.zeros(num_envs, 12))
    command_manager = _FakeManager("base_velocity", gait_command)
    environment = SimpleNamespace(
        num_envs=num_envs,
        step_dt=0.02,
        physics_dt=0.005,
        device="cpu",
        scene=_FakeScene(robot, num_envs),
        action_manager=action_manager,
        command_manager=command_manager,
    )
    return SimpleNamespace(unwrapped=environment), gait_command


class _NllActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(12))

    def forward(self, observations, *, stochastic_output=False, **_kwargs):
        self._current_mean = self.mean.expand(observations.batch_size[0], -1)
        if stochastic_output:
            return self._current_mean + torch.randn_like(self._current_mean)
        return self._current_mean

    def get_output_log_prob(self, actions):
        return -(actions - self._current_mean).square().sum(dim=-1)


_PUBLICATION_SYMMETRY_FIELDS = (
    "use_tr_policy_consistency",
    "use_tr_value_consistency",
    "log_disabled_raw_consistency",
    "tr_policy_schedule",
    "tr_value_schedule",
    "tr_validity",
    "tr_gradient_diagnostics",
    "tr_augmentation",
)

_SATURATION_SYMMETRY_FIELDS = (
    "tr_policy_output_space",
    "actor_mean_bound_mode",
    "actor_mean_feasible_margin_fraction",
)


def _legacy_environment_projection(config) -> dict:
    """Remove additive saturation/profile fields before hashing legacy defaults."""
    resolved = _config_dict(config)
    resolved["rewards"]["foot_phase"]["params"].pop("reduction")
    resolved["rewards"]["joint_target_limits"]["params"].pop("mode")
    resolved["commands"]["base_velocity"].pop("gait_sampling_profile")
    resolved["commands"]["base_velocity"].pop("gait_curriculum_iterations")
    return resolved


@pytest.mark.parametrize(
    ("config_type", "expected_sha256"),
    [
        (UnitreeGo2SymmFlatEnvCfg, "2610e4aa014407b7900e9047ac49a1c15cd05ef585d576c32a1002b08b029574"),
        (DobotX1SymmFlatEnvCfg, "74b1094cc8701cb7beed8a2078936a62f7c3ae4f14d5ed9d32c584ae0b1c9509"),
    ],
)
def test_default_environment_config_hash_is_unchanged(config_type, expected_sha256):
    assert sha256_value(_legacy_environment_projection(config_type())) == expected_sha256


@pytest.mark.parametrize(
    ("config_type", "expected_sha256"),
    [
        (UnitreeGo2SymmFlatPPORunnerCfg, "a688f79ad55fcb56c917d06950ff09fa3d35689eab96fc863e29d5673ebb97b4"),
        (DobotX1SymmFlatPPORunnerCfg, "ee8ce1c951aade23e5e1e481d7ae62e2eb617db94be74f075d14e2c6f4cdf91b"),
    ],
)
def test_default_agent_legacy_projection_hash_is_unchanged(config_type, expected_sha256):
    resolved = _config_dict(config_type())
    symmetry = resolved["algorithm"]["symmetry_cfg"]
    for field in (*_PUBLICATION_SYMMETRY_FIELDS, *_SATURATION_SYMMETRY_FIELDS):
        symmetry.pop(field)
    # Normalize the one deliberately changed field so the frozen projection
    # continues to guard every other historical runner default.
    symmetry["value_loss_coeff"] = 0.05
    assert sha256_value(resolved) == expected_sha256


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (1, 0.0),
        (2, 0.0),
        (3, 0.5),
        (4, 1.0),
        (5, 1.0),
        (6, 1.0),
        (7, 0.75),
        (8, 0.5),
        (9, 0.5),
    ],
)
def test_full_schedule_boundaries(iteration, expected):
    scale = time_reversal_schedule_scale(iteration, 2, 2, 2, 2, 0.5, "linear")
    assert scale == pytest.approx(expected)


@pytest.mark.parametrize("shape", ["linear", "half_cosine"])
def test_zero_duration_schedule_phases_are_exact(shape):
    assert time_reversal_schedule_scale(4, 5, 0, 0, 0, 0.25, shape) == 0.0
    assert time_reversal_schedule_scale(5, 5, 0, 0, 0, 0.25, shape) == 1.0
    assert time_reversal_schedule_scale(6, 5, 0, 0, 0, 0.25, shape) == 0.25


def test_config_to_dict_preserves_independent_canonical_schedules():
    cfg = TimeReversalSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=False,
        data_augmentation_func=lambda **_kwargs: (None, None),
        use_tr_policy_consistency=True,
        use_tr_value_consistency=True,
        tr_policy_schedule=TimeReversalScheduleCfg(
            enabled=True,
            target_coeff=0.4,
            warmup_iterations=2,
            rampup_iterations=2,
        ),
        tr_value_schedule=TimeReversalScheduleCfg(
            enabled=True,
            target_coeff=0.7,
            warmup_iterations=0,
            rampup_iterations=0,
        ),
    )
    resolved = cfg.to_dict()

    policy = resolve_time_reversal_schedule(resolved, "policy")
    value = resolve_time_reversal_schedule(resolved, "value")

    assert policy.coefficient(3) == pytest.approx(0.2)
    assert value.coefficient(3) == pytest.approx(0.7)
    assert resolved["tr_policy_schedule"]["warmup_iterations"] == 2
    assert resolved["tr_value_schedule"]["warmup_iterations"] == 0


@pytest.mark.parametrize(
    ("policy_enabled", "value_enabled", "expected"),
    [
        (True, False, (0.3, 0.0)),
        (False, True, (0.0, 0.2)),
        (True, True, (0.3, 0.2)),
        (False, False, (0.0, 0.0)),
    ],
)
def test_canonical_policy_and_value_switches_are_independent(policy_enabled, value_enabled, expected):
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.current_learning_iteration = 0
    algorithm.symmetry = {
        "use_time_reversal_regularization": False,
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.3,
        "value_loss_coeff": 0.2,
        "use_tr_policy_consistency": policy_enabled,
        "use_tr_value_consistency": value_enabled,
        "warmup_iterations": 0,
        "rampup_iterations": 0,
        "ramp_shape": "linear",
    }

    _, policy_coefficient, value_coefficient = algorithm._effective_time_reversal_coefficients()

    assert (policy_coefficient, value_coefficient) == expected


@pytest.mark.parametrize(
    "runner_cfg_type",
    [UnitreeGo2SymmFlatPPORunnerCfg, DobotX1SymmFlatPPORunnerCfg],
    ids=("go2", "x1"),
)
def test_robot_defaults_resolve_actor_only_trs_schedule(runner_cfg_type):
    runner_cfg = runner_cfg_type()
    symmetry = runner_cfg.algorithm.symmetry_cfg.to_dict()
    policy_schedule = resolve_time_reversal_schedule(symmetry, "policy")
    value_schedule = resolve_time_reversal_schedule(symmetry, "value")

    assert symmetry["mirror_loss_coeff"] == pytest.approx(0.1)
    assert symmetry["value_loss_coeff"] == 0.0
    assert policy_schedule.enabled is True
    assert policy_schedule.coefficient(499) == 0.0
    assert policy_schedule.coefficient(500) == pytest.approx(0.1)
    assert value_schedule.enabled is False
    assert all(value_schedule.coefficient(iteration) == 0.0 for iteration in (0, 499, 500, 20_000))
    assert symmetry["tr_augmentation"]["enabled"] is False


@pytest.mark.parametrize(
    ("canonical_flag", "coefficient", "expected_enabled"),
    [
        (None, 0.0, False),
        (None, 0.05, True),
        (False, 0.05, False),
        (True, 0.0, True),
        (True, 0.05, True),
    ],
)
def test_value_consistency_enablement_preserves_legacy_inference(canonical_flag, coefficient, expected_enabled):
    symmetry = TimeReversalSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        data_augmentation_func=lambda **_kwargs: (None, None),
        use_time_reversal_regularization=True,
        mirror_loss_coeff=0.1,
        value_loss_coeff=coefficient,
        use_tr_value_consistency=canonical_flag,
        warmup_iterations=0,
        rampup_iterations=0,
    ).to_dict()

    schedule = resolve_time_reversal_schedule(symmetry, "value")

    assert schedule.enabled is expected_enabled
    assert schedule.target_coeff == pytest.approx(coefficient)
    expected_coefficient = coefficient if expected_enabled else 0.0
    assert schedule.coefficient(0) == pytest.approx(expected_coefficient)


def test_explicitly_enabled_zero_value_ablation_warns_clearly():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.symmetry = UnitreeGo2SymmFlatPPORunnerCfg().algorithm.symmetry_cfg.to_dict()
    algorithm.symmetry["use_tr_value_consistency"] = True

    with pytest.warns(UserWarning, match="critic-consistency ablation.*coefficient is zero"):
        algorithm._validate_time_reversal_configuration()


@pytest.mark.parametrize(
    ("runner_cfg_type", "relative_agent_path"),
    [
        (
            UnitreeGo2SymmFlatPPORunnerCfg,
            "logs/rsl_rl/good_runs/unitree_go2_symm_flat/"
            "2026-09-03_00-14-58_m5_go2_actor_only_trs_m0p1_v0_w500_r0_"
            "fp0p3sum_jtlw0p2_amf0_g2fc1_s43/params/agent.yaml",
        ),
        (
            DobotX1SymmFlatPPORunnerCfg,
            "logs/rsl_rl/good_runs/dobot_x1_symm_flat/"
            "2026-09-03_00-15-11_m5_x1_actor_only_trs_m0p1_v0_w500_r0_x1def_s42/params/agent.yaml",
        ),
    ],
    ids=("go2", "x1"),
)
def test_actor_only_v5_agent_yaml_remains_loadable(runner_cfg_type, relative_agent_path):
    agent_path = Path(__file__).resolve().parents[3] / relative_agent_path
    archived_bytes = agent_path.read_bytes()

    archived = load_yaml(str(agent_path))
    runner_cfg = runner_cfg_type()
    runner_cfg.from_dict(archived)

    symmetry = runner_cfg.algorithm.symmetry_cfg
    policy_schedule = resolve_time_reversal_schedule(symmetry.to_dict(), "policy")
    value_schedule = resolve_time_reversal_schedule(symmetry.to_dict(), "value")
    assert symmetry.value_loss_coeff == pytest.approx(0.0)
    assert symmetry.use_tr_policy_consistency is True
    assert symmetry.use_tr_value_consistency is False
    assert policy_schedule.enabled is True
    assert policy_schedule.coefficient(500) == pytest.approx(0.1)
    assert value_schedule.enabled is False
    assert value_schedule.coefficient(500) == pytest.approx(0.0)
    assert symmetry.tr_augmentation.enabled is False
    assert agent_path.read_bytes() == archived_bytes


@pytest.mark.parametrize(
    ("runner_cfg_type", "relative_checkpoint_path"),
    [
        (
            UnitreeGo2SymmFlatPPORunnerCfg,
            "logs/rsl_rl/good_runs/unitree_go2_symm_flat/"
            "2026-09-03_00-14-58_m5_go2_actor_only_trs_m0p1_v0_w500_r0_"
            "fp0p3sum_jtlw0p2_amf0_g2fc1_s43/model_19999.pt",
        ),
        (
            DobotX1SymmFlatPPORunnerCfg,
            "logs/rsl_rl/good_runs/dobot_x1_symm_flat/"
            "2026-09-03_00-15-11_m5_x1_actor_only_trs_m0p1_v0_w500_r0_x1def_s42/model_19999.pt",
        ),
    ],
    ids=("go2", "x1"),
)
def test_actor_only_v5_checkpoint_remains_loadable(runner_cfg_type, relative_checkpoint_path):
    checkpoint_path = Path(__file__).resolve().parents[3] / relative_checkpoint_path
    archived_bytes = checkpoint_path.read_bytes()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runner_cfg = runner_cfg_type()
    observations = TensorDict({"policy": torch.zeros(1, 72)}, batch_size=[1])
    actor = MLPModel(
        observations,
        runner_cfg.obs_groups,
        "actor",
        12,
        hidden_dims=runner_cfg.actor.hidden_dims,
        activation=runner_cfg.actor.activation,
        obs_normalization=runner_cfg.actor.obs_normalization,
        distribution_cfg=runner_cfg.actor.distribution_cfg.to_dict(),
    )
    critic = MLPModel(
        observations,
        runner_cfg.obs_groups,
        "critic",
        1,
        hidden_dims=runner_cfg.critic.hidden_dims,
        activation=runner_cfg.critic.activation,
        obs_normalization=runner_cfg.critic.obs_normalization,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert checkpoint["iter"] == 19999
    assert torch.all(torch.isfinite(actor(observations)))
    assert torch.all(torch.isfinite(critic(observations)))
    assert len(optimizer.state) == len(checkpoint["optimizer_state_dict"]["state"])
    assert checkpoint["time_reversal_state"]["policy_schedule"]["enabled"] is True
    assert checkpoint["time_reversal_state"]["policy_schedule"]["target_coeff"] == pytest.approx(0.1)
    assert checkpoint["time_reversal_state"]["value_schedule"]["enabled"] is False
    assert checkpoint["time_reversal_state"]["value_schedule"]["target_coeff"] == pytest.approx(0.0)
    assert checkpoint_path.read_bytes() == archived_bytes


def test_shared_agent_config_emits_project_local_symmetry_schema():
    symmetry = UnitreeGo2SymmFlatPPORunnerCfg().algorithm.symmetry_cfg
    serialized = symmetry.to_dict()

    assert isinstance(symmetry, TimeReversalSymmetryCfg)
    assert "tr_policy_schedule" in serialized
    assert "tr_value_schedule" in serialized
    assert "tr_validity" in serialized
    assert "tr_gradient_diagnostics" in serialized
    assert "tr_augmentation" in serialized
    assert serialized["tr_augmentation"]["filter_enabled"] is True


def test_gradient_diagnostics_do_not_mutate_parameter_gradients():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    unused = torch.nn.Parameter(torch.tensor(3.0))
    parameter.grad = torch.tensor(7.0)
    main_loss = parameter.square()
    auxiliary_loss = -parameter

    diagnostics = time_reversal_gradient_diagnostics(
        main_loss,
        auxiliary_loss,
        (parameter, unused),
        effective_coefficient=0.5,
        prefix="actor",
    )

    assert parameter.grad.item() == 7.0
    assert unused.grad is None
    assert diagnostics["tr_gradient/actor/main_norm"] == pytest.approx(4.0)
    assert diagnostics["tr_gradient/actor/auxiliary_norm"] == pytest.approx(1.0)
    assert diagnostics["tr_gradient/actor/cosine"] == pytest.approx(-1.0)
    assert diagnostics["tr_gradient/actor/main_parameter_fraction"] == 0.5
    assert diagnostics["tr_gradient/actor/auxiliary_parameter_fraction"] == 0.5
    assert diagnostics["tr_gradient/actor/nonfinite"] == 0.0


def test_gradient_diagnostics_do_not_mutate_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam((parameter,), lr=0.1)
    optimizer.zero_grad()
    parameter.square().backward()
    optimizer.step()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    parameter_gradient = parameter.grad.clone()

    time_reversal_gradient_diagnostics(
        parameter.square(),
        -parameter,
        (parameter,),
        effective_coefficient=0.5,
        prefix="actor",
    )

    _assert_nested_equal(optimizer.state_dict(), optimizer_state)
    assert torch.equal(parameter.grad, parameter_gradient)


def test_gradient_diagnostics_zero_and_nonfinite_inputs_are_finite_and_flagged():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    zero = time_reversal_gradient_diagnostics(
        parameter * 0.0,
        parameter * 0.0,
        (parameter,),
        effective_coefficient=0.5,
        prefix="zero",
    )
    nonfinite = time_reversal_gradient_diagnostics(
        parameter * float("nan"),
        parameter * float("inf"),
        (parameter,),
        effective_coefficient=0.5,
        prefix="nonfinite",
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in zero.values())
    assert zero["tr_gradient/zero/main_norm"] == 0.0
    assert zero["tr_gradient/zero/auxiliary_norm"] == 0.0
    assert zero["tr_gradient/zero/cosine"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in nonfinite.values())
    assert nonfinite["tr_gradient/nonfinite/nonfinite"] == 1.0


def _validity_observations() -> TensorDict:
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT
    policy = torch.zeros(4, 72)
    policy[:, layout.desired_base_twist.start] = 2.0  # physical command 1 m/s at scale 2
    policy[:, layout.measured_base_twist.start] = 1.8  # physical velocity 0.9 m/s at scale 2
    policy[:, layout.projected_gravity] = torch.tensor((0.0, 0.0, -1.0))
    phase = torch.full((4, 4), 0.25)
    policy[:, layout.foot_phase_sin] = torch.sin(2.0 * torch.pi * phase)
    policy[:, layout.foot_phase_cos] = torch.cos(2.0 * torch.pi * phase)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0)).repeat(4, 1)
    policy[:, layout.foot_theta_sin] = torch.sin(2.0 * torch.pi * offsets)
    policy[:, layout.foot_theta_cos] = torch.cos(2.0 * torch.pi * offsets)
    policy[:, layout.swing_ratio] = 0.55
    policy[:, layout.stance_ratio] = 0.45
    policy[1, layout.measured_base_twist.start] = 0.0
    policy[2, layout.projected_gravity.start] = 1.0
    policy[3, layout.foot_phase_sin] = 0.0
    policy[3, layout.foot_phase_cos] = 1.0
    return TensorDict({"policy": policy}, batch_size=[4])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("command", [1.0, 1.0, 1.0, 1.0]),
        ("command_tracking", [1.0, 0.0, 1.0, 1.0]),
        ("command_upright_phase", [1.0, 1.0, 0.0, 0.0]),
        ("command_tracking_upright_phase", [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_historical_validity_modes_use_only_observation_fields(mode, expected):
    mask = time_reversal_validity_mask(
        _validity_observations(),
        {
            "mode": mode,
            "min_abs_command_velocity": 0.5,
            "tracking_abs_tolerance": 0.25,
            "tracking_rel_tolerance": 0.0,
            "projected_gravity_tolerance": 0.35,
            "phase_boundary_margin": 0.03,
            "command_speed_bin_edges": (0.5, 1.0, 1.5),
        },
    )

    assert mask.combined.squeeze(-1).tolist() == expected
    diagnostics = time_reversal_mask_diagnostics(mask)
    assert "tr_mask/gait_family/trot" in diagnostics
    assert "tr_mask/command_sign/positive" in diagnostics
    assert all(f"tr_mask/command_speed_bin/{index}" in diagnostics for index in range(4))


def test_legacy_instantaneous_ppo_augmentation_warns_and_retains_its_schedule():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.current_learning_iteration = 0
    algorithm.symmetry = {
        "use_time_reversal_regularization": True,
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
        "warmup_iterations": 0,
        "rampup_iterations": 0,
        "ramp_shape": "linear",
        "tr_augmentation": {"enabled": False},
    }

    with pytest.warns(DeprecationWarning, match="is deprecated"):
        algorithm._validate_time_reversal_configuration()

    assert algorithm._legacy_data_augmentation_scale() == 1.0

    algorithm.symmetry["warmup_iterations"] = 500
    algorithm.symmetry["rampup_iterations"] = 1000
    algorithm.current_learning_iteration = 499
    assert algorithm._legacy_data_augmentation_scale() == 0.0
    algorithm.current_learning_iteration = 500
    assert algorithm._legacy_data_augmentation_scale() == 1.0


def test_enabled_augmentation_rejects_disabled_dynamics_filter():
    with pytest.raises(ValueError, match="requires filter_enabled=True"):
        TimeReversalSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            data_augmentation_func=lambda **_kwargs: (None, None),
            tr_augmentation=TimeReversalAugmentationCfg(enabled=True, filter_enabled=False),
        )


def test_dynamics_state_time_reversal_is_an_involution():
    state = _dynamics_state(batch_size=3, phase=0.23, command=1.2, previous_action=0.4)

    round_trip = time_reverse_dynamics_state(time_reverse_dynamics_state(state))

    for name, expected in vars(state).items():
        actual = getattr(round_trip, name)
        if expected.dtype.is_floating_point:
            assert torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0), name
        else:
            assert torch.equal(actual, expected), name
    assert_time_reverse_action_transform(12, torch.device("cpu"))


def _dynamics_component_count(state):
    layout = state.dynamics_feature_layout()
    foot_count = layout.foot_phase_sin.stop - layout.foot_phase_sin.start
    euclidean_count = layout.dimension - 6 - 2 - 2 * foot_count
    return euclidean_count + 1 + 1 + foot_count


def test_component_aware_dynamics_residual_wraps_phase_in_cycles():
    state = _dynamics_state(phase=0.01)
    target = state.dynamics_features()
    prediction = target.clone()
    layout = state.dynamics_feature_layout()
    prediction[:, layout.common_phase_sin] = torch.sin(torch.tensor(2.0 * torch.pi * 0.99))
    prediction[:, layout.common_phase_cos] = torch.cos(torch.tensor(2.0 * torch.pi * 0.99))

    residual = time_reversal_dynamics_residual(
        prediction,
        target,
        feature_mean=torch.zeros(layout.dimension),
        feature_variance=torch.ones(layout.dimension),
        layout=layout,
    )

    expected = 0.02**2 / _dynamics_component_count(state)
    assert residual.item() == pytest.approx(expected, rel=1.0e-5, abs=1.0e-10)

    prediction[:, layout.common_phase_sin] = 0.0
    prediction[:, layout.common_phase_cos] = 0.0
    zero_embedding_residual = time_reversal_dynamics_residual(
        prediction,
        target,
        feature_mean=torch.zeros(layout.dimension),
        feature_variance=torch.ones(layout.dimension),
        layout=layout,
    )
    assert zero_embedding_residual.item() > 0.5 / _dynamics_component_count(state)


def test_component_aware_dynamics_residual_uses_so3_geodesic_angle():
    state = _dynamics_state()
    target = state.dynamics_features()
    prediction = target.clone()
    layout = state.dynamics_feature_layout()
    half_angle = torch.tensor(torch.pi / 4.0)
    yaw_quaternion = torch.stack(
        (torch.cos(half_angle), torch.tensor(0.0), torch.tensor(0.0), torch.sin(half_angle))
    ).unsqueeze(0)
    prediction[:, layout.rotation] = quaternion_to_rotation_6d(yaw_quaternion)

    residual = time_reversal_dynamics_residual(
        prediction,
        target,
        feature_mean=torch.zeros(layout.dimension),
        feature_variance=torch.ones(layout.dimension),
        layout=layout,
    )

    expected = 0.5**2 / _dynamics_component_count(state)
    assert residual.item() == pytest.approx(expected, rel=1.0e-5)


def test_reversed_sequence_builder_uses_prior_reversed_action_and_drops_first_pair():
    states = [
        _dynamics_state(phase=0.1),
        _dynamics_state(phase=0.2),
        _dynamics_state(phase=0.3),
    ]
    actions = [torch.ones(1, 12), torch.full((1, 12), 2.0)]
    states[1].actuator_target.fill_(1.0)
    states[2].actuator_target.fill_(2.0)

    reversed_sequence = build_reversed_sequence_segment(states, actions)
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT

    assert reversed_sequence.observations.shape == (1, 72)
    assert torch.equal(reversed_sequence.actions, actions[0])
    assert torch.equal(
        reversed_sequence.observations[:, layout.previous_action],
        actions[1],
    )
    assert torch.equal(reversed_sequence.current_state.previous_action, actions[1])
    assert torch.equal(reversed_sequence.successor_state.previous_action, actions[0])
    assert torch.equal(reversed_sequence.current_state.actuator_target, states[2].actuator_target)
    assert torch.equal(reversed_sequence.successor_state.actuator_target, states[1].actuator_target)
    assert reversed_sequence.observations[0, layout.desired_base_twist.start].item() == -2.0
    phase_delta = torch.remainder(
        reversed_sequence.successor_state.common_gait_phase - reversed_sequence.current_state.common_gait_phase,
        1.0,
    )
    assert phase_delta.item() == pytest.approx(0.1)


def test_learned_inverse_rebuilds_observation_action_and_actuator_target_histories():
    wrapper, _ = _augmentation_wrapper(num_envs=1)
    augmentation = TimeReversalAugmentation(
        wrapper,
        {
            "action_source": "learned_inverse",
            "dynamics_hidden_dims": (8,),
            "inverse_hidden_dims": (8,),
            "rng_seed": 3,
        },
        "cpu",
    )

    class DistinctLearnedActions(torch.nn.Module):
        def forward(self, features):
            values = torch.arange(features.shape[0], 0, -1, dtype=features.dtype, device=features.device) + 2.0
            return values.unsqueeze(-1).expand(-1, 12)

    augmentation.inverse_target = DistinctLearnedActions()
    states = [replace(_dynamics_state(phase=value), body_mass=torch.ones(1, 13)) for value in (0.1, 0.2, 0.3)]
    analytic_actions = [torch.ones(1, 12), torch.full((1, 12), 2.0)]

    sequence = augmentation._learned_reversed_sequence(states, analytic_actions, environment_index=0)
    previous_action = sequence.current_state.previous_action
    final_action = sequence.actions
    layout = SYMM_QUADRUPED_POLICY_OBS_LAYOUT

    assert not torch.allclose(previous_action, analytic_actions[1])
    assert not torch.allclose(final_action, analytic_actions[0])
    assert torch.equal(sequence.observations[:, layout.previous_action], previous_action)
    assert torch.allclose(sequence.current_state.actuator_target, previous_action * 0.5 + 0.1)
    assert torch.allclose(sequence.successor_state.actuator_target, final_action * 0.5 + 0.1)
    assert torch.equal(sequence.successor_state.previous_action, final_action)


def test_learned_inverse_excludes_action_history_and_uses_physical_state():
    wrapper, _ = _augmentation_wrapper(num_envs=1)
    augmentation = TimeReversalAugmentation(
        wrapper,
        {
            "action_source": "learned_inverse",
            "dynamics_hidden_dims": (8,),
            "inverse_hidden_dims": (8,),
            "rng_seed": 3,
        },
        "cpu",
    )

    class RootPositionInverse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_widths = []

        def forward(self, features):
            self.input_widths.append(features.shape[-1])
            return features[:, :1].expand(-1, 12)

    inverse = RootPositionInverse()
    augmentation.inverse_target = inverse
    states = [replace(_dynamics_state(phase=value), body_mass=torch.ones(1, 13)) for value in (0.1, 0.2, 0.3)]
    for index, state in enumerate(states):
        state.root_position[:, 0] = float(index + 1)
    changed_history = [state.clone() for state in states]
    for index, state in enumerate(changed_history):
        state.previous_action.fill_(10.0 + index)
        state.actuator_target.fill_(-20.0 - index)

    first = augmentation._learned_reversed_sequence(
        states,
        [torch.ones(1, 12), torch.full((1, 12), 2.0)],
        environment_index=0,
    )
    history_changed = augmentation._learned_reversed_sequence(
        changed_history,
        [torch.full((1, 12), -7.0), torch.full((1, 12), 9.0)],
        environment_index=0,
    )
    physical_changed = [state.clone() for state in states]
    physical_changed[1].root_position[:, 0].add_(1.0)
    physical = augmentation._learned_reversed_sequence(
        physical_changed,
        [torch.ones(1, 12), torch.full((1, 12), 2.0)],
        environment_index=0,
    )

    excluded_width = 24
    assert inverse.input_widths == [2 * (augmentation.feature_layout.dimension - excluded_width)] * 3
    assert torch.equal(first.actions, history_changed.actions)
    assert not torch.equal(first.actions, physical.actions)


def test_sidecar_rejects_post_reset_successor_for_done_environment():
    environment = SimpleNamespace(
        num_envs=2,
        step_dt=0.02,
        physics_dt=0.005,
        device="cpu",
        scene=SimpleNamespace(sensors={}),
    )
    sidecar = TRSidecarBuffer(SimpleNamespace(unwrapped=environment))
    observation = TensorDict({"policy": torch.zeros(2, 72)}, batch_size=[2])
    state = _dynamics_state(batch_size=2, phase=0.1)
    successor = _dynamics_state(batch_size=2, phase=0.15)
    action = torch.zeros(2, 12)
    sidecar.capture_before_step(observation, state, action, action, torch.ones_like(action))

    sidecar.capture_after_step(
        observation,
        successor,
        torch.ones(2),
        torch.tensor((0, 1)),
        {"time_outs": torch.tensor((False, True))},
    )

    transition = sidecar.transitions[0]
    assert transition.valid_successor.tolist() == [True, False]
    assert transition.terminal_state_available.tolist() == [False, False]
    assert transition.timeout.tolist() == [False, True]


def test_sidecar_treats_timeout_without_done_as_episode_boundary():
    environment = SimpleNamespace(
        num_envs=2,
        step_dt=0.02,
        physics_dt=0.005,
        device="cpu",
        scene=SimpleNamespace(sensors={}),
    )
    sidecar = TRSidecarBuffer(SimpleNamespace(unwrapped=environment))
    observation = TensorDict({"policy": torch.zeros(2, 72)}, batch_size=[2])
    state = _dynamics_state(batch_size=2, phase=0.1)
    successor = _dynamics_state(batch_size=2, phase=0.15)
    action = torch.zeros(2, 12)
    sidecar.capture_before_step(observation, state, action, action, torch.ones_like(action))

    sidecar.capture_after_step(
        observation,
        successor,
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        {"time_outs": torch.tensor((False, True))},
    )

    transition = sidecar.transitions[0]
    assert transition.done.tolist() == [False, False]
    assert transition.timeout.tolist() == [False, True]
    assert transition.valid_successor.tolist() == [True, False]
    assert transition.episode_id.tolist() == [0, 0]
    assert sidecar.episode_ids.tolist() == [0, 1]


@pytest.mark.parametrize(("successor_offset", "expected"), [(-0.01, True), (0.01, False)])
def test_sidecar_compares_phase_offsets_circularly(successor_offset, expected):
    environment = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        physics_dt=0.005,
        device="cpu",
        scene=SimpleNamespace(sensors={}),
    )
    sidecar = TRSidecarBuffer(SimpleNamespace(unwrapped=environment))
    observation = TensorDict({"policy": torch.zeros(1, 72)}, batch_size=[1])
    state = _dynamics_state(phase=0.1)
    successor = _dynamics_state(phase=0.15)
    state.foot_phase_offsets[:, 0] = 0.99
    successor.foot_phase_offsets[:, 0] = successor_offset
    action = torch.zeros(1, 12)
    sidecar.capture_before_step(observation, state, action, action, torch.ones_like(action))

    sidecar.capture_after_step(observation, successor, torch.ones(1), torch.zeros(1), {})

    assert sidecar.transitions[0].task_continuity.item() is expected


def test_contact_impulse_integrates_force_over_control_step():
    force_history = torch.zeros(1, 2, 1, 3)
    force_history[0, :, 0, 0] = torch.tensor((1.0, 3.0))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force_history))
    environment = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        physics_dt=0.005,
        device="cpu",
        scene=SimpleNamespace(sensors={"foot": sensor}),
    )

    impulse, _ = _capture_contact_diagnostics(SimpleNamespace(unwrapped=environment))

    assert impulse.item() == pytest.approx(0.04)
    assert impulse.item() != pytest.approx(3.0 * environment.physics_dt)


def test_augmentation_fails_closed_when_randomized_material_state_cannot_be_captured():
    wrapper, _ = _augmentation_wrapper(include_material_properties=False)

    with pytest.raises(RuntimeError, match="material properties.*fails closed"):
        TimeReversalAugmentation(
            wrapper,
            {"dynamics_hidden_dims": (8,), "inverse_hidden_dims": (8,)},
            "cpu",
        )


def test_augmentation_nll_is_exact_zero_when_no_candidate_is_valid():
    log_probability = torch.tensor((float("nan"), -2.0), requires_grad=True)
    loss = time_reversal_augmentation_nll(
        log_probability,
        torch.tensor((False, False)),
        torch.tensor((1.0, 1.0)),
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_augmentation_nll_uses_detached_confidence_weighting():
    log_probability = torch.tensor((-1.0, -3.0), requires_grad=True)
    confidence = torch.tensor((1.0, 0.5), requires_grad=True)

    loss = time_reversal_augmentation_nll(
        log_probability,
        torch.tensor((True, True)),
        confidence,
    )
    loss.backward()

    assert loss.item() == pytest.approx((1.0 + 1.5) / 1.5)
    assert confidence.grad is None
    assert log_probability.grad is not None


def test_nonfinite_real_transition_is_rejected_before_normalizer_or_model_update():
    wrapper, gait_command = _augmentation_wrapper(num_envs=1)
    augmentation = TimeReversalAugmentation(
        wrapper,
        {"dynamics_hidden_dims": (8,), "inverse_hidden_dims": (8,), "rng_seed": 4},
        "cpu",
    )
    observation = TensorDict({"policy": torch.zeros(1, 72)}, batch_size=[1])
    action = torch.zeros(1, 12)
    augmentation.capture_before_step(observation, action, action, torch.ones_like(action))
    augmentation.sidecar.pending.action[0, 0] = float("nan")
    gait_command.phase.add_(0.05)
    augmentation.capture_after_step(observation, torch.ones(1), torch.zeros(1, dtype=torch.bool), {})
    forward_before = copy.deepcopy(augmentation.forward_model.state_dict())
    inverse_before = copy.deepcopy(augmentation.inverse_model.state_dict())

    diagnostics = augmentation._train_models()

    assert diagnostics["tr_augmentation/rejected/nonfinite_real_transition"] == 1.0
    assert augmentation.state_normalizer.count.item() == 0.0
    assert augmentation.action_normalizer.count.item() == 0.0
    _assert_nested_equal(augmentation.forward_model.state_dict(), forward_before)
    _assert_nested_equal(augmentation.inverse_model.state_dict(), inverse_before)


def test_enabled_augmentation_builds_pool_and_round_trips_checkpoint_state():
    wrapper, gait_command = _augmentation_wrapper()
    config = {
        "action_source": "analytic",
        "dynamics_hidden_dims": (16,),
        "inverse_hidden_dims": (16,),
        "learning_rate": 1.0e-3,
        "ema_decay": 0.0,
        "validation_fraction": 0.5,
        "validation_quantile": 0.95,
        "threshold_multiplier": 1.0e6,
        "minimum_validation_samples": 1,
        "maximum_validation_loss": 100.0,
        "filter_enabled": True,
        "phase_boundary_margin": 0.01,
        "maximum_contact_impulse": 20.0,
        "action_abs_limit": 10.0,
        "max_augmented_to_original_ratio": 0.5,
        "use_confidence_weights": True,
        "model_updates_per_rollout": 1,
        "model_batch_size": 32,
        "rng_seed": 11,
    }
    augmentation = TimeReversalAugmentation(wrapper, config, "cpu")
    observation = TensorDict({"policy": torch.zeros(4, 72)}, batch_size=[4])
    for step in range(3):
        action = torch.full((4, 12), 0.1 * (step + 1))
        augmentation.capture_before_step(observation, action, action, torch.ones_like(action))
        wrapper.unwrapped.action_manager.action.copy_(action)
        wrapper.unwrapped.scene["robot"].data.joint_pos_target.copy_(action)
        gait_command.phase.add_(0.05)
        augmentation.capture_after_step(
            observation,
            torch.ones(4),
            torch.zeros(4, dtype=torch.long),
            {"time_outs": torch.zeros(4, dtype=torch.bool)},
        )

    diagnostics = augmentation.prepare_update()
    actor = _NllActor()
    nll = augmentation.actor_nll(actor, original_batch_size=8)

    assert diagnostics["tr_augmentation/candidate_count"] == 8.0
    assert diagnostics["tr_augmentation/accepted_count"] == 8.0
    assert augmentation.pool is not None
    assert augmentation.pool.count == 8
    assert nll.requires_grad
    assert torch.isfinite(nll)

    checkpoint = augmentation.state_dict()
    restored = TimeReversalAugmentation(wrapper, config, "cpu")
    restored.load_state_dict(checkpoint)

    assert torch.equal(restored.generator.get_state(), augmentation.generator.get_state())
    assert torch.equal(restored.filter_beta, augmentation.filter_beta)
    assert len(restored.sidecar.transitions) == len(augmentation.sidecar.transitions)
    for restored_parameter, parameter in zip(
        restored.forward_model.parameters(), augmentation.forward_model.parameters()
    ):
        assert torch.equal(restored_parameter, parameter)
    for restored_parameter, parameter in zip(
        restored.inverse_model.parameters(), augmentation.inverse_model.parameters()
    ):
        assert torch.equal(restored_parameter, parameter)
    _assert_nested_equal(restored.forward_optimizer.state_dict(), augmentation.forward_optimizer.state_dict())
    _assert_nested_equal(restored.inverse_optimizer.state_dict(), augmentation.inverse_optimizer.state_dict())
    _assert_nested_equal(restored.state_normalizer.state_dict(), augmentation.state_normalizer.state_dict())
    _assert_nested_equal(restored.action_normalizer.state_dict(), augmentation.action_normalizer.state_dict())
    assert torch.equal(restored.sidecar.episode_ids, augmentation.sidecar.episode_ids)


def test_replay_diagnostic_retains_accepted_and_rejected_candidates_from_same_rollout():
    wrapper, gait_command = _augmentation_wrapper(num_envs=2)
    augmentation = TimeReversalAugmentation(
        wrapper,
        {
            "action_source": "analytic",
            "dynamics_hidden_dims": (8,),
            "inverse_hidden_dims": (8,),
            "filter_enabled": True,
            "phase_boundary_margin": 0.01,
            "maximum_contact_impulse": 20.0,
            "action_abs_limit": 1.0,
            "rng_seed": 3,
        },
        "cpu",
    )
    observation = TensorDict({"policy": torch.zeros(2, 72)}, batch_size=[2])
    for step in range(3):
        action = torch.stack((torch.full((12,), 0.1 * (step + 1)), torch.full((12,), 5.0)))
        augmentation.capture_before_step(observation, action, action, torch.ones_like(action))
        wrapper.unwrapped.action_manager.action.copy_(action)
        wrapper.unwrapped.scene["robot"].data.joint_pos_target.copy_(action)
        gait_command.phase.add_(0.05)
        augmentation.capture_after_step(
            observation,
            torch.ones(2),
            torch.zeros(2, dtype=torch.long),
            {"time_outs": torch.zeros(2, dtype=torch.bool)},
        )
    augmentation.validation_ready = True
    augmentation.filter_beta.fill_(float("inf"))

    training_pool = augmentation._build_candidate_pool({})
    replay_pool = augmentation.replay_candidates

    assert training_pool is not None and training_pool.count == 2
    assert replay_pool is not None and replay_pool.count == 4
    assert replay_pool.accepted_count == 2
    assert replay_pool.rejected_count == 2
    assert replay_pool.current_state.batch_size == 4
    assert replay_pool.successor_state.batch_size == 4
    assert replay_pool.environment_id.tolist() == [0, 0, 1, 1]
    assert replay_pool.rollout_step.tolist() == [1, 0, 1, 0]
    assert torch.equal(replay_pool.filter_accepted, replay_pool.action_gate)
    augmentation.finish_update()
    assert augmentation.pool is None
    assert augmentation.replay_candidates is replay_pool


def _replay_candidate_fixture() -> TRReplayCandidatePool:
    count = 4
    state = _dynamics_state(count)
    successor = replace(state, root_position=state.root_position + 0.01)
    accepted = torch.tensor((True, False, False, True))
    return TRReplayCandidatePool(
        observations=torch.arange(count * 72, dtype=torch.float32).reshape(count, 72),
        successor_observations=torch.arange(count * 72, dtype=torch.float32).reshape(count, 72) + 1.0,
        actions=torch.arange(count, dtype=torch.float32).unsqueeze(-1).expand(count, 12),
        current_state=state,
        successor_state=successor,
        learned_residual=torch.tensor((0.1, 2.0, 2.0, 0.1)),
        learned_threshold=torch.ones(count),
        filter_accepted=accepted,
        finite_gate=torch.ones(count, dtype=torch.bool),
        action_gate=torch.tensor((True, True, False, True)),
        phase_gate=torch.ones(count, dtype=torch.bool),
        contact_impulse_gate=torch.ones(count, dtype=torch.bool),
        dynamics_gate=torch.tensor((True, False, False, True)),
        model_quality_gate=torch.ones(count, dtype=torch.bool),
        contact_impulse=torch.arange(count, dtype=torch.float32),
        contact_mode=torch.arange(count, dtype=torch.long),
        gait_row=torch.tensor((0, 1, 2, 3)),
        command=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        environment_id=torch.tensor((0, 0, 1, 1)),
        episode_id=torch.tensor((7, 7, 9, 9)),
        rollout_step=torch.tensor((3, 2, 8, 7)),
    )


def test_replay_candidate_sampling_keeps_both_decisions_and_markov_metadata():
    candidates = _replay_candidate_fixture()

    sample = candidates.sample_by_filter_decision(maximum_per_decision=1)

    assert sample.count == 2
    assert sample.filter_accepted.tolist() == [True, False]
    assert sample.environment_id.tolist() == [0, 0]
    assert sample.episode_id.tolist() == [7, 7]
    assert sample.rollout_step.tolist() == [3, 2]
    assert sample.current_state.batch_size == 2
    assert sample.successor_state.batch_size == 2
    assert sample.observations.shape == sample.successor_observations.shape == (2, 72)


def test_one_step_replay_reports_false_negative_and_all_confusion_paths():
    candidates = _replay_candidate_fixture()
    callback_calls = 0

    def replay(sample: TRReplayCandidatePool) -> torch.Tensor:
        nonlocal callback_calls
        callback_calls += 1
        assert sample.current_state.batch_size == sample.count
        assert sample.actions.shape == (sample.count, 12)
        assert sample.successor_state.root_position.shape == (sample.count, 3)
        assert sample.environment_id.tolist() == [0, 0, 1, 1]
        return torch.tensor((0.1, 0.1, 2.0, 2.0))

    result = validate_filter_with_one_step_replay(candidates, replay, simulator_threshold=1.0)

    assert callback_calls == 1
    assert result == pytest.approx(
        {
            "true_positive": 1.0,
            "false_positive": 1.0,
            "false_negative": 1.0,
            "true_negative": 1.0,
            "agreement": 0.5,
            "candidate_count": 4.0,
            "learned_accepted_count": 2.0,
            "learned_rejected_count": 2.0,
            "simulator_accepted_count": 2.0,
        }
    )


def test_augmentation_rng_is_private_and_checkpoint_reproduces_next_actor_nll():
    wrapper, _ = _augmentation_wrapper()
    config = {
        "action_source": "analytic",
        "dynamics_hidden_dims": (8,),
        "inverse_hidden_dims": (8,),
        "max_augmented_to_original_ratio": 0.5,
        "rng_seed": 17,
    }
    torch.manual_seed(12345)
    global_rng_before_construction = torch.get_rng_state().clone()
    cuda_rng_before_construction = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    augmentation = TimeReversalAugmentation(wrapper, config, "cpu")
    assert torch.equal(torch.get_rng_state(), global_rng_before_construction)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng_before_construction)
    )

    checkpoint = copy.deepcopy(augmentation.state_dict())
    restored = TimeReversalAugmentation(wrapper, config, "cpu")
    restored.load_state_dict(checkpoint)
    assert torch.equal(torch.get_rng_state(), global_rng_before_construction)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng_before_construction)
    )
    assert torch.equal(
        restored.initialization_generator.get_state(),
        augmentation.initialization_generator.get_state(),
    )

    pool = TRAugmentationPool(
        observations=torch.arange(4 * 72, dtype=torch.float32).reshape(4, 72) / 100.0,
        actions=torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(4, 12),
        confidence=torch.ones(4),
        residual=torch.zeros(4),
        gait_row=torch.zeros(4, dtype=torch.long),
        command=torch.zeros(4, 3),
        contact_mode=torch.zeros(4, dtype=torch.long),
    )
    augmentation.pool = pool
    restored.pool = pool
    actor = _NllActor()
    global_rng_before_nll = torch.get_rng_state().clone()

    expected_loss = augmentation.actor_nll(actor, original_batch_size=4)
    restored_loss = restored.actor_nll(actor, original_batch_size=4)

    assert torch.equal(torch.get_rng_state(), global_rng_before_nll)
    assert torch.equal(expected_loss, restored_loss)
    assert torch.equal(restored.generator.get_state(), augmentation.generator.get_state())


def test_augmentation_checkpoint_rejects_action_source_semantic_mismatch():
    wrapper, _ = _augmentation_wrapper()
    common = {"dynamics_hidden_dims": (8,), "inverse_hidden_dims": (8,), "rng_seed": 5}
    analytic = TimeReversalAugmentation(wrapper, {**common, "action_source": "analytic"}, "cpu")
    learned = TimeReversalAugmentation(wrapper, {**common, "action_source": "learned_inverse"}, "cpu")

    with pytest.raises(ValueError, match="semantic_config"):
        learned.load_state_dict(analytic.state_dict())


def test_augmentation_checkpoint_rejects_missing_schema_field_before_mutation():
    wrapper, _ = _augmentation_wrapper()
    config = {"dynamics_hidden_dims": (8,), "inverse_hidden_dims": (8,), "rng_seed": 5}
    source = TimeReversalAugmentation(wrapper, config, "cpu")
    restored = TimeReversalAugmentation(wrapper, config, "cpu")
    checkpoint = source.state_dict()
    checkpoint.pop("generator_state")
    parameters_before = copy.deepcopy(restored.forward_model.state_dict())

    with pytest.raises(ValueError, match="schema fields"):
        restored.load_state_dict(checkpoint)

    _assert_nested_equal(restored.forward_model.state_dict(), parameters_before)


def _second_order_step(position, velocity, dt=0.05):
    acceleration = -position
    next_position = position + dt * velocity + 0.5 * dt * dt * acceleration
    next_acceleration = -next_position
    next_velocity = velocity + 0.5 * dt * (acceleration + next_acceleration)
    return next_position, next_velocity


def _filter_for_scalar_residual(
    residual,
    *,
    phase=0.2,
    successor_phase=None,
    event_crossing=False,
    impulse=0.0,
    beta=1.0e-8,
):
    successor_phase = phase + 0.01 if successor_phase is None else successor_phase
    return time_reversal_filter_mask(
        observation=torch.zeros(1, 72),
        action=torch.zeros(1, 12),
        reverse_residual=torch.as_tensor((residual,)),
        beta=beta,
        foot_phase=torch.full((1, 4), phase),
        successor_foot_phase=torch.full((1, 4), successor_phase),
        phase_event_crossing=torch.tensor((event_crossing,)),
        swing_ratio=torch.full((1,), 0.55),
        contact_impulse=torch.as_tensor((impulse,)),
        model_quality=True,
        phase_boundary_margin=0.03,
        maximum_contact_impulse=20.0,
        action_abs_limit=10.0,
    )


def test_reversible_second_order_transition_passes_dynamics_filter():
    position = torch.tensor(0.7, dtype=torch.float64)
    velocity = torch.tensor(-0.4, dtype=torch.float64)
    next_position, next_velocity = _second_order_step(position, velocity)

    replay_position, replay_velocity = _second_order_step(next_position, -next_velocity)
    residual = (replay_position - position).square() + (replay_velocity + velocity).square()
    gates = _filter_for_scalar_residual(residual.item(), beta=1.0e-12)

    assert residual.item() < 1.0e-20
    assert gates.dynamics.item()
    assert gates.accepted.item()


def test_dissipative_friction_transition_fails_dynamics_filter():
    position = torch.tensor(0.7, dtype=torch.float64)
    velocity = torch.tensor(-0.4, dtype=torch.float64)
    dt = 0.05
    damping = 0.8

    next_velocity = (1.0 - damping * dt) * velocity
    next_position = position + dt * next_velocity
    replay_velocity = (1.0 - damping * dt) * (-next_velocity)
    replay_position = next_position + dt * replay_velocity
    residual = (replay_position - position).square() + (replay_velocity + velocity).square()
    gates = _filter_for_scalar_residual(residual.item(), beta=1.0e-8)

    assert residual.item() > 1.0e-6
    assert not gates.dynamics.item()
    assert not gates.accepted.item()


@pytest.mark.parametrize(
    ("phase", "impulse", "rejection_field"),
    [(0.0, 0.0, "phase"), (0.2, 25.0, "contact_impulse")],
)
def test_phase_boundary_and_high_impulse_candidates_are_rejected(phase, impulse, rejection_field):
    gates = _filter_for_scalar_residual(0.0, phase=phase, impulse=impulse)

    assert not getattr(gates, rejection_field).item()
    assert not gates.accepted.item()


def test_filter_rejects_successor_phase_boundary_and_event_crossing():
    successor_boundary = _filter_for_scalar_residual(0.0, phase=0.2, successor_phase=0.55)
    current_phase = torch.full((1, 4), 0.50)
    successor_phase = torch.full((1, 4), 0.60)
    crossing = time_reversal_phase_event_crossing(current_phase, successor_phase, torch.full((1,), 0.55))
    event_crossing = _filter_for_scalar_residual(
        0.0,
        phase=0.50,
        successor_phase=0.60,
        event_crossing=crossing.item(),
    )

    assert not successor_boundary.phase_successor.item()
    assert not successor_boundary.accepted.item()
    assert crossing.item()
    assert not event_crossing.phase_event.item()
    assert not event_crossing.accepted.item()


def test_actor_augmentation_nll_decreases_under_supervised_optimization():
    mean = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD((mean,), lr=0.1)
    target = torch.tensor(0.25)

    def loss_value():
        log_probability = -0.5 * (target - mean).square().unsqueeze(0)
        return time_reversal_augmentation_nll(
            log_probability,
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
        )

    initial = loss_value().item()
    for _ in range(30):
        loss = loss_value()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss_value().item() < initial * 0.01


def test_filtered_augmentation_schedule_does_not_enable_legacy_batch_duplication():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.current_learning_iteration = 0
    algorithm.symmetry = {
        "use_time_reversal_regularization": True,
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
        "tr_augmentation": {
            "enabled": True,
            "coefficient": 0.5,
            "schedule": {"enabled": True, "warmup_iterations": 0, "rampup_iterations": 0},
        },
    }

    assert algorithm._legacy_data_augmentation_scale() == 0.0
    assert algorithm._resolved_time_reversal_augmentation_schedule().coefficient(0) == pytest.approx(0.5)


def test_disabled_augmentation_resolves_to_exact_zero_without_allocation():
    algorithm = TimeReversalPPO.__new__(TimeReversalPPO)
    algorithm.current_learning_iteration = 123
    algorithm.symmetry = {
        "use_time_reversal_regularization": False,
        "use_data_augmentation": False,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "value_loss_coeff": 0.0,
        "tr_augmentation": {"enabled": False, "coefficient": 1.0},
    }

    schedule = algorithm._resolved_time_reversal_augmentation_schedule()

    assert not schedule.enabled
    assert schedule.coefficient(algorithm.current_learning_iteration) == 0.0
    assert getattr(algorithm, "_tr_augmentation", None) is None
