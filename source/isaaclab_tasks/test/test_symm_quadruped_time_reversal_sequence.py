# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for transition-aligned causal-history time-reversal reconstruction."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_sequence import (
    ACTION_HISTORY_LENGTH,
    TR_CONSISTENCY_MAPPING_VERSION,
    TransitionAlignedTRBuffer,
    TransitionAlignedTRRecord,
    TRSequenceValidityConfig,
    TRSimulatorValidityMetadata,
    build_transition_aligned_reversed_history,
    time_reversal_tracking_mask,
)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import symm_quadruped

_LAYOUT = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_LAYOUT
_SCALE = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_SCALE
_OBS_DIM = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_DIM
_ACTION_DIM = symm_quadruped.SYMM_QUADRUPED_POLICY_OBS_DIMENSIONS.previous_action


def _action(index: int, shape: tuple[int, ...], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((*shape, _ACTION_DIM), float(index), device=device, dtype=dtype)


def _frame(
    index: int,
    shape: tuple[int, ...] = (),
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    dt: float = 0.02,
    velocity: float = 0.4,
    phase_increment: float = 0.017,
) -> torch.Tensor:
    frame = torch.zeros((*shape, _OBS_DIM), device=device, dtype=dtype)
    frame[..., _LAYOUT.projected_gravity] = torch.tensor((0.0, 0.0, -1.0), device=device, dtype=dtype)
    frame[..., _LAYOUT.velocity_command] = torch.tensor(
        (2.0, 0.0, 0.0),
        device=device,
        dtype=dtype,
    )
    frame[..., _LAYOUT.joint_position] = index * dt * velocity
    frame[..., _LAYOUT.joint_velocity] = velocity * _SCALE.joint_velocity[0]
    frame[..., _LAYOUT.previous_action] = _action(index - 1, shape, device=device, dtype=dtype)
    frame[..., _LAYOUT.second_previous_action] = _action(index - 2, shape, device=device, dtype=dtype)
    frame[..., _LAYOUT.gait_period] = 1.5
    frame[..., _LAYOUT.duty_factor] = 0.6
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=device, dtype=dtype)
    phase = torch.remainder(0.23 + index * phase_increment + offsets, 1.0)
    frame[..., _LAYOUT.foot_phase_sin] = torch.sin(2.0 * torch.pi * phase)
    frame[..., _LAYOUT.foot_phase_cos] = torch.cos(2.0 * torch.pi * phase)
    return frame


def _record(
    index: int,
    history_length: int,
    *,
    num_envs: int = 1,
    update: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    simulator: TRSimulatorValidityMetadata | None = None,
) -> TransitionAlignedTRRecord:
    latest = _frame(index, (num_envs,), device=device, dtype=dtype)
    frames = torch.stack(
        [
            _frame(frame_index, (num_envs,), device=device, dtype=dtype)
            for frame_index in range(index - history_length + 1, index + 1)
        ],
        dim=1,
    )
    policy_observation = symm_quadruped.pack_term_major_policy_history(frames)
    if simulator is None:
        simulator = TRSimulatorValidityMetadata.neutral(num_envs, device=device, dtype=dtype)
        simulator.body_linear_velocity_b[:, 0] = 1.0
    zeros = torch.zeros(num_envs, dtype=torch.long, device=device)
    return TransitionAlignedTRRecord(
        policy_observation=policy_observation,
        latest_frame=latest,
        action=_action(index, (num_envs,), device=device, dtype=dtype),
        episode_id=zeros.clone(),
        command_segment_id=zeros.clone(),
        gait_segment_id=zeros.clone(),
        disturbance_generation_id=zeros.clone(),
        collection_update_id=torch.full_like(zeros, update),
        transition_valid=torch.ones(num_envs, dtype=torch.bool, device=device),
        done=torch.zeros(num_envs, dtype=torch.bool, device=device),
        timeout=torch.zeros(num_envs, dtype=torch.bool, device=device),
        history_warmup_complete=torch.ones(num_envs, dtype=torch.bool, device=device),
        gait_row=zeros.clone(),
        simulator=simulator,
    )


def _replace_frame(record: TransitionAlignedTRRecord, frame: torch.Tensor) -> TransitionAlignedTRRecord:
    history_length = record.policy_observation.shape[-1] // _OBS_DIM
    history = frame[:, None, :].repeat(1, history_length, 1)
    return replace(
        record,
        policy_observation=symm_quadruped.pack_term_major_policy_history(history),
        latest_frame=frame,
    )


def test_mapping_version_and_two_action_lag_are_explicit():
    assert TR_CONSISTENCY_MAPPING_VERSION == "transition_aligned_causal_sequence_v1"
    assert ACTION_HISTORY_LENGTH == 2
    frame = _frame(0)
    frame[_LAYOUT.previous_action] = 1.0
    frame[_LAYOUT.second_previous_action] = 0.0
    reversed_frame = symm_quadruped.build_reversed_causal_policy_frame(
        frame,
        torch.full((_ACTION_DIM,), 2.0, dtype=frame.dtype),
        torch.full((_ACTION_DIM,), 3.0, dtype=frame.dtype),
    )
    approximate = symm_quadruped.time_reverse_observations_framewise_approx(frame)

    assert torch.equal(reversed_frame[_LAYOUT.previous_action], torch.full((_ACTION_DIM,), 2.0, dtype=frame.dtype))
    assert torch.equal(
        reversed_frame[_LAYOUT.second_previous_action],
        torch.full((_ACTION_DIM,), 3.0, dtype=frame.dtype),
    )
    assert torch.equal(approximate[_LAYOUT.previous_action], torch.full((_ACTION_DIM,), 1.0, dtype=frame.dtype))
    assert torch.equal(approximate[_LAYOUT.second_previous_action], torch.zeros(_ACTION_DIM, dtype=frame.dtype))


def test_exact_action_history_gives_zero_synthetic_actor_loss_while_frozen_history_does_not():
    frame_at_t = _frame(0)
    frame_at_t[_LAYOUT.previous_action] = 3.0
    frame_at_t_plus_1 = _frame(1)
    frame_at_t_plus_1[_LAYOUT.previous_action] = 2.0
    action_t_plus_1 = torch.full((_ACTION_DIM,), 3.0, dtype=frame_at_t.dtype)
    action_t_plus_2 = torch.full((_ACTION_DIM,), 4.0, dtype=frame_at_t.dtype)

    exact_reversed = symm_quadruped.build_reversed_causal_policy_frame(
        frame_at_t_plus_1,
        action_t_plus_1,
        action_t_plus_2,
    )
    frozen_reversed = symm_quadruped.time_reverse_observations_framewise_approx(frame_at_t_plus_1)

    def synthetic_actor(frame: torch.Tensor) -> torch.Tensor:
        return frame[_LAYOUT.previous_action]

    target = synthetic_actor(frame_at_t).detach()
    exact_loss = torch.mean((synthetic_actor(exact_reversed) - target).square())
    frozen_loss = torch.mean((synthetic_actor(frozen_reversed) - target).square())
    assert exact_loss.item() == 0.0
    assert frozen_loss.item() > 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_exact_history_builder_supports_arbitrary_leading_shapes_and_reversed_order(device, dtype):
    history_length = 4
    leading_shape = (2, 3)
    frames = torch.stack(
        [_frame(index, leading_shape, device=device, dtype=dtype) for index in range(1, history_length + 1)],
        dim=-2,
    )
    actions = torch.stack(
        [_action(index, leading_shape, device=device, dtype=dtype) for index in range(1, history_length + 2)],
        dim=-2,
    )

    packed = build_transition_aligned_reversed_history(frames, actions, history_length)
    reversed_frames = symm_quadruped.unpack_term_major_policy_history(packed)

    assert packed.shape == (*leading_shape, history_length * _OBS_DIM)
    assert packed.dtype == dtype
    assert packed.device.type == device
    expected_ids = torch.tensor((history_length, history_length - 1, 2, 1), device=device, dtype=dtype)
    assert torch.allclose(
        reversed_frames[..., _LAYOUT.joint_position.start],
        expected_ids * 0.02 * 0.4,
        atol=1.0e-7,
        rtol=0.0,
    )
    for reversed_index, forward_index in enumerate(range(history_length, 0, -1)):
        assert torch.equal(
            reversed_frames[..., reversed_index, _LAYOUT.previous_action],
            _action(forward_index, leading_shape, device=device, dtype=dtype),
        )
        assert torch.equal(
            reversed_frames[..., reversed_index, _LAYOUT.second_previous_action],
            _action(forward_index + 1, leading_shape, device=device, dtype=dtype),
        )


def test_transition_aligned_history_is_kinematically_consistent_but_framewise_order_is_not():
    history_length = 5
    dt = 0.02
    velocity = 0.4
    frames = torch.stack([_frame(index, dt=dt, velocity=velocity) for index in range(1, history_length + 1)])
    actions = torch.stack(
        [_action(index, (), device="cpu", dtype=frames.dtype) for index in range(1, history_length + 2)]
    )
    exact = symm_quadruped.unpack_term_major_policy_history(
        build_transition_aligned_reversed_history(frames, actions, history_length)
    )
    approximate = symm_quadruped.time_reverse_observations_framewise_approx(frames)

    exact_position_delta = exact[1:, _LAYOUT.joint_position] - exact[:-1, _LAYOUT.joint_position]
    exact_velocity = exact[:-1, _LAYOUT.joint_velocity] / _SCALE.joint_velocity[0]
    approximate_position_delta = approximate[1:, _LAYOUT.joint_position] - approximate[:-1, _LAYOUT.joint_position]
    approximate_velocity = approximate[:-1, _LAYOUT.joint_velocity] / _SCALE.joint_velocity[0]

    assert torch.allclose(exact_position_delta, dt * exact_velocity, atol=1.0e-12, rtol=0.0)
    assert not torch.allclose(approximate_position_delta, dt * approximate_velocity, atol=1.0e-12, rtol=0.0)


def test_reflected_phase_progresses_consistently_in_reversed_time():
    history_length = 6
    phase_increment = 0.017
    frames = torch.stack([_frame(index, phase_increment=phase_increment) for index in range(1, history_length + 1)])
    actions = torch.stack(
        [_action(index, (), device="cpu", dtype=frames.dtype) for index in range(1, history_length + 2)]
    )
    reversed_frames = symm_quadruped.unpack_term_major_policy_history(
        build_transition_aligned_reversed_history(frames, actions, history_length)
    )
    phase = torch.remainder(
        torch.atan2(
            reversed_frames[:, _LAYOUT.foot_phase_sin],
            reversed_frames[:, _LAYOUT.foot_phase_cos],
        )
        / (2.0 * torch.pi),
        1.0,
    )
    phase_delta = torch.remainder(phase[1:] - phase[:-1], 1.0)

    assert torch.allclose(phase_delta, torch.full_like(phase_delta, phase_increment), atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize("history_length,required", [(1, 3), (30, 32)])
def test_buffer_requires_exactly_history_plus_two_finalized_records(history_length, required):
    buffer = TransitionAlignedTRBuffer(1, history_length, "cpu")
    for index in range(required - 1):
        buffer.append(_record(index, history_length, update=index // 24))
        assert buffer.candidate_count(index // 24) == 0

    buffer.append(_record(required - 1, history_length, update=(required - 1) // 24))

    assert buffer.required_sequence_records == required
    assert buffer.candidate_count((required - 1) // 24) == 1


def test_history_30_buffer_spans_the_24_plus_8_rollout_boundary():
    buffer = TransitionAlignedTRBuffer(1, 30, "cpu", allowed_policy_version_span=1)
    for index in range(24):
        buffer.append(_record(index, 30, update=0))
    assert buffer.records_collected == 24
    assert buffer.candidate_count(0) == 0

    for index in range(24, 31):
        buffer.append(_record(index, 30, update=1))
        assert buffer.candidate_count(1) == 0
    buffer.append(_record(31, 30, update=1))
    candidates = buffer.peek_candidates(1)

    assert candidates is not None
    assert candidates.count == 1
    assert candidates.policy_version_span.item() == 1
    assert torch.equal(candidates.actor_source_observation, _record(0, 30).policy_observation)
    assert torch.equal(candidates.value_source_observation, _record(1, 30).policy_observation)
    reversed_frames = symm_quadruped.unpack_term_major_policy_history(candidates.reversed_observation)
    expected_first = _frame(30)[_LAYOUT.joint_position.start]
    expected_last = _frame(1)[_LAYOUT.joint_position.start]
    assert reversed_frames[0, 0, _LAYOUT.joint_position.start] == expected_first
    assert reversed_frames[0, -1, _LAYOUT.joint_position.start] == expected_last


def test_buffer_retains_only_linear_compact_policy_history():
    history_length = 30
    num_envs = 2
    buffer = TransitionAlignedTRBuffer(num_envs, history_length, "cpu")

    for index in range(history_length + 2):
        buffer.append(_record(index, history_length, num_envs=num_envs))

    assert len(buffer._policy_frames) == 2 * history_length + 1
    assert all(frame.shape == (num_envs, _OBS_DIM) for frame in buffer._policy_frames)
    assert all(not hasattr(record, "policy_observation") for record in buffer._records)
    retained_policy_elements = sum(frame.numel() for frame in buffer._policy_frames)
    formerly_retained_elements = num_envs * (history_length + 2) * history_length * _OBS_DIM
    assert retained_policy_elements == num_envs * (2 * history_length + 1) * _OBS_DIM
    assert retained_policy_elements < formerly_retained_elements


def test_buffer_defers_dense_candidate_construction_until_collection(monkeypatch):
    history_length = 5
    buffer = TransitionAlignedTRBuffer(1, history_length, "cpu")
    build_calls = 0
    original_build = buffer._build_dense_candidate

    def counted_build(window):
        nonlocal build_calls
        build_calls += 1
        return original_build(window)

    monkeypatch.setattr(buffer, "_build_dense_candidate", counted_build)
    for index in range(history_length + 2):
        buffer.append(_record(index, history_length), collection_update=0)

    assert build_calls == 0
    assert buffer.peek_candidates(0) is not None
    assert build_calls == 1


def test_reset_invalidation_can_cross_the_rollout_inference_mode_boundary():
    with torch.inference_mode():
        buffer = TransitionAlignedTRBuffer(1, 1, "cpu")
        for index in range(3):
            buffer.append(_record(index, 1), collection_update=0)

    assert buffer._records_since_clear.is_inference()
    buffer.clear_environment_mask(torch.ones(1, dtype=torch.bool))

    assert not buffer._records_since_clear.is_inference()
    assert buffer._records_since_clear.item() == 0
    assert buffer.candidate_count(0) == 0


@pytest.mark.parametrize(
    "boundary",
    ["reset", "timeout", "command", "gait", "period", "duty", "push", "invalid_final"],
)
def test_buffer_rejects_every_required_boundary(boundary):
    records = [_record(index, 1) for index in range(3)]
    target = 2 if boundary == "invalid_final" else 1
    if boundary == "reset":
        records[target] = replace(records[target], episode_id=torch.ones(1, dtype=torch.long))
    elif boundary == "timeout":
        records[target] = replace(records[target], timeout=torch.ones(1, dtype=torch.bool))
    elif boundary == "command":
        changed = records[target].latest_frame.clone()
        changed[:, _LAYOUT.velocity_command.start] += 0.5
        records[target] = _replace_frame(records[target], changed)
    elif boundary == "gait":
        records[target] = replace(
            records[target],
            gait_row=torch.ones(1, dtype=torch.long),
            gait_segment_id=torch.ones(1, dtype=torch.long),
        )
    elif boundary == "period":
        changed = records[target].latest_frame.clone()
        changed[:, _LAYOUT.gait_period] += 0.1
        records[target] = _replace_frame(records[target], changed)
    elif boundary == "duty":
        changed = records[target].latest_frame.clone()
        changed[:, _LAYOUT.duty_factor] += 0.1
        records[target] = _replace_frame(records[target], changed)
    elif boundary == "push":
        records[target] = replace(records[target], disturbance_generation_id=torch.ones(1, dtype=torch.long))
    else:
        records[target] = replace(records[target], transition_valid=torch.zeros(1, dtype=torch.bool))

    buffer = TransitionAlignedTRBuffer(1, 1, "cpu")
    for record in records:
        buffer.append(record)

    assert buffer.candidate_count(0) == 0


def test_tracking_metadata_changes_validity_without_changing_policy_tensors():
    cfg = TRSequenceValidityConfig(
        mode="command_tracking",
        tracking_abs_xy=0.05,
        tracking_rel_xy=0.0,
        tracking_abs_yaw=0.05,
        tracking_rel_yaw=0.0,
    )
    good = TRSimulatorValidityMetadata.neutral(1, device="cpu", dtype=torch.float64)
    good.body_linear_velocity_b[:, 0] = 1.0
    bad = replace(good, body_linear_velocity_b=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    assert time_reversal_tracking_mask(good, command, cfg).item()
    assert not time_reversal_tracking_mask(bad, command, cfg).item()

    good_buffer = TransitionAlignedTRBuffer(1, 1, "cpu", validity_cfg=cfg)
    bad_buffer = TransitionAlignedTRBuffer(1, 1, "cpu", validity_cfg=cfg)
    for index in range(3):
        good_record = _record(index, 1, simulator=good)
        bad_record = replace(good_record, simulator=bad)
        assert torch.equal(good_record.policy_observation, bad_record.policy_observation)
        actor = torch.nn.Linear(good_record.policy_observation.shape[-1], _ACTION_DIM, bias=False).double()
        critic = torch.nn.Linear(good_record.policy_observation.shape[-1], 1, bias=False).double()
        torch.testing.assert_close(
            actor(good_record.policy_observation),
            actor(bad_record.policy_observation),
        )
        torch.testing.assert_close(
            critic(good_record.policy_observation),
            critic(bad_record.policy_observation),
        )
        good_buffer.append(good_record)
        bad_buffer.append(bad_record)

    assert good_buffer.candidate_count(0) == 1
    assert bad_buffer.candidate_count(0) == 0


def test_complete_segment_double_reversal_recovers_internal_frames_and_action_history():
    history_length = 5
    forward_frames = torch.stack([_frame(index) for index in range(1, history_length + 1)])
    forward_actions = torch.stack(
        [_action(index, (), device="cpu", dtype=forward_frames.dtype) for index in range(1, history_length + 2)]
    )
    reversed_frames = symm_quadruped.unpack_term_major_policy_history(
        build_transition_aligned_reversed_history(forward_frames, forward_actions, history_length)
    )
    reversed_dynamics_actions = torch.stack(
        [_action(index, (), device="cpu", dtype=forward_frames.dtype) for index in range(history_length - 1, -2, -1)]
    )

    recovered = symm_quadruped.unpack_term_major_policy_history(
        build_transition_aligned_reversed_history(
            reversed_frames,
            reversed_dynamics_actions,
            history_length,
        )
    )

    assert torch.allclose(recovered, forward_frames, atol=1.0e-12, rtol=0.0)


def test_buffer_keeps_only_newest_eligible_candidate_per_environment_and_update():
    buffer = TransitionAlignedTRBuffer(2, 1, "cpu")
    for index in range(5):
        buffer.append(_record(index, 1, num_envs=2))

    candidates = buffer.pop_candidates(0)

    assert candidates is not None
    assert candidates.count == 2
    latest_frames = symm_quadruped.unpack_term_major_policy_history(candidates.value_source_observation)
    assert torch.equal(latest_frames[:, -1, _LAYOUT.previous_action], torch.full((2, 12), 2.0, dtype=torch.float64))
    assert buffer.candidate_count(0) == 0


def test_policy_version_span_and_candidate_age_bound_replay():
    span_buffer = TransitionAlignedTRBuffer(1, 1, "cpu", allowed_policy_version_span=0)
    span_buffer.append(_record(0, 1, update=0))
    span_buffer.append(_record(1, 1, update=0))
    span_buffer.append(_record(2, 1, update=1))
    assert span_buffer.candidate_count(1) == 0

    age_buffer = TransitionAlignedTRBuffer(1, 1, "cpu", candidate_max_age_updates=1)
    for index in range(3):
        age_buffer.append(_record(index, 1, update=0))
    candidates = age_buffer.peek_candidates(1)
    assert candidates is not None
    assert candidates.candidate_age_updates.item() == 1
    assert age_buffer.candidate_count(2) == 0


def test_legacy_observation_wrapper_warns_and_preserves_approximate_behavior():
    frame = _frame(2)

    with pytest.warns(DeprecationWarning, match="framewise feature approximation"):
        legacy = symm_quadruped.time_reverse_observations(frame)

    assert torch.equal(legacy, symm_quadruped.time_reverse_observations_framewise_approx(frame))
