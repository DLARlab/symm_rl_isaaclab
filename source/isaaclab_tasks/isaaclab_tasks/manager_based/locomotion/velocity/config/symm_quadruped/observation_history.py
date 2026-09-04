# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-major observation history for symmetric quadruped policies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ObservationManager
from isaaclab.utils.configclass import configclass


@configclass
class PolicyObservationHistoryCfg:
    """Configuration for the policy observation-history wrapper."""

    history_length: int = 20
    """Number of complete observation frames supplied to the policy, including the current frame."""

    group_name: str = "policy"
    """Concatenated observation group to stack."""


class FrameMajorObservationHistoryWrapper:
    """Stack complete observation frames before flattening them for an MLP.

    The oldest complete frame is placed first and the current frame is placed last. Reset environments are
    zero-padded, so a history of length three begins as ``[0, 0, observation]``. Calls that do not update history
    return the existing stack without duplicating the current frame.
    """

    def __init__(
        self,
        manager: ObservationManager,
        *,
        num_envs: int,
        device: str | torch.device,
        history_length: int,
        group_name: str = "policy",
    ) -> None:
        """Initialize the history wrapper.

        Args:
            manager: Observation manager that produces the current observation frame.
            num_envs: Number of vectorized environments.
            device: Device on which observations are stored.
            history_length: Number of complete frames in the flattened policy input, including the current frame.
            group_name: Concatenated observation group to stack.

        Raises:
            ValueError: If the history length or selected observation group is invalid.
        """
        if history_length < 1:
            raise ValueError(f"history_length must be positive, got {history_length}.")
        if group_name not in manager.group_obs_dim:
            raise ValueError(f"Unknown observation group {group_name!r}.")

        group_dim = manager.group_obs_dim[group_name]
        if not isinstance(group_dim, tuple):
            raise ValueError(f"Frame-major history requires a concatenated observation group, got {group_dim!r}.")

        self._manager = manager
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._history_length = history_length
        self._group_name = group_name
        self._frame_dim = math.prod(group_dim)
        self._history: torch.Tensor | None = None
        self._needs_initial_frame = torch.ones(num_envs, dtype=torch.bool, device=self._device)
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None
        self._group_obs_dim = dict(manager.group_obs_dim)
        self._group_obs_dim[group_name] = (history_length * self._frame_dim,)

    def __getattr__(self, name: str):
        """Delegate observation-manager attributes not changed by this wrapper."""
        manager = self.__dict__.get("_manager")
        if manager is None:
            raise AttributeError(name)
        return getattr(manager, name)

    def __str__(self) -> str:
        """Return the wrapped observation-manager description."""
        return (
            f"{self._manager}\n"
            f"Frame-major history: group={self._group_name!r}, length={self._history_length}, "
            f"shape={self._group_obs_dim[self._group_name]}"
        )

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        """Shapes of computed observation groups after frame stacking."""
        return self._group_obs_dim

    @property
    def history_length(self) -> int:
        """Number of complete frames in the policy input, including the current frame."""
        return self._history_length

    @property
    def frame_dim(self) -> int:
        """Flattened dimension of one observation frame."""
        return self._frame_dim

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the wrapped manager and clear history for the selected environments."""
        extras = self._manager.reset(env_ids=env_ids)
        if self._history is not None:
            if env_ids is None:
                self._history.zero_()
                self._needs_initial_frame.fill_(True)
            else:
                self._history[env_ids] = 0.0
                self._needs_initial_frame[env_ids] = True
        return extras

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute observations and optionally append one complete frame to history."""
        observations = self._manager.compute(update_history=update_history)
        current_observation = observations[self._group_name]
        if not isinstance(current_observation, torch.Tensor):
            raise ValueError(
                f"Frame-major history requires a concatenated observation group, got {type(current_observation)}."
            )

        current_frame = current_observation.reshape(self._num_envs, -1)
        if current_frame.shape[1] != self._frame_dim:
            raise ValueError(
                f"Observation group {self._group_name!r} changed from {self._frame_dim} to "
                f"{current_frame.shape[1]} dimensions."
            )
        if self._history is None:
            self._history = torch.zeros(
                (self._num_envs, self._history_length, self._frame_dim),
                dtype=current_frame.dtype,
                device=self._device,
            )

        if update_history:
            if self._history_length > 1:
                self._history[:, :-1].copy_(self._history[:, 1:].clone())
            self._history[:, -1].copy_(current_frame)
            self._needs_initial_frame.fill_(False)
        else:
            self._history[self._needs_initial_frame, -1] = current_frame[self._needs_initial_frame]
            self._needs_initial_frame.fill_(False)
        # RSL-RL retains the returned observation until after the next environment step. Keep the returned tensor
        # independent from the internal history, which is updated and reset in-place during that step.
        stacked_observation = self._history.reshape(self._num_envs, -1).clone()

        wrapped_observations = dict(observations)
        wrapped_observations[self._group_name] = stacked_observation
        self._obs_buffer = wrapped_observations
        return wrapped_observations
