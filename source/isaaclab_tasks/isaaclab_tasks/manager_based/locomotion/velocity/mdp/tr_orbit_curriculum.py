# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal-orbit competence curriculum for symmetric locomotion commands."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import torch

TR_ORBIT_COMMAND_CURRICULUM_MODE = "tr_orbit_reward_threshold_v1"
"""Name of the reward-threshold time-reversal-orbit curriculum."""


class TimeReversalOrbitCurriculum:
    """Stateful gait/signed-speed competence curriculum closed under time reversal.

    The state lives on CPU because it is updated only at command-segment
    boundaries. This makes the random-number state portable between CPU and GPU
    training and permits exact checkpoint continuation.
    """

    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        num_envs: int,
        gait_partner_indices: Sequence[int],
        velocity_range: tuple[float, float],
        velocity_bin_count: int,
        ewma_coefficient: float,
        unlock_threshold: float,
        current_cell_increment: float,
        neighbor_increment: float,
        exploration_floor: float,
        maximum_weight: float,
        seed: int,
        initial_gait_weights: Sequence[float] | None = None,
    ) -> None:
        """Initialize the curriculum.

        Args:
            num_envs: Number of parallel environments.
            gait_partner_indices: Involutive time-reversal partner row map.
            velocity_range: Symmetric forward-velocity range [m/s].
            velocity_bin_count: Number of signed forward-velocity bins.
            ewma_coefficient: Success EWMA update coefficient.
            unlock_threshold: EWMA threshold for increasing curriculum weights.
            current_cell_increment: Weight increment for a competent cell.
            neighbor_increment: Weight increment for adjacent signed-speed bins.
            exploration_floor: Strictly positive minimum sampling weight.
            maximum_weight: Maximum sampling weight.
            seed: CPU generator seed used by curriculum sampling.
            initial_gait_weights: Optional gait-row prior replicated across velocity bins.
        """
        self._validate_positive_integer("num_envs", num_envs)
        self._validate_positive_integer("velocity_bin_count", velocity_bin_count)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError(f"seed must be a nonnegative integer; received {seed!r}.")
        for name, value in (
            ("ewma_coefficient", ewma_coefficient),
            ("unlock_threshold", unlock_threshold),
            ("current_cell_increment", current_cell_increment),
            ("neighbor_increment", neighbor_increment),
            ("exploration_floor", exploration_floor),
            ("maximum_weight", maximum_weight),
        ):
            self._validate_finite_real(name, value)
        if not 0.0 < ewma_coefficient <= 1.0:
            raise ValueError(f"ewma_coefficient must be in (0, 1]; received {ewma_coefficient!r}.")
        if not 0.0 <= unlock_threshold <= 1.0:
            raise ValueError(f"unlock_threshold must be in [0, 1]; received {unlock_threshold!r}.")
        if current_cell_increment < 0.0 or neighbor_increment < 0.0:
            raise ValueError("current_cell_increment and neighbor_increment must be nonnegative.")
        if exploration_floor <= 0.0:
            raise ValueError(f"exploration_floor must be positive; received {exploration_floor!r}.")
        if maximum_weight < exploration_floor:
            raise ValueError("maximum_weight must be greater than or equal to exploration_floor.")

        velocity_min, velocity_max = velocity_range
        self._validate_finite_real("velocity_range[0]", velocity_min)
        self._validate_finite_real("velocity_range[1]", velocity_max)
        if velocity_min >= velocity_max or not math.isclose(
            float(velocity_min), -float(velocity_max), rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"velocity_range must be finite, increasing, and symmetric about zero; received {velocity_range!r}."
            )

        self.num_envs = int(num_envs)
        self.gait_partner_indices = torch.as_tensor(gait_partner_indices, dtype=torch.long, device="cpu")
        if self.gait_partner_indices.ndim != 1 or len(self.gait_partner_indices) == 0:
            raise ValueError("gait_partner_indices must be a nonempty one-dimensional sequence.")
        expected_rows = torch.arange(len(self.gait_partner_indices), dtype=torch.long)
        if torch.any((self.gait_partner_indices < 0) | (self.gait_partner_indices >= len(expected_rows))):
            raise ValueError("gait_partner_indices contains an out-of-range row.")
        if not torch.equal(self.gait_partner_indices[self.gait_partner_indices], expected_rows):
            raise ValueError("gait_partner_indices must be an involution.")

        self.velocity_bin_edges = torch.linspace(
            float(velocity_min), float(velocity_max), int(velocity_bin_count) + 1, dtype=torch.float64
        )
        self.velocity_bin_centers = 0.5 * (self.velocity_bin_edges[:-1] + self.velocity_bin_edges[1:])
        self.velocity_partner_indices = torch.arange(int(velocity_bin_count) - 1, -1, -1, dtype=torch.long)
        if not torch.allclose(
            self.velocity_bin_centers[self.velocity_partner_indices],
            -self.velocity_bin_centers,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("Constructed velocity-bin partner map is not sign symmetric.")

        self.ewma_coefficient = float(ewma_coefficient)
        self.unlock_threshold = float(unlock_threshold)
        self.current_cell_increment = float(current_cell_increment)
        self.neighbor_increment = float(neighbor_increment)
        self.exploration_floor = float(exploration_floor)
        self.maximum_weight = float(maximum_weight)
        self.seed = int(seed)

        num_gaits = len(self.gait_partner_indices)
        if initial_gait_weights is None:
            gait_weights = torch.ones(num_gaits, dtype=torch.float64)
        else:
            gait_weights = torch.as_tensor(initial_gait_weights, dtype=torch.float64, device="cpu")
            if gait_weights.shape != (num_gaits,):
                raise ValueError(
                    f"initial_gait_weights must have shape ({num_gaits},); received {tuple(gait_weights.shape)}."
                )
            if torch.any(~torch.isfinite(gait_weights)) or torch.any(gait_weights < 0.0):
                raise ValueError("initial_gait_weights must contain finite, nonnegative values.")
            if gait_weights.sum() <= 0.0:
                raise ValueError("initial_gait_weights must have a positive sum.")
        if not torch.equal(gait_weights, gait_weights[self.gait_partner_indices]):
            raise ValueError("initial_gait_weights must be equal across time-reversal gait partners.")

        self.gait_prior_weights = gait_weights.clamp_min(self.exploration_floor).clone()
        self.weights = self.gait_prior_weights[:, None].expand(-1, int(velocity_bin_count)).clone()
        self.weights.clamp_(min=self.exploration_floor, max=self.maximum_weight)
        self.ewma_success = torch.zeros_like(self.weights)
        self.visit_counts = torch.zeros_like(self.weights, dtype=torch.long)
        self.unlock_state = torch.zeros_like(self.weights, dtype=torch.bool)

        self.current_gait_indices = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.current_velocity_bin_indices = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.segment_error_xy_sum = torch.zeros(self.num_envs, dtype=torch.float64)
        self.segment_error_yaw_sum = torch.zeros(self.num_envs, dtype=torch.float64)
        self.segment_step_count = torch.zeros(self.num_envs, dtype=torch.long)

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._validate_orbit_symmetry()

    @property
    def num_gaits(self) -> int:
        """Number of gait rows in the curriculum grid."""
        return int(self.weights.shape[0])

    @property
    def velocity_bin_count(self) -> int:
        """Number of signed forward-velocity bins in the curriculum grid."""
        return int(self.weights.shape[1])

    @property
    def sampling_eligibility(self) -> torch.Tensor:
        """Cells explicitly unlocked by competence updates."""
        return self.unlock_state.clone()

    def velocity_bin_indices(self, velocity: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Return signed-speed bins for forward velocities [m/s]."""
        values = torch.as_tensor(velocity, dtype=torch.float64, device="cpu")
        if torch.any(~torch.isfinite(values)):
            raise ValueError("Forward velocities must be finite.")
        return torch.bucketize(values, self.velocity_bin_edges[1:-1]).clamp(0, self.velocity_bin_count - 1)

    def sample_joint(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample joint gait/signed-speed cells."""
        self._validate_nonnegative_integer("count", count)
        if count == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty
        flat = torch.multinomial(self.weights.reshape(-1), count, replacement=True, generator=self._generator)
        return torch.div(flat, self.velocity_bin_count, rounding_mode="floor"), flat % self.velocity_bin_count

    def sample_velocity_given_gait(self, gait_indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Sample signed-speed bins conditional on gait rows."""
        gait = self._validate_indices("gait_indices", gait_indices, self.num_gaits)
        choices = [
            torch.multinomial(self.weights[row], 1, replacement=True, generator=self._generator)[0] for row in gait
        ]
        return torch.stack(choices) if choices else torch.empty(0, dtype=torch.long)

    def sample_gait_given_velocity(self, velocity_bin_indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Sample gait rows conditional on signed-speed bins."""
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        choices = [
            torch.multinomial(self.weights[:, column], 1, replacement=True, generator=self._generator)[0]
            for column in velocity_bin
        ]
        return torch.stack(choices) if choices else torch.empty(0, dtype=torch.long)

    def sample_forward_velocity(self, velocity_bin_indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Sample forward velocity uniformly within each selected bin [m/s]."""
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        if len(velocity_bin) == 0:
            return torch.empty(0, dtype=torch.float64)
        lower = self.velocity_bin_edges[velocity_bin]
        upper = self.velocity_bin_edges[velocity_bin + 1]
        unit = torch.rand(len(velocity_bin), dtype=torch.float64, generator=self._generator)
        return lower + unit * (upper - lower)

    def update_gait_prior_weights(self, gait_weights: torch.Tensor | Sequence[float]) -> None:
        """Apply an iteration-dependent gait prior without resetting learned competence.

        The existing grid is scaled by the ratio between the new and previous
        gait-row priors. Partner-equal priors therefore preserve every TR orbit,
        while learned within-row speed preferences remain intact.
        """
        updated = torch.as_tensor(gait_weights, dtype=torch.float64, device="cpu")
        if updated.shape != (self.num_gaits,):
            raise ValueError(f"gait_weights must have shape ({self.num_gaits},); received {tuple(updated.shape)}.")
        if torch.any(~torch.isfinite(updated)) or torch.any(updated < 0.0) or updated.sum() <= 0.0:
            raise ValueError("gait_weights must contain finite, nonnegative values with a positive sum.")
        if not torch.equal(updated, updated[self.gait_partner_indices]):
            raise ValueError("gait_weights must be equal across time-reversal gait partners.")
        updated = updated.clamp_min(self.exploration_floor)
        self.weights.mul_((updated / self.gait_prior_weights)[:, None])
        self.weights.clamp_(min=self.exploration_floor, max=self.maximum_weight)
        self.gait_prior_weights.copy_(updated)
        self._validate_orbit_symmetry()

    def set_current_cells(
        self,
        env_ids: torch.Tensor | Sequence[int],
        gait_indices: torch.Tensor | Sequence[int],
        velocity_bin_indices: torch.Tensor | Sequence[int],
    ) -> None:
        """Set active cells and clear their segment accumulators."""
        ids = self._validate_indices("env_ids", env_ids, self.num_envs)
        gait = self._validate_indices("gait_indices", gait_indices, self.num_gaits)
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        if len(ids) != len(gait) or len(ids) != len(velocity_bin):
            raise ValueError("env_ids, gait_indices, and velocity_bin_indices must have equal lengths.")
        self.current_gait_indices[ids] = gait
        self.current_velocity_bin_indices[ids] = velocity_bin
        self._clear_segments(ids)

    def clear_current_cells(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Invalidate active cells and clear partial segments for selected environments."""
        ids = self._validate_indices("env_ids", env_ids, self.num_envs)
        self.current_gait_indices[ids] = -1
        self.current_velocity_bin_indices[ids] = -1
        self._clear_segments(ids)

    def accumulate(
        self,
        error_xy: torch.Tensor,
        error_yaw: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Accumulate body-frame tracking errors for active command segments.

        Args:
            error_xy: Body-frame XY velocity error [m/s].
            error_yaw: Body-frame yaw-rate error [rad/s].
            env_ids: Environment indices represented by the error tensors.
        """
        ids = (
            torch.arange(self.num_envs)
            if env_ids is None
            else self._validate_indices("env_ids", env_ids, self.num_envs)
        )
        xy = torch.as_tensor(error_xy, dtype=torch.float64, device="cpu").reshape(-1)
        yaw = torch.as_tensor(error_yaw, dtype=torch.float64, device="cpu").reshape(-1)
        if len(ids) != len(xy) or len(ids) != len(yaw):
            raise ValueError("Tracking errors must contain one value per selected environment.")
        if torch.any(~torch.isfinite(xy)) or torch.any(~torch.isfinite(yaw)):
            raise ValueError("Tracking errors must be finite.")
        valid = (self.current_gait_indices[ids] >= 0) & (self.current_velocity_bin_indices[ids] >= 0)
        valid_ids = ids[valid]
        self.segment_error_xy_sum[valid_ids] += xy[valid]
        self.segment_error_yaw_sum[valid_ids] += yaw[valid]
        self.segment_step_count[valid_ids] += 1

    def finish_segments(
        self,
        env_ids: torch.Tensor | Sequence[int],
        *,
        xy_success_threshold: torch.Tensor | Sequence[float],
        yaw_success_threshold: torch.Tensor | Sequence[float],
        terminated: torch.Tensor | Sequence[bool] | bool,
    ) -> torch.Tensor:
        """Finish selected segments, update orbit state, and return binary success."""
        ids = self._validate_indices("env_ids", env_ids, self.num_envs)
        xy_threshold = self._selected_values("xy_success_threshold", xy_success_threshold, len(ids), torch.float64)
        yaw_threshold = self._selected_values("yaw_success_threshold", yaw_success_threshold, len(ids), torch.float64)
        terminated_values = self._selected_values("terminated", terminated, len(ids), torch.bool)
        if torch.any(~torch.isfinite(xy_threshold)) or torch.any(xy_threshold < 0.0):
            raise ValueError("xy_success_threshold must be finite and nonnegative.")
        if torch.any(~torch.isfinite(yaw_threshold)) or torch.any(yaw_threshold < 0.0):
            raise ValueError("yaw_success_threshold must be finite and nonnegative.")

        success = torch.zeros(len(ids), dtype=torch.bool)
        valid = (
            (self.current_gait_indices[ids] >= 0)
            & (self.current_velocity_bin_indices[ids] >= 0)
            & (self.segment_step_count[ids] > 0)
        )
        if torch.any(valid):
            valid_ids = ids[valid]
            count = self.segment_step_count[valid_ids].to(torch.float64)
            mean_xy = self.segment_error_xy_sum[valid_ids] / count
            mean_yaw = self.segment_error_yaw_sum[valid_ids] / count
            valid_success = (
                (mean_xy < xy_threshold[valid]) & (mean_yaw < yaw_threshold[valid]) & ~terminated_values[valid]
            )
            success[valid] = valid_success
            self.update_cells(
                self.current_gait_indices[valid_ids],
                self.current_velocity_bin_indices[valid_ids],
                valid_success,
            )
        self._clear_segments(ids)
        return success

    def update_cells(
        self,
        gait_indices: torch.Tensor | Sequence[int],
        velocity_bin_indices: torch.Tensor | Sequence[int],
        success: torch.Tensor | Sequence[bool],
    ) -> None:
        """Apply arbitrary binary segment outcomes while preserving orbit equality."""
        gait = self._validate_indices("gait_indices", gait_indices, self.num_gaits)
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        outcomes = torch.as_tensor(success, dtype=torch.bool, device="cpu").reshape(-1)
        if len(gait) != len(velocity_bin) or len(gait) != len(outcomes):
            raise ValueError("gait_indices, velocity_bin_indices, and success must have equal lengths.")
        eta = self.ewma_coefficient
        for gait_value, bin_value, outcome in zip(gait.tolist(), velocity_bin.tolist(), outcomes.tolist(), strict=True):
            orbit = self._orbit_cells(gait_value, bin_value)
            reference_gait, reference_bin = orbit[0]
            updated_ewma = (1.0 - eta) * self.ewma_success[reference_gait, reference_bin] + eta * float(outcome)
            updated_visits = self.visit_counts[reference_gait, reference_bin] + 1
            for orbit_gait, orbit_bin in orbit:
                self.ewma_success[orbit_gait, orbit_bin] = updated_ewma
                self.visit_counts[orbit_gait, orbit_bin] = updated_visits
            if updated_ewma > self.unlock_threshold:
                self._increment_orbit(gait_value, bin_value, self.current_cell_increment)
                updated_neighbor_orbits: set[tuple[int, int]] = {min(orbit)}
                for neighbor_bin in (bin_value - 1, bin_value + 1):
                    if 0 <= neighbor_bin < self.velocity_bin_count:
                        orbit_key = min(self._orbit_cells(gait_value, neighbor_bin))
                        if orbit_key in updated_neighbor_orbits:
                            continue
                        updated_neighbor_orbits.add(orbit_key)
                        self._increment_orbit(gait_value, neighbor_bin, self.neighbor_increment)
        self.weights.clamp_(min=self.exploration_floor, max=self.maximum_weight)
        self._validate_orbit_symmetry()

    def state_dict(self) -> dict:
        """Return deterministic curriculum and per-environment segment state."""
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "config": self._config_state(),
            "gait_prior_weights": self.gait_prior_weights.clone(),
            "weights": self.weights.clone(),
            "ewma_success": self.ewma_success.clone(),
            "visit_counts": self.visit_counts.clone(),
            "unlock_state": self.unlock_state.clone(),
            "current_gait_indices": self.current_gait_indices.clone(),
            "current_velocity_bin_indices": self.current_velocity_bin_indices.clone(),
            "segment_error_xy_sum": self.segment_error_xy_sum.clone(),
            "segment_error_yaw_sum": self.segment_error_yaw_sum.clone(),
            "segment_step_count": self.segment_step_count.clone(),
            "rng_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore an exactly compatible curriculum checkpoint."""
        expected_keys = set(self.state_dict())
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            received = set(state) if isinstance(state, Mapping) else set()
            raise ValueError(
                "Curriculum checkpoint fields do not match: "
                f"missing={sorted(expected_keys - received)}, unexpected={sorted(received - expected_keys)}."
            )
        if state["schema_version"] != self._STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported curriculum checkpoint schema: "
                f"expected {self._STATE_SCHEMA_VERSION}, received {state['schema_version']!r}."
            )
        if state["config"] != self._config_state():
            raise ValueError(
                "Curriculum checkpoint configuration does not match the active command configuration: "
                f"expected {self._config_state()!r}, received {state['config']!r}."
            )
        tensor_names = (
            "gait_prior_weights",
            "weights",
            "ewma_success",
            "visit_counts",
            "unlock_state",
            "current_gait_indices",
            "current_velocity_bin_indices",
            "segment_error_xy_sum",
            "segment_error_yaw_sum",
            "segment_step_count",
        )
        for name in tensor_names:
            target = getattr(self, name)
            source = torch.as_tensor(state[name], device="cpu")
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"Curriculum checkpoint {name} must have shape {tuple(target.shape)} and dtype {target.dtype}; "
                    f"received {tuple(source.shape)} and {source.dtype}."
                )
            target.copy_(source)
        rng_state = torch.as_tensor(state["rng_state"], dtype=torch.uint8, device="cpu")
        self._generator.set_state(rng_state)
        self._validate_orbit_symmetry()

    def _config_state(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "gait_partner_indices": self.gait_partner_indices.tolist(),
            "velocity_bin_edges": self.velocity_bin_edges.tolist(),
            "ewma_coefficient": self.ewma_coefficient,
            "unlock_threshold": self.unlock_threshold,
            "current_cell_increment": self.current_cell_increment,
            "neighbor_increment": self.neighbor_increment,
            "exploration_floor": self.exploration_floor,
            "maximum_weight": self.maximum_weight,
            "seed": self.seed,
        }

    def _orbit_cells(self, gait_index: int, velocity_bin_index: int) -> list[tuple[int, int]]:
        partner = (
            int(self.gait_partner_indices[gait_index]),
            int(self.velocity_partner_indices[velocity_bin_index]),
        )
        cell = (gait_index, velocity_bin_index)
        return [cell] if partner == cell else [cell, partner]

    def _increment_orbit(self, gait_index: int, velocity_bin_index: int, increment: float) -> None:
        for orbit_gait, orbit_bin in self._orbit_cells(gait_index, velocity_bin_index):
            self.weights[orbit_gait, orbit_bin] += increment
            self.unlock_state[orbit_gait, orbit_bin] = True

    def _validate_orbit_symmetry(self) -> None:
        if not torch.equal(self.gait_prior_weights, self.gait_prior_weights[self.gait_partner_indices]):
            raise ValueError("Curriculum gait_prior_weights are not equal across time-reversal gait partners.")
        partner_gaits = self.gait_partner_indices[:, None].expand_as(self.weights)
        partner_velocity = self.velocity_partner_indices[None, :].expand_as(self.weights)
        for name in ("weights", "ewma_success", "visit_counts", "unlock_state"):
            value = getattr(self, name)
            if not torch.equal(value, value[partner_gaits, partner_velocity]):
                raise ValueError(f"Curriculum {name} is not equal across time-reversal orbit partners.")

    def _clear_segments(self, env_ids: torch.Tensor) -> None:
        self.segment_error_xy_sum[env_ids] = 0.0
        self.segment_error_yaw_sum[env_ids] = 0.0
        self.segment_step_count[env_ids] = 0

    @staticmethod
    def _selected_values(name: str, value, count: int, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device="cpu")
        if tensor.ndim == 0:
            return tensor.expand(count).clone()
        tensor = tensor.reshape(-1)
        if len(tensor) != count:
            raise ValueError(f"{name} must be scalar or contain {count} values; received {len(tensor)}.")
        return tensor

    @staticmethod
    def _validate_indices(name: str, value, upper_bound: int) -> torch.Tensor:
        indices = torch.as_tensor(value, dtype=torch.long, device="cpu").reshape(-1)
        if torch.any((indices < 0) | (indices >= upper_bound)):
            raise IndexError(f"{name} must be in [0, {upper_bound}); received {indices.tolist()}.")
        return indices

    @staticmethod
    def _validate_finite_real(name: str, value: Real) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; received {value!r}.")

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; received {value!r}.")

    @staticmethod
    def _validate_nonnegative_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer; received {value!r}.")
