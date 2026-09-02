# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-reversal-orbit competence curriculum for symmetric locomotion commands."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import torch

TR_ORBIT_COMMAND_CURRICULUM_MODE = "tr_orbit_reward_threshold_v1"
"""Name of the reward-threshold time-reversal-orbit curriculum."""


def _type_exact_equal(first: object, second: object) -> bool:
    """Return whether nested checkpoint metadata matches in value and type."""
    if type(first) is not type(second):
        return False
    if isinstance(first, Mapping):
        return first.keys() == second.keys() and all(_type_exact_equal(first[key], second[key]) for key in first)
    if isinstance(first, list | tuple):
        return len(first) == len(second) and all(
            _type_exact_equal(first_value, second_value) for first_value, second_value in zip(first, second)
        )
    return bool(first == second)


class TimeReversalOrbitCurriculum:
    """Stateful gait/signed-speed competence curriculum closed under time reversal.

    The state lives on CPU because it is updated only at command-segment
    boundaries. Persistent checkpoints continue global competence and sampling
    state; per-environment physical/task segments are deliberately transient.
    """

    _STATE_SCHEMA_VERSION = 2
    _LEGACY_STATE_SCHEMA_VERSION = 1
    _PERSISTENT_TENSOR_FIELDS = (
        "gait_prior_weights",
        "priority",
        "eligible",
        "mastered",
        "ewma_success",
        "visit_counts",
    )

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
        initial_max_abs_speed: float = 0.5,
        minimum_visits: int = 20,
        locked_cell_weight: float = 0.0,
    ) -> None:
        """Initialize the curriculum.

        Args:
            num_envs: Number of parallel environments.
            gait_partner_indices: Involutive time-reversal partner row map.
            velocity_range: Symmetric forward-velocity range [m/s].
            velocity_bin_count: Number of signed forward-velocity bins.
            ewma_coefficient: Success EWMA update coefficient.
            unlock_threshold: EWMA threshold for mastering an eligible orbit.
            current_cell_increment: Priority increment for a competent orbit.
            neighbor_increment: Initial priority increment for a newly eligible speed orbit.
            exploration_floor: Strictly positive minimum sampling weight.
            maximum_weight: Maximum sampling weight.
            seed: CPU generator seed used by curriculum sampling.
            initial_gait_weights: Optional gait-row prior replicated across velocity bins.
            initial_max_abs_speed: Largest initially eligible absolute forward speed [m/s].
            minimum_visits: Minimum completed segments required before mastery.
            locked_cell_weight: Sampling weight assigned to ineligible cells.
        """
        self._validate_positive_integer("num_envs", num_envs)
        self._validate_positive_integer("velocity_bin_count", velocity_bin_count)
        self._validate_positive_integer("minimum_visits", minimum_visits)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError(f"seed must be a nonnegative integer; received {seed!r}.")
        for name, value in (
            ("ewma_coefficient", ewma_coefficient),
            ("unlock_threshold", unlock_threshold),
            ("current_cell_increment", current_cell_increment),
            ("neighbor_increment", neighbor_increment),
            ("exploration_floor", exploration_floor),
            ("maximum_weight", maximum_weight),
            ("initial_max_abs_speed", initial_max_abs_speed),
            ("locked_cell_weight", locked_cell_weight),
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
        if initial_max_abs_speed < 0.0:
            raise ValueError(f"initial_max_abs_speed must be nonnegative; received {initial_max_abs_speed!r}.")
        if not 0.0 <= locked_cell_weight <= maximum_weight:
            raise ValueError(f"locked_cell_weight must be in [0, maximum_weight]; received {locked_cell_weight!r}.")

        velocity_min, velocity_max = velocity_range
        self._validate_finite_real("velocity_range[0]", velocity_min)
        self._validate_finite_real("velocity_range[1]", velocity_max)
        if velocity_min >= velocity_max or not math.isclose(
            float(velocity_min), -float(velocity_max), rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError(
                f"velocity_range must be finite, increasing, and symmetric about zero; received {velocity_range!r}."
            )
        if initial_max_abs_speed > abs(float(velocity_max)):
            raise ValueError(
                "initial_max_abs_speed must not exceed the configured maximum absolute forward speed; "
                f"received {initial_max_abs_speed!r} for {velocity_range!r}."
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
        self.initial_max_abs_speed = float(initial_max_abs_speed)
        self.minimum_visits = int(minimum_visits)
        self.locked_cell_weight = float(locked_cell_weight)

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

        self.gait_prior_weights = gait_weights.clone()
        grid_shape = (num_gaits, int(velocity_bin_count))
        self.priority = torch.zeros(grid_shape, dtype=torch.float64)
        self.eligible = torch.zeros(grid_shape, dtype=torch.bool)
        self.mastered = torch.zeros(grid_shape, dtype=torch.bool)
        self.ewma_success = torch.zeros(grid_shape, dtype=torch.float64)
        self.visit_counts = torch.zeros(grid_shape, dtype=torch.long)
        self._activate_initial_support()

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
        return int(self.priority.shape[0])

    @property
    def velocity_bin_count(self) -> int:
        """Number of signed forward-velocity bins in the curriculum grid."""
        return int(self.priority.shape[1])

    @property
    def weights(self) -> torch.Tensor:
        """Effective sampling weights after applying the staged eligibility gate."""
        locked = torch.full_like(self.priority, self.locked_cell_weight)
        effective = torch.where(self.eligible, self.priority, locked)
        return torch.where(self.gait_prior_weights[:, None] > 0.0, effective, torch.zeros_like(effective))

    @property
    def sampling_eligibility(self) -> torch.Tensor:
        """Cells currently eligible for staged curriculum sampling."""
        return self.eligible.clone()

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
        weights = self.weights.reshape(-1)
        self._validate_sampling_support(weights, "joint gait/velocity curriculum")
        flat = torch.multinomial(weights, count, replacement=True, generator=self._generator)
        return torch.div(flat, self.velocity_bin_count, rounding_mode="floor"), flat % self.velocity_bin_count

    def sample_velocity_given_gait(self, gait_indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Sample signed-speed bins conditional on gait rows."""
        gait = self._validate_indices("gait_indices", gait_indices, self.num_gaits)
        choices = []
        weights = self.weights
        for row in gait.tolist():
            self._validate_sampling_support(weights[row], f"gait row {row}")
            choices.append(torch.multinomial(weights[row], 1, replacement=True, generator=self._generator)[0])
        return torch.stack(choices) if choices else torch.empty(0, dtype=torch.long)

    def sample_gait_given_velocity(self, velocity_bin_indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Sample gait rows conditional on signed-speed bins."""
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        choices = []
        weights = self.weights
        for column in velocity_bin.tolist():
            self._validate_sampling_support(weights[:, column], f"velocity bin {column}")
            choices.append(torch.multinomial(weights[:, column], 1, replacement=True, generator=self._generator)[0])
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

        Positive-to-positive prior changes rescale existing priorities, while
        a row whose prior becomes zero is removed from the sampling support.
        When a previously zero row becomes positive, only its configured
        initial speed band becomes eligible.
        """
        updated = torch.as_tensor(gait_weights, dtype=torch.float64, device="cpu")
        if updated.shape != (self.num_gaits,):
            raise ValueError(f"gait_weights must have shape ({self.num_gaits},); received {tuple(updated.shape)}.")
        if torch.any(~torch.isfinite(updated)) or torch.any(updated < 0.0) or updated.sum() <= 0.0:
            raise ValueError("gait_weights must contain finite, nonnegative values with a positive sum.")
        if not torch.equal(updated, updated[self.gait_partner_indices]):
            raise ValueError("gait_weights must be equal across time-reversal gait partners.")

        previous = self.gait_prior_weights.clone()
        remained_positive = (previous > 0.0) & (updated > 0.0)
        if torch.any(remained_positive):
            rows = remained_positive.nonzero(as_tuple=False).flatten()
            self.priority[rows] *= (updated[rows] / previous[rows]).unsqueeze(-1)
        became_zero = updated <= 0.0
        if torch.any(became_zero):
            self.priority[became_zero] = 0.0
            self.eligible[became_zero] = False
            self.mastered[became_zero] = False
        self.gait_prior_weights.copy_(updated)
        self.priority.clamp_(min=0.0, max=self.maximum_weight)
        eligible_priority = self.priority[self.eligible].clamp_min(self.exploration_floor)
        self.priority[self.eligible] = eligible_priority.clamp_max(self.maximum_weight)
        self._activate_initial_support()
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
        """Aggregate binary outcomes by TR orbit and update competence once per orbit.

        Multiple environments can finish the same orbit simultaneously. Their
        outcomes are reduced to one mean before the EWMA update, while every
        completed segment contributes to the visit count. Sorting canonical
        orbit keys makes the update independent of input order.
        """
        gait = self._validate_indices("gait_indices", gait_indices, self.num_gaits)
        velocity_bin = self._validate_indices("velocity_bin_indices", velocity_bin_indices, self.velocity_bin_count)
        outcomes = torch.as_tensor(success, dtype=torch.bool, device="cpu").reshape(-1)
        if len(gait) != len(velocity_bin) or len(gait) != len(outcomes):
            raise ValueError("gait_indices, velocity_bin_indices, and success must have equal lengths.")

        grouped_outcomes: dict[tuple[int, int], list[bool]] = {}
        for gait_value, bin_value, outcome in zip(gait.tolist(), velocity_bin.tolist(), outcomes.tolist(), strict=True):
            orbit_key = min(self._orbit_cells(gait_value, bin_value))
            grouped_outcomes.setdefault(orbit_key, []).append(outcome)

        eta = self.ewma_coefficient
        for gait_value, bin_value in sorted(grouped_outcomes):
            orbit = self._orbit_cells(gait_value, bin_value)
            reference_gait, reference_bin = orbit[0]
            orbit_outcomes = grouped_outcomes[(gait_value, bin_value)]
            mean_success = sum(orbit_outcomes) / len(orbit_outcomes)
            updated_ewma = (1.0 - eta) * self.ewma_success[reference_gait, reference_bin] + eta * mean_success
            updated_visits = self.visit_counts[reference_gait, reference_bin] + len(orbit_outcomes)
            for orbit_gait, orbit_bin in orbit:
                self.ewma_success[orbit_gait, orbit_bin] = updated_ewma
                self.visit_counts[orbit_gait, orbit_bin] = updated_visits

            is_eligible = bool(self.eligible[reference_gait, reference_bin])
            qualifies = (
                is_eligible
                and int(updated_visits) >= self.minimum_visits
                and float(updated_ewma) >= self.unlock_threshold
            )
            if not qualifies:
                continue
            first_mastery = not bool(self.mastered[reference_gait, reference_bin])
            self._increment_orbit_priority(gait_value, bin_value, self.current_cell_increment)
            if first_mastery:
                for orbit_gait, orbit_bin in orbit:
                    self.mastered[orbit_gait, orbit_bin] = True
                unlocked_orbits: set[tuple[int, int]] = set()
                for next_bin in self._next_larger_velocity_bins(reference_bin):
                    next_orbit_key = min(self._orbit_cells(reference_gait, next_bin))
                    if next_orbit_key in unlocked_orbits:
                        continue
                    unlocked_orbits.add(next_orbit_key)
                    self._activate_orbit(reference_gait, next_bin, self.neighbor_increment)
        self._validate_orbit_symmetry()

    def state_dict(self) -> dict:
        """Return persistent global curriculum state without rollout transients."""
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "config": self._config_state(),
            "gait_prior_weights": self.gait_prior_weights.clone(),
            "priority": self.priority.clone(),
            "eligible": self.eligible.clone(),
            "mastered": self.mastered.clone(),
            "ewma_success": self.ewma_success.clone(),
            "visit_counts": self.visit_counts.clone(),
            "rng_state": self._generator.get_state().clone(),
        }

    def validate_state_dict(
        self,
        state: Mapping,
        *,
        warn_migration: bool = True,
        legacy_gait_prior_weights: torch.Tensor | Sequence[float] | None = None,
    ) -> Mapping:
        """Validate and return a current-schema persistent checkpoint mapping."""
        state = self.migrate_state_dict(
            state,
            warn_migration=warn_migration,
            legacy_gait_prior_weights=legacy_gait_prior_weights,
        )
        expected_keys = set(self.state_dict())
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            received = set(state) if isinstance(state, Mapping) else set()
            raise ValueError(
                "Curriculum checkpoint fields do not match: "
                f"missing={sorted(expected_keys - received)}, unexpected={sorted(received - expected_keys)}."
            )
        if type(state["schema_version"]) is not int or state["schema_version"] != self._STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported curriculum checkpoint schema: "
                f"expected {self._STATE_SCHEMA_VERSION}, received {state['schema_version']!r}."
            )
        if not _type_exact_equal(state["config"], self._config_state()):
            raise ValueError(
                "Curriculum checkpoint configuration does not match the active command configuration: "
                f"expected {self._config_state()!r}, received {state['config']!r}."
            )
        for name in self._PERSISTENT_TENSOR_FIELDS:
            target = getattr(self, name)
            source = torch.as_tensor(state[name], device="cpu")
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"Curriculum checkpoint {name} must have shape {tuple(target.shape)} and dtype {target.dtype}; "
                    f"received {tuple(source.shape)} and {source.dtype}."
                )
        rng_state = torch.as_tensor(state["rng_state"], dtype=torch.uint8, device="cpu")
        if rng_state.ndim != 1:
            raise ValueError(
                f"Curriculum checkpoint rng_state must be one-dimensional; received {tuple(rng_state.shape)}."
            )
        self._validate_persistent_state_values(state)
        generator_probe = torch.Generator(device="cpu")
        try:
            generator_probe.set_state(rng_state.clone())
        except RuntimeError as exc:
            raise ValueError("Curriculum checkpoint rng_state is not a valid CPU generator state.") from exc
        return state

    def _validate_persistent_state_values(self, state: Mapping) -> None:
        """Validate checkpoint tensor invariants without mutating live curriculum state."""
        tensors = {name: torch.as_tensor(state[name], device="cpu") for name in self._PERSISTENT_TENSOR_FIELDS}
        gait_prior = tensors["gait_prior_weights"]
        if torch.any(~torch.isfinite(gait_prior)) or torch.any(gait_prior < 0.0) or gait_prior.sum() <= 0.0:
            raise ValueError("Curriculum checkpoint gait_prior_weights must be finite, nonnegative, and nonzero.")
        if not torch.equal(gait_prior, gait_prior[self.gait_partner_indices]):
            raise ValueError(
                "Curriculum checkpoint gait_prior_weights are not equal across time-reversal gait partners."
            )

        priority = tensors["priority"]
        eligible = tensors["eligible"]
        mastered = tensors["mastered"]
        ewma_success = tensors["ewma_success"]
        visit_counts = tensors["visit_counts"]
        partner_gaits = self.gait_partner_indices[:, None].expand_as(priority)
        partner_velocity = self.velocity_partner_indices[None, :].expand_as(priority)
        for name in ("priority", "eligible", "mastered", "ewma_success", "visit_counts"):
            value = tensors[name]
            if not torch.equal(value, value[partner_gaits, partner_velocity]):
                raise ValueError(f"Curriculum checkpoint {name} is not equal across time-reversal orbit partners.")
        if torch.any(~torch.isfinite(priority)) or torch.any((priority < 0.0) | (priority > self.maximum_weight)):
            raise ValueError("Curriculum checkpoint priority must be finite and in [0, maximum_weight].")
        if torch.any(~torch.isfinite(ewma_success)) or torch.any((ewma_success < 0.0) | (ewma_success > 1.0)):
            raise ValueError("Curriculum checkpoint ewma_success must be finite and in [0, 1].")
        if torch.any(visit_counts < 0):
            raise ValueError("Curriculum checkpoint visit_counts must be nonnegative.")
        if torch.any(mastered & ~eligible):
            raise ValueError("Curriculum checkpoint mastered cells must remain eligible.")
        zero_prior = gait_prior[:, None] <= 0.0
        if torch.any(eligible & zero_prior):
            raise ValueError("Curriculum checkpoint cells with zero gait prior must remain ineligible.")
        if torch.any(priority[eligible] < self.exploration_floor):
            raise ValueError("Eligible curriculum checkpoint priority must satisfy exploration_floor.")

    def load_state_dict(
        self,
        state: Mapping,
        *,
        legacy_gait_prior_weights: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        """Restore compatible persistent state and clear rollout transients."""
        state = self.validate_state_dict(state, legacy_gait_prior_weights=legacy_gait_prior_weights)
        for name in self._PERSISTENT_TENSOR_FIELDS:
            getattr(self, name).copy_(torch.as_tensor(state[name], device="cpu"))
        rng_state = torch.as_tensor(state["rng_state"], dtype=torch.uint8, device="cpu")
        self._generator.set_state(rng_state)
        self.clear_current_cells(torch.arange(self.num_envs))
        self._validate_orbit_symmetry()

    def migrate_state_dict(
        self,
        state: Mapping,
        *,
        warn_migration: bool = True,
        legacy_gait_prior_weights: torch.Tensor | Sequence[float] | None = None,
    ) -> Mapping:
        """Return current-schema persistent state, migrating V5 schema 1 when needed.

        Schema 1 stored per-environment command segments and runtime state. The
        migration intentionally retains only global competence tensors and the
        curriculum RNG. Legacy ``weights`` become priorities; legacy unlocked
        cells become eligible, and become mastered only when their saved EWMA
        and visit count satisfy the new thresholds.
        """
        if not isinstance(state, Mapping):
            raise ValueError("Curriculum checkpoint state must be a mapping.")
        schema_version = state.get("schema_version")
        if type(schema_version) is int and schema_version == self._STATE_SCHEMA_VERSION:
            return state
        if type(schema_version) is not int or schema_version != self._LEGACY_STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported curriculum checkpoint schema: "
                f"expected {self._STATE_SCHEMA_VERSION} or legacy {self._LEGACY_STATE_SCHEMA_VERSION}, "
                f"received {schema_version!r}."
            )
        if warn_migration:
            warnings.warn(
                "Migrating command-curriculum schema 1 global competence state; stale per-environment cells, "
                "partial tracking sums, commands, gaits, timers, and phases are discarded.",
                UserWarning,
                stacklevel=2,
            )
        return self._migrate_v1_state_dict(state, legacy_gait_prior_weights=legacy_gait_prior_weights)

    def _config_state(self) -> dict:
        return {
            "gait_partner_indices": self.gait_partner_indices.tolist(),
            "velocity_bin_edges": self.velocity_bin_edges.tolist(),
            "ewma_coefficient": self.ewma_coefficient,
            "unlock_threshold": self.unlock_threshold,
            "current_cell_increment": self.current_cell_increment,
            "neighbor_increment": self.neighbor_increment,
            "exploration_floor": self.exploration_floor,
            "maximum_weight": self.maximum_weight,
            "initial_max_abs_speed": self.initial_max_abs_speed,
            "minimum_visits": self.minimum_visits,
            "locked_cell_weight": self.locked_cell_weight,
            "seed": self.seed,
        }

    def _migrate_v1_state_dict(
        self,
        state: Mapping,
        *,
        legacy_gait_prior_weights: torch.Tensor | Sequence[float] | None,
    ) -> dict:
        required = {
            "schema_version",
            "config",
            "gait_prior_weights",
            "weights",
            "ewma_success",
            "visit_counts",
            "unlock_state",
            "rng_state",
        }
        if not required.issubset(state):
            raise ValueError(
                f"Legacy curriculum checkpoint fields do not match: missing={sorted(required - set(state))}."
            )
        legacy_config = state["config"]
        if not isinstance(legacy_config, Mapping):
            raise ValueError("Legacy curriculum checkpoint config must be a mapping.")
        current_config = self._config_state()
        shared_config_names = (
            "gait_partner_indices",
            "velocity_bin_edges",
            "ewma_coefficient",
            "unlock_threshold",
            "current_cell_increment",
            "neighbor_increment",
            "exploration_floor",
            "maximum_weight",
            "seed",
        )
        mismatched = {
            name: (current_config[name], legacy_config.get(name))
            for name in shared_config_names
            if not _type_exact_equal(legacy_config.get(name), current_config[name])
        }
        if mismatched:
            raise ValueError(
                "Legacy curriculum checkpoint configuration does not match the active command configuration: "
                f"{mismatched!r}."
            )

        expected_grid_shape = (self.num_gaits, self.velocity_bin_count)
        gait_prior = self._legacy_tensor(state, "gait_prior_weights", (self.num_gaits,), torch.float64)
        priority = self._legacy_tensor(state, "weights", expected_grid_shape, torch.float64)
        ewma_success = self._legacy_tensor(state, "ewma_success", expected_grid_shape, torch.float64)
        visit_counts = self._legacy_tensor(state, "visit_counts", expected_grid_shape, torch.long)
        legacy_unlocked = self._legacy_tensor(state, "unlock_state", expected_grid_shape, torch.bool)
        if torch.any(~torch.isfinite(gait_prior)) or torch.any(gait_prior < 0.0):
            raise ValueError("Legacy curriculum gait_prior_weights must be finite and nonnegative.")
        if torch.any(~torch.isfinite(priority)) or torch.any(priority < 0.0):
            raise ValueError("Legacy curriculum weights must be finite and nonnegative.")
        if torch.any(~torch.isfinite(ewma_success)) or torch.any((ewma_success < 0.0) | (ewma_success > 1.0)):
            raise ValueError("Legacy curriculum ewma_success must be finite and in [0, 1].")
        if torch.any(visit_counts < 0):
            raise ValueError("Legacy curriculum visit_counts must be nonnegative.")

        # Schema 1 floored every configured gait prior, so its serialized
        # tensor cannot distinguish a true positive row from a configured
        # zero-prior row. Intersect with the active prior, which the command
        # wrapper resolves at the saved training iteration before migration.
        active_gait_prior = (
            self.gait_prior_weights
            if legacy_gait_prior_weights is None
            else torch.as_tensor(legacy_gait_prior_weights, dtype=torch.float64, device="cpu")
        )
        if active_gait_prior.shape != gait_prior.shape:
            raise ValueError(
                "Active legacy gait-prior support must match the saved gait-prior shape; "
                f"received {tuple(active_gait_prior.shape)} and {tuple(gait_prior.shape)}."
            )
        active_positive_prior = active_gait_prior > 0.0
        gait_prior = torch.where(active_positive_prior, gait_prior, torch.zeros_like(gait_prior))
        positive_prior = gait_prior[:, None] > 0.0
        initial = positive_prior & (
            torch.abs(self.velocity_bin_centers)[None, :] <= self.initial_max_abs_speed + 1.0e-12
        )
        eligible = (initial | legacy_unlocked) & positive_prior
        mastered = (
            legacy_unlocked & eligible & (visit_counts >= self.minimum_visits) & (ewma_success >= self.unlock_threshold)
        )
        priority = priority.clamp(min=0.0, max=self.maximum_weight)
        base_priority = (
            gait_prior[:, None].expand_as(priority).clamp(min=self.exploration_floor, max=self.maximum_weight)
        )
        priority = torch.where(eligible, torch.maximum(priority, base_priority), priority)
        priority = torch.where(positive_prior, priority, torch.zeros_like(priority))
        rng_state = torch.as_tensor(state["rng_state"], dtype=torch.uint8, device="cpu").clone()
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "config": current_config,
            "gait_prior_weights": gait_prior.clone(),
            "priority": priority,
            "eligible": eligible,
            "mastered": mastered,
            "ewma_success": ewma_success.clone(),
            "visit_counts": visit_counts.clone(),
            "rng_state": rng_state,
        }

    def _orbit_cells(self, gait_index: int, velocity_bin_index: int) -> list[tuple[int, int]]:
        partner = (
            int(self.gait_partner_indices[gait_index]),
            int(self.velocity_partner_indices[velocity_bin_index]),
        )
        cell = (gait_index, velocity_bin_index)
        return [cell] if partner == cell else [cell, partner]

    def _increment_orbit_priority(self, gait_index: int, velocity_bin_index: int, increment: float) -> None:
        for orbit_gait, orbit_bin in self._orbit_cells(gait_index, velocity_bin_index):
            self.priority[orbit_gait, orbit_bin] = min(
                float(self.priority[orbit_gait, orbit_bin]) + increment,
                self.maximum_weight,
            )

    def _activate_initial_support(self) -> None:
        speed_eligible = torch.abs(self.velocity_bin_centers) <= self.initial_max_abs_speed + 1.0e-12
        if not torch.any(speed_eligible):
            raise ValueError(
                "initial_max_abs_speed does not include any velocity-bin center; increase it or change the grid."
            )
        initial = (self.gait_prior_weights[:, None] > 0.0) & speed_eligible[None, :]
        newly_eligible = initial & ~self.eligible
        self.eligible |= initial
        base_priority = (
            self.gait_prior_weights[:, None]
            .expand_as(self.priority)
            .clamp(
                min=self.exploration_floor,
                max=self.maximum_weight,
            )
        )
        self.priority[newly_eligible] = base_priority[newly_eligible]

    def _activate_orbit(self, gait_index: int, velocity_bin_index: int, increment: float) -> None:
        orbit = self._orbit_cells(gait_index, velocity_bin_index)
        if any(self.gait_prior_weights[orbit_gait] <= 0.0 for orbit_gait, _ in orbit):
            return
        for orbit_gait, orbit_bin in orbit:
            if not self.eligible[orbit_gait, orbit_bin]:
                self.eligible[orbit_gait, orbit_bin] = True
                self.priority[orbit_gait, orbit_bin] = min(
                    max(float(self.gait_prior_weights[orbit_gait]), self.exploration_floor),
                    self.maximum_weight,
                )
            self.priority[orbit_gait, orbit_bin] = min(
                float(self.priority[orbit_gait, orbit_bin]) + increment,
                self.maximum_weight,
            )

    def _next_larger_velocity_bins(self, velocity_bin_index: int) -> list[int]:
        center = float(self.velocity_bin_centers[velocity_bin_index])
        centers = self.velocity_bin_centers
        tolerance = 1.0e-12
        if center > tolerance:
            candidates = torch.nonzero(centers > center + tolerance, as_tuple=False).flatten()
        elif center < -tolerance:
            candidates = torch.nonzero(centers < center - tolerance, as_tuple=False).flatten()
        else:
            candidates = torch.nonzero(torch.abs(centers) > tolerance, as_tuple=False).flatten()
        if len(candidates) == 0:
            return []
        candidate_magnitudes = torch.abs(centers[candidates])
        next_magnitude = candidate_magnitudes.min()
        selected = candidates[torch.isclose(candidate_magnitudes, next_magnitude, rtol=0.0, atol=tolerance)]
        return selected.tolist()

    def _validate_orbit_symmetry(self) -> None:
        if not torch.equal(self.gait_prior_weights, self.gait_prior_weights[self.gait_partner_indices]):
            raise ValueError("Curriculum gait_prior_weights are not equal across time-reversal gait partners.")
        partner_gaits = self.gait_partner_indices[:, None].expand_as(self.priority)
        partner_velocity = self.velocity_partner_indices[None, :].expand_as(self.priority)
        for name in ("priority", "eligible", "mastered", "ewma_success", "visit_counts"):
            value = getattr(self, name)
            if not torch.equal(value, value[partner_gaits, partner_velocity]):
                raise ValueError(f"Curriculum {name} is not equal across time-reversal orbit partners.")
        if torch.any(~torch.isfinite(self.priority)) or torch.any(
            (self.priority < 0.0) | (self.priority > self.maximum_weight)
        ):
            raise ValueError("Curriculum priority must be finite and in [0, maximum_weight].")
        if torch.any(~torch.isfinite(self.ewma_success)) or torch.any(
            (self.ewma_success < 0.0) | (self.ewma_success > 1.0)
        ):
            raise ValueError("Curriculum ewma_success must be finite and in [0, 1].")
        if torch.any(self.visit_counts < 0):
            raise ValueError("Curriculum visit_counts must be nonnegative.")
        if torch.any(self.mastered & ~self.eligible):
            raise ValueError("Curriculum mastered cells must remain eligible.")
        zero_prior = self.gait_prior_weights[:, None] <= 0.0
        if torch.any(self.eligible & zero_prior):
            raise ValueError("Curriculum cells with zero gait prior must remain ineligible.")
        if torch.any(self.priority[self.eligible] < self.exploration_floor):
            raise ValueError("Eligible curriculum priority must satisfy exploration_floor.")

    def _clear_segments(self, env_ids: torch.Tensor) -> None:
        self.segment_error_xy_sum[env_ids] = 0.0
        self.segment_error_yaw_sum[env_ids] = 0.0
        self.segment_step_count[env_ids] = 0

    @staticmethod
    def _legacy_tensor(
        state: Mapping,
        name: str,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(state[name], device="cpu")
        if tensor.shape != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"Legacy curriculum {name} must have shape {expected_shape} and dtype {expected_dtype}; "
                f"received {tuple(tensor.shape)} and {tensor.dtype}."
            )
        return tensor.clone()

    @staticmethod
    def _validate_sampling_support(weights: torch.Tensor, description: str) -> None:
        if not torch.isfinite(weights).all() or torch.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError(f"No finite positive sampling support is available for {description}.")

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
