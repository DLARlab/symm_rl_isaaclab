.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Changed :func:`~isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped.leg_permutation_symmetry_penalty`
  to penalize summed absolute logical joint-angle differences with an exponential
  coefficient of 3 [rad^-1], retaining the default reward weight of 0.4 and the
  existing phase-based pair selection. Retrain policies to use the revised
  reward shaping. The ``joint_ranges`` argument was retained for compatibility
  and no longer affects the penalty.
