.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Disabled the leg-permutation symmetry reward by default for the Go2 and X1
  symmetric quadruped tasks by setting its weight to zero. Set
  ``env.rewards.leg_permutation_symmetry.weight=0.2`` when launching training
  to restore the previous reward weight.
* Disabled the torque-smoothness reward by default for the Go2 and X1 symmetric
  quadruped tasks. Set ``env.rewards.smoothness.weight=0.1`` when launching
  training to restore its previous weight.
* Changed :func:`~isaaclab_tasks.manager_based.locomotion.velocity.mdp.symm_quadruped.action_rate_exp_penalty`
  to use the L1 norm of action changes. Changed its configured exponential scale
  to 4.0 while retaining the ``action_rate_l2`` key and reward weight of 0.05.
  Retrain policies to use the revised reward shaping; custom callers should
  select their scale for summed absolute action changes.
