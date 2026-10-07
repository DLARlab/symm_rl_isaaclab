.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Changed the default planar velocity and yaw-rate tracking reward error scales
  for Go2 and X1 symmetric locomotion tasks from 0.5 to 0.2 m/s and rad/s,
  respectively. Retrain policies to use the stricter tracking rewards. To restore
  the previous scales, set ``env.rewards.track_lin_vel_xy_exp.params.std=0.5``
  and ``env.rewards.track_ang_vel_z_exp.params.error_scale=0.5``.
