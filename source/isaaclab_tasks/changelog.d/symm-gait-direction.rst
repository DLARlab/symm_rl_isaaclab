.. Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Changed symmetric quadruped gait direction to prioritize forward velocity,
  then lateral velocity, then yaw rate, using deadbands of 0.05 m/s and
  0.05 rad/s. Near-zero commands retain the previous direction within an
  episode. Clock advancement and foot offsets use the same direction.
  Retrain policies using lateral or yaw commands near zero forward velocity
  to adopt the new gait convention.
