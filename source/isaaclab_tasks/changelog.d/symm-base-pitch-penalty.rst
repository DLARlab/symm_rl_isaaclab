Changed
^^^^^^^

* Changed the existing ``base_roll_exp`` reward term in symmetric quadruped tasks
  to penalize absolute roll around zero using
  ``exp(-abs(roll) / error_scale) - 1`` without a pitch penalty. Retrain policies to use
  the updated orientation objective; existing reward names and configuration
  overrides remain valid.
* Changed Dobot X1 joint damping from 0.65 to 1.2 N·m·s/rad while retaining
  joint stiffness of 30 N·m/rad. Retrain policies for the updated actuator dynamics.
* Changed the symmetric quadruped base-height penalty to track a fixed height
  of 0.35 m using ``exp(-20 * abs(0.35 - base_height)) - 1`` with weight 0.3.
  The reward's ``height_range`` parameter is
  retained but no longer controls the height objective. Retrain policies to
  use the new target height.
* Changed the Dobot X1 peak swing-clearance target from 0.04 m to 0.10 m.
  Retrain policies to use the higher target.

Fixed
^^^^^

* Fixed the swing-clearance penalty being disabled for lateral and pure yaw
  commands and reduced for diagonal commands. The gate now uses the maximum
  of normalized planar speed and absolute yaw rate, with full activation at
  0.20 m/s or 0.20 rad/s. The yaw threshold can be configured independently
  through the reward's ``min_command_yaw_rate`` parameter.
