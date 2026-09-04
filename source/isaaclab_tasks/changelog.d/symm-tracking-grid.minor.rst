Added
^^^^^

* Added a headless batched velocity-command tracking evaluation that writes
  per-command X, Y, XY, and yaw mean absolute errors to CSV.

Changed
^^^^^^^

* Changed the Dobot X1 command curriculum to cover X velocities up to 3 m/s
  and yaw rates up to 2 rad/s, and to expand when linear and yaw tracking
  rewards exceed 0.85.
* Changed Dobot X1 yaw-rate tracking to use a 0.5 rad/s error scale, matching
  the Walk These Ways reward denominator of 0.25. Retrain policies to use the
  widened yaw-tracking reward.
* Changed the Dobot X1 runtime calf joint limits to preserve the intended front
  and rear bending branches without modifying the source URDF. Retrain policies
  against the constrained task configuration.
