Changed
^^^^^^^

* Disabled the composite ``straight_line_motion`` reward in symmetric quadruped
  tasks to avoid duplicating the independently configured tracking, roll, and
  base-height objectives. Existing overrides should use the individual reward
  terms instead.
* Changed the symmetric quadruped linear-velocity tracking weight from 1.0 to
  0.5 and the alive-bonus weight from 0.2 to 1.0. Retrain policies to use the
  rebalanced reward scales.
