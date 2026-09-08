Changed
^^^^^^^

* Disabled the composite ``straight_line_motion`` reward in symmetric quadruped
  tasks to avoid duplicating the independently configured tracking, roll, and
  base-height objectives. Existing overrides should use the individual reward
  terms instead.
* Changed the symmetric quadruped linear-velocity tracking weight to 0.5 and
  disabled the alive-bonus and termination-penalty rewards. Retrain policies
  to use the rebalanced reward scales.
* Changed the configured symmetric quadruped base-roll term to use the bounded
  negative ``base_roll_exp_penalty`` function. Direct callers that require the
  positive tracking score can continue using ``base_roll_exp``.
* Changed symmetric quadruped reward aggregation from positive clipping to the
  Ji22 composition, which exponentially scales positive rewards by accumulated
  negative rewards before adding the termination term. Retrain policies because
  the policy-facing reward semantics have changed.
* Changed the symmetric quadruped command curriculum to use 21 x-velocity bins,
  one y-velocity bin, and 21 yaw-rate bins while retaining progressive bin
  unlocking. Set ``curriculum.enabled`` to ``False`` to sample uniformly from
  every bin instead.
