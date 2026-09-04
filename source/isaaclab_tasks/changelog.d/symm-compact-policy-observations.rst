Added
^^^^^

* Added configurable frame-major policy observation history for symmetric
  quadruped tasks, defaulting to 20 frames, while retaining a feed-forward
  policy network.

Changed
^^^^^^^

* Changed the symmetric quadruped PhysX pair-buffer capacities to the validated
  2048-environment training values to reduce GPU memory usage.
* Changed the initial symmetric quadruped policy action standard deviation from
  0.5 to 1.0 and removed its upper clamp to increase exploration.
* Changed the symmetric quadruped leg-permutation reward to compare only leg
  pairs commanded in synchrony and average over the active pairs, and reduced
  its default weight from 0.30 to 0.20. Retrain policies that depend on the
  former smooth phase weighting.
* **Breaking:** Changed each symmetric quadruped policy observation frame from
  72 to 56 dimensions by removing constant-zero values and base angular
  velocity, and by replacing the sine/cosine foot phase-offset encoding with
  four canonical raw phase offsets. Changed the default policy input to 20
  frames (1120 dimensions). Retrain policies and update deployment observation
  layouts before loading new checkpoints.
* **Breaking:** Changed historical time-reversal regularization to reverse the
  complete frame sequence, realign embedded previous actions, and compare the
  reversed policy distribution mean with the corresponding earlier rollout
  state. Retrain policies that used the former frame-independent transform.

Fixed
^^^^^

* Fixed frame-major observation history mutating policy observations retained
  by the rollout runner during an environment step.
* Fixed historical time reversal reflecting the foot-phase sine after the
  observation sequence had already been reversed.
