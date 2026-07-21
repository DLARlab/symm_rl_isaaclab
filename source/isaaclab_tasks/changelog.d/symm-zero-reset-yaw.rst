Added
^^^^^

* Added action, joint-limit, reward-component, reward-clipping, swing-clearance,
  per-leg applied-torque, mechanical-power, and filtered ground-reaction-force
  diagnostics to symmetric quadruped playback.

Changed
^^^^^^^

* Changed symmetric Go2 and Dobot X1 training to use a fixed zero yaw-rate
  command and positively reward commanded x motion only while the robot remains
  straight and supported. Restored the hip-action penalty to prevent splayed-leg
  reward exploits. Retrain policies to use the updated reward behavior.
* Changed the ``morphological_symmetry`` reward term to
  ``leg_permutation_symmetry``. Update Hydra overrides to use the new reward
  name.

Deprecated
^^^^^^^^^^

* Deprecated ``morphological_symmetry_penalty`` in favor of
  ``leg_permutation_symmetry_penalty``. The old function and reward-config name
  remain compatibility aliases during the deprecation period.

Fixed
^^^^^

* Fixed symmetric Go2 and Dobot X1 training to use zero lateral velocity during
  resets and periodic pushes, and zero yaw angular velocity during resets,
  preventing artificial direction drift.
* Fixed symmetric Go2 and Dobot X1 training collapse by preserving signed
  recovery rewards and terminal penalties, bounding policy exploration noise,
  and preserving the unclipped action behavior of known-good runs. Restart
  training rather than resuming a policy whose action standard deviation has
  diverged.
* Fixed slow symmetric Go2 and Dobot X1 training by restoring the proven PhysX
  aggregate-pair capacity, defaulting generic task launches to 256 environments,
  and disabling unused contact air-time tracking.
* Fixed the symmetric quadruped straight-line reward so observable lateral
  motion and posture terms cannot suppress the forward-tracking signal, while
  collapsed pitch and height retain a bounded support loss.
* Fixed symmetric quadruped foot clearance so commanded swing trajectories are
  ground-relative and remain active before the robot reaches its target speed.
* Fixed Dobot X1 face-down crawling by clamping unsafe joint targets and
  terminating low front-body postures, and restored its conservative clearance
  shaping. Retrain the policy from a stable checkpoint or a fresh policy.
* Fixed low Unitree Go2 swing-foot clearance with a positive clearance and
  no-contact reward. Retrain the policy to use the updated reward behavior.
* Fixed GIF recording so a failed playback cannot silently convert an MP4 from
  an earlier run.
