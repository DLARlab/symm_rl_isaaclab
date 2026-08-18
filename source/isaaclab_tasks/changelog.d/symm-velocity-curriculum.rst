Added
^^^^^

* Added independent x-velocity, y-velocity, yaw-rate, and base-roll rewards to
  symmetric quadruped locomotion tasks.

Changed
^^^^^^^

* Changed symmetric quadruped forward-velocity curricula to start at
  ``[-0.5, 0.5]`` m/s, expand in 0.5 m/s bins, and treat commands up to
  0.1 m/s as standing commands.
* Changed symmetric quadruped command transitions to resample velocity, gait,
  or both at equal probability every 10 seconds while preserving continuous
  gait phase.
* Changed symmetric quadruped rewards to stop pulling the robot toward its
  initial world-y position and zero heading during omnidirectional commands.
* Changed the symmetric quadruped sagittal-plane policy observation to a
  three-zero placeholder, preserving checkpoint dimensions without exposing
  simulated world position or heading.
* Changed velocity curriculum advancement to use the completed command window's
  mean linear and yaw tracking rewards, matching the Walk These Ways threshold
  method. Configure the new ``curriculum_tracking_*`` fields instead of the
  deprecated ``vel_*_success_*`` fields.
* Fixed command-transition boundary rewards so they are evaluated against the
  command that produced the transition before the next command is sampled.
