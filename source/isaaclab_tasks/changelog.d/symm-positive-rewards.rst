Fixed
^^^^^

* Fixed direction-test playback to apply the configured gait period and stance
  duty factor at command changes, matching the training timing implementation.
  Applied timing immediately at startup and after resets, and used smooth
  transitions for later command changes without restarting the gait clock.

* Fixed time-reversal PPO to skip critic consistency computation when its
  coefficient was zero, while retaining ordinary PPO critic training and
  reporting a zero ``tr_value`` loss.

Changed
^^^^^^^

* Changed symmetric locomotion TRS run names to append the value and actor
  consistency coefficients, including custom ``--run-name`` labels. Update
  explicit selectors for new runs to include the
  ``_valuecoef<V>_actorcoef<A>`` suffix. Runs using ``--no-trs`` retained
  their previous naming.

* Changed the default TRS loss warmup to zero for symmetric Go2 and X1 training.
  Invalid or incomplete history pairs remained masked, so the first fresh
  rollout with 24 steps and 20 history frames contributed zero TRS loss.
  Use ``--tr-warmup-iterations 500`` or set
  ``agent.algorithm.symmetry_cfg.warmup_iterations=500`` to restore the previous
  warmup delay.

* Changed training and playback to default to fixed cycle durations of 0.52 s
  for Go2 and 0.60 s for X1, with stance duty factor 0.56 for both robots.
  Retained the ``calculate_from_sampling_curve`` selector: false used fixed
  values and true used the velocity-based curve. To restore the previous
  velocity-based timing, enable ``calculate_from_sampling_curve`` and,
  optionally, ``add_noise_period`` in the command configuration.

* Enabled the leg-permutation symmetry penalty with weight 0.1 for symmetric
  Go2 and X1. Set ``rewards.leg_permutation_symmetry.weight=0.0`` to restore
  the previous disabled setting.

* Changed the base-height reward targets to 0.30 m for symmetric Go2 and
  0.35 m for X1. To restore the previous higher targets, set
  ``rewards.base_height.params["target_height"]`` to ``0.35`` for Go2 or
  ``0.40`` for X1.

* Changed all symmetric Go2 joint PD gains to stiffness 25 N m/rad and
  damping 0.6 N m s/rad. To restore the previous thigh and calf gains, set
  ``scene.robot.actuators["thigh_legs"]`` and
  ``scene.robot.actuators["calf_legs"]`` stiffness to ``30.0`` and damping
  to ``0.65``.

* Changed the Go2 command-velocity curriculum to match X1, with maximum
  forward/backward speeds of 3 m/s, maximum yaw rates of 2 rad/s, and a linear
  tracking success threshold of 0.85. Retained the yaw threshold of 0.7.
  To restore the previous Go2 settings, override
  ``commands.base_velocity.ranges.lin_vel_x`` and ``ang_vel_z`` to
  ``(-4.0, 4.0)`` and ``curriculum_tracking_lin_vel_threshold`` to ``0.9``.

* Restored Ji22 reward composition for symmetric Go2 and X1, including the
  Go2-bend X1 variant, with negative-reward scale 0.02 and termination rewards
  added separately. Restored the backup's stability-gated torque-difference
  penalty with weight 0.1 and exponential coefficient 0.1, alongside the existing
  torque-magnitude penalty. Retrain policies to evaluate the restored reward;
  set ``rewards.smoothness.weight=0.0`` to disable torque-difference smoothness.
  Restored Ji22 diagnostics and distinguished negative signed reward sums from
  clipping in the training diagnostics.

* Changed the symmetric Go2 and X1 tasks to use nine bounded exponential
  penalties. Hip and joint-limit penalties now used measured positions;
  joint limits allowed the central 90% of each full runtime range. Added
  vertical velocity, joint velocity, torque, action smoothness, roll/pitch
  angular velocity, and control-step joint acceleration penalties. Existing
  reward keys remained supported. To retain previous behavior, explicitly
  select the legacy hip-action, action-rate, and joint-target-limit helpers
  and their previous parameters and weights. Configure the new penalties
  through ``scale`` for the exponential coefficient and ``weight`` for
  their outer weight; use ``dof_vel_l2`` and ``action_smoothness`` for the
  newly introduced keys.

* Changed the symmetric Go2 and X1 tasks to use four positive exponential
  reward terms with weights totaling 1.0 before timestep scaling: planar
  tracking 0.4, yaw tracking 0.3, height 0.15, and roll/pitch 0.15. Added
  roll-and-pitch tracking and restored a swing-trajectory clearance penalty.
  Split foot periodicity into per-foot force and velocity penalties.
  Update reward overrides and log consumers to use ``base_roll_pitch_exp``,
  ``foot_periodicity_force``, and ``foot_periodicity_velocity`` instead of the
  previously active ``base_roll_exp`` and ``foot_periodicity`` terms. The
  previous reward helpers remained available for explicit configurations.

* Changed foot clearance to reuse the force-periodicity swing gate and penalize
  each foot's absolute height error from a squared-sine swing trajectory
  without command gating, using a 0.025 m error scale and outer weight 0.1.
  The trajectory started and ended at the
  resting foot offset with zero slope at both swing boundaries. The periodic
  gate remained continuous across phase wrap. Retained peak foot-link heights
  of 0.082 m for Go2 and 0.1075 m for X1. Set
  ``env.rewards.foot_clearance.params.min_height`` to
  customize the peak and ``foot_height_offset`` to customize the baseline.
  Set ``use_absolute_error=False`` and ``foot_height_offset=0.0`` to restore
  the previous shortfall-only error and zero baseline with the new trajectory
  and gate. Retrain policies to evaluate the new swing trajectory and gate.
  The helper retained those legacy defaults and command gating, while the
  active task configuration selected
  absolute error with robot-specific offsets and ``use_command_gate=False``.
  Select ``foot_clearance_mode="speed_tracking_reward"``
  in ``configure_rewards`` to reproduce the previous positive clearance reward.

* Changed force and velocity periodicity to apply exponential-minus-one
  penalties independently to each foot before summing, with phase gates
  outside the exponentials. Set coefficients to 0.05 and 2.0, respectively,
  and both outer weights to 0.3. The combined weighted periodicity penalty
  was bounded by [-1.2, 0]. Set both outer weights to 0.1 to restore their
  previous strength. To restore the previous positive formulation,
  select ``foot_periodicity_force_exp`` and ``foot_periodicity_velocity_exp``
  explicitly with their previous coefficients and weights. Playback penalty
  columns used ``mean_force_penalty`` and ``mean_speed_penalty``; readers
  of previous positive-reward recordings could continue using the cost columns.

* Changed the action-rate and action-smoothness exponential coefficients to
  0.05 for Go2 and X1, keeping both outer reward weights at 0.05. Set
  ``env.rewards.action_rate_l2.params.scale=1.0`` and
  ``env.rewards.action_smoothness.params.scale=0.4`` explicitly to retain
  the previous coefficients.

* Changed playback diagnostics to report per-foot periodicity penalties and
  the swing-trajectory clearance penalty and height errors. Use
  ``foot_clearance_height_errors`` for the error used by the clearance penalty;
  ``foot_clearance_shortfalls`` continued to report underlifting only. Use
  ``mean_force_penalty`` and ``mean_speed_penalty`` for the current CSV format; previous penalty and
  positive-reward recordings remained supported.
