Changed
^^^^^^^

* Changed symmetric Go2 and X1 reward aggregation to clip the full weighted,
  timestep-scaled sum at zero, including any termination reward. Commented out
  Ji22 composition and its diagnostics. Retrain policies for the new reward;
  restore the commented Ji22 setup and composition in ``env.py`` to reproduce
  the previous aggregation.

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

* Changed foot clearance to penalize each foot's shortfall below a sinusoidal
  swing trajectory without command gating, using a 0.025 m shortfall scale
  and outer weight 0.1. Retained peak foot-link heights of 0.082 m for Go2 and
  0.1075 m for X1. Set ``env.rewards.foot_clearance.params.min_height`` to
  customize the peak. The clearance helper retained its legacy command gate
  by default; the new task configuration disabled it with
  ``use_command_gate=False``. Select ``foot_clearance_mode="speed_tracking_reward"``
  in ``configure_rewards`` to reproduce the previous positive clearance reward.

* Changed force and velocity periodicity to apply exponential-minus-one
  penalties independently to each foot before summing, with phase gates
  outside the exponentials. Set coefficients to 0.05 and 2.0, respectively,
  and both outer weights to 0.1. The combined weighted periodicity penalty
  was bounded by [-0.4, 0]. To restore the previous positive formulation,
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
  the swing-trajectory clearance penalty. Use ``mean_force_penalty`` and
  ``mean_speed_penalty`` for the current CSV format; previous penalty and
  positive-reward recordings remained supported.
