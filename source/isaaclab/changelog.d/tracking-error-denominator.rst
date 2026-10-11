Changed
^^^^^^^

* Changed :func:`~isaaclab.envs.mdp.track_lin_vel_xy_exp` and
  :func:`~isaaclab.envs.mdp.track_ang_vel_z_exp` to use ``std`` directly as the
  squared-error denominator. To preserve the previous reward curve, pass the
  square of the previously configured ``std`` value.
