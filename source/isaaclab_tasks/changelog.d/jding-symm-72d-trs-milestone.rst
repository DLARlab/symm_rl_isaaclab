Changed
^^^^^^^

* Changed symmetric quadruped backward commands to retain the forward-time gait
  schedule and corrected auxiliary time-reversal phase features to use a
  duty-aware reflection. Existing 72D checkpoints with the former
  negative-command phase semantics require their archived environment code and
  configuration.
* Changed symmetric quadruped command sampling to retain low-speed commands by
  disabling the XY command deadband. Set ``min_xy_command_norm=0.2`` to restore
  the former near-zero command snapping.
* Changed symmetric quadruped training defaults to 20,000 iterations and 512
  environments. Pass ``--iterations`` and ``--num-envs`` to restore a previous
  launcher scale.
* Changed reset base-velocity randomization to include lateral and yaw velocity
  in ``(-0.5, 0.5)`` and periodic pushes to include lateral velocity in
  ``(-0.25, 0.25)``. Override the corresponding event velocity ranges to
  restore axis-specific zero disturbances.

Added
^^^^^

* Added manifest-driven matched TRS study analysis and the Phase Mapping V2
  Go2/X1 milestone reports.
