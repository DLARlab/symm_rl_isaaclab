Changed
^^^^^^^

* Changed Go2 and X1 symmetric locomotion tasks to use a unified reward
  configuration with identical functions and coefficients and shared defaults
  based on X1's rewards, including joint-symmetry normalization. Robot configs
  supply joint and contact bindings, joint-axis conventions, and height targets.
  Configure common reward settings in ``symm_quadruped/flat_env_cfg.py``;
  retrain Go2 policies to use the updated rewards in TRS and no-TRS runs.
  Existing explicit reward overrides and X1 compatibility functions remain
  available.
* Changed the Go2 base-height target to 0.30 m and peak swing-foot clearance to
  0.08 m. X1 retained its 0.35 m and 0.10 m targets. To restore the previous Go2
  targets, set ``base_height_target=0.35`` and ``foot_clearance_height=0.10`` in
  the Go2 call to ``configure_rewards``.
