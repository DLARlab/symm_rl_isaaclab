Added
^^^^^

* Added independent time-reversal policy, value, and filtered trajectory
  supervision controls with schedules, validity masks, gradient diagnostics,
  checkpoint state, and matched-cohort provenance.
* Added optional foot-phase reductions, requested-target overflow penalties,
  per-joint feasible actor-mean bounds, normalized-target consistency, and
  gait-sampling curricula for symmetric quadrupeds.
* Added light and full symmetric-quadruped evaluation protocols,
  result-independent cohort registries, resumable study launch, and a
  platform-neutral isolated-process command scheduler.

Changed
^^^^^^^

* Changed symmetric-locomotion playback to load checkpoints according to the
  RSL-RL version and runner type. Existing on-policy, distillation, and pre-v4
  checkpoint paths remain supported.
* Changed study and evaluation provenance to bind resolved commands, protocol
  facts, input artifacts, and hashes while keeping new training and evaluation
  behavior opt-in or explicitly versioned.

Deprecated
^^^^^^^^^^

* Deprecated ``use_data_augmentation=True`` for
  :class:`~isaaclab_tasks.manager_based.locomotion.velocity.config.symm_quadruped.time_reversal_ppo.TimeReversalPPO`
  in favor of filtered ``tr_augmentation`` supervision. The legacy PPO
  minibatch-duplication behavior remains active during the deprecation window.
* Deprecated the ``analyze_leg_usage`` scripts and CLI command in favor of the
  broader ``evaluation`` entry points. Existing commands continue to run the
  legacy full-grid protocol.
* Deprecated the ``compare.py``, ``compare.sh``, and ``compare.ps1``
  convenience launchers. Use the ``compare`` subcommand of
  ``symm_locomotion.ps1`` or ``symm_locomotion.sh`` instead.
