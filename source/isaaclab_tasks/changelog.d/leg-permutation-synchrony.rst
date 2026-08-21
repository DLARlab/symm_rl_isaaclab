Added
^^^^^

* Added a deterministic six-gait sequence for symmetric-quadruped play and
  recording, with configurable per-gait duration and a random-sampling opt-out.
* Added optional weighted gait-row sampling for symmetric-quadruped commands.

Changed
^^^^^^^

* Changed the symmetric-quadruped leg-permutation reward to constrain only
  leg pairs whose sampled phase offsets are within ``0.02`` cycles and to
  normalize over those active pairs. Retrain policies because this replaces
  the former Gaussian phase weighting.
* Changed symmetric-quadruped training to use a fore/hind-balanced,
  time-reversal-closed ten-row gait library with small phase-offset noise.
  Play and recording retain six rows but now use one front-spread and one
  hind-spread half-bound. Use an archived environment configuration to
  reproduce earlier gait libraries and playback sequences.
* Changed symmetric-quadruped training to retain one velocity resample at
  ``10`` s, the existing velocity disturbance at ``15`` s, and one gait-row
  resample at ``20`` s during each ``30`` s episode. Set the corresponding
  once-after-reset option to ``False`` for recurring timer resampling.
* Changed play and recording to disable phase-offset noise while retaining
  period noise and one velocity resample at ``10`` s across the six-gait
  sequence. Set ``resample_once_after_reset`` to ``False`` for recurring
  velocity resampling, or disable the gait sequence for random gait sampling.
* Changed the default symmetric-quadruped hip-action and leg-permutation
  weights from ``0.15`` and ``0.30`` to ``0.10`` and ``0.20``, respectively.
  The summed foot-phase weight remains ``0.30``. Restore ``0.15`` and ``0.30``
  to reproduce the preceding hip-action and leg-permutation coefficients.

Fixed
^^^^^

* Fixed period and duty factor remaining stale after a velocity-only resample.
  Velocity and gait resampling now both refresh timing from the final command;
  coincident changes share one refresh to avoid duplicate timing-noise draws.
  The piecewise-integrated common phase stays continuous across timing changes,
  completed transitions use their original desired signal for reward and
  recording, and the next observation receives the new signal atomically.
  Retrain policies for the new phase observations, or use an archived
  environment implementation to reproduce earlier checkpoints.
* Fixed the renamed foot-phase penalty accidentally averaging over feet by
  restoring the summed four-foot scale used by existing reward coefficients.

Deprecated
^^^^^^^^^^

* Deprecated the ``foot_periodicity`` reward name and
  ``foot_periodicity_penalty`` function in favor of ``foot_phase`` and
  ``foot_phase_penalty``. Existing configurations continue to work through
  compatibility aliases during the deprecation period.
