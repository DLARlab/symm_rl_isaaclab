Added
^^^^^

* Added native 30-frame proprioceptive observation history, history-aware
  actor/value time-reversal consistency, and an optional time-reversal-orbit
  command competence curriculum for symmetric quadruped tasks.

Changed
^^^^^^^

* **Breaking:** Changed the active symmetric-quadruped policy observation from
  the archived 72-dimensional contract to the versioned 64-dimensional
  ``hardware_proprio_history_64d_v1`` contract. Retrain policies with matching
  ``--history`` or ``--no-history`` settings; 72-dimensional checkpoints are
  rejected rather than adapted.
* Changed the scalar straight-line motion reward to exclude absolute world
  lateral-position and heading recovery. Use the simulator-only diagnostics
  when those world-pose measurements are needed for analysis.
