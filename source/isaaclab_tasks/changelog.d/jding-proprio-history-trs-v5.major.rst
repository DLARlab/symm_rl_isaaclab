Added
^^^^^

* Added native 30-frame proprioceptive observation history, transition-aligned
  causal-history actor/value time-reversal consistency, and an optional staged
  time-reversal-orbit command competence curriculum for symmetric quadruped
  tasks.

Changed
^^^^^^^

* **Breaking:** Changed the active symmetric-quadruped policy observation from
  the archived 72-dimensional contract to the versioned 64-dimensional
  ``hardware_proprio_history_64d_v1`` contract. Retrain policies with matching
  ``--history`` or ``--no-history`` settings; 72-dimensional checkpoints are
  rejected rather than adapted.
* Changed the scalar straight-line motion reward to exclude absolute world
  lateral-position and heading recovery in the active V5 configuration. Use
  the retained ``pose_weight`` compatibility option or simulator-only
  diagnostics when those world-pose measurements are needed for analysis.
* **Breaking:** Changed the default history-aware time-reversal objective from
  the same-order framewise feature heuristic to an exact causal sequence built
  from ``H + 2`` transition records. Use
  ``tr_consistency_mode="framewise_feature_approx"`` to reproduce the former
  feature-level ablation; legacy checkpoints require explicit transfer or the
  matching approximate mode rather than silently resuming under the new map.
* Changed symmetric Go2 and Dobot X1 PPO defaults to use time-reversal policy
  consistency without auxiliary critic consistency. Set
  ``symmetry_cfg.value_loss_coeff`` to a positive value to restore the optional
  time-reversal critic-consistency ablation; standard PPO value regression is
  unchanged.
