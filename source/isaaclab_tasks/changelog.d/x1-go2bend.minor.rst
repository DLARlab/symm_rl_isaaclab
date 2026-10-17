Added
^^^^^

* Added the ``x1_go2bend`` training and playback variant with hip, thigh, and
  calf defaults of 0, 0.6983, and -1.2842 rad on every leg. Reused X1 dynamics,
  rewards, gait timing, curriculum, and PPO/TRS settings with matching rear-calf
  limits and leg-symmetry conventions. Supported the variant in recording and
  evaluation, with separate logs under ``dobot_x1_go2bend_symm_flat``.
