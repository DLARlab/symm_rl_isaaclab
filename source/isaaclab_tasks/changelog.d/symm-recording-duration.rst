Added
^^^^^

* Added automatic symmetric rollout plots during play and recording for base
  velocity and position tracking, gait phase weights, foot contact forces, and
  foot speeds. Use ``--no-plots`` to disable plot generation.

Changed
^^^^^^^

* Changed symmetric Go2 and Dobot X1 rollout recordings to last 30 seconds by
  default. Use ``--video-length`` to override the duration in environment steps.
