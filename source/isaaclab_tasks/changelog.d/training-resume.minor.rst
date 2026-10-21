Added
^^^^^

* Added command curriculum checkpoint state with validation of command grid
  ranges, bin counts, and sampling weights.
* Added optional TensorBoard and W&B logging continuation through
  ``--resume_logging`` and an offline utility for combining scalar histories.

Fixed
^^^^^

* Fixed resumed RSL-RL training to start at the next iteration and retain
  cumulative training counters. Older checkpoints remained loadable, with a
  warning when command curriculum state was unavailable.
