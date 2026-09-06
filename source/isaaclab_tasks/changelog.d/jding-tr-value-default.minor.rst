Changed
^^^^^^^

* Changed symmetric Go2 and Dobot X1 PPO defaults to use time-reversal policy
  consistency without auxiliary critic consistency. Set
  ``symmetry_cfg.value_loss_coeff`` to a positive value to restore the optional
  time-reversal critic-consistency ablation; standard PPO value regression is
  unchanged.
