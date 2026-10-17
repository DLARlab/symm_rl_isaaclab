Added
^^^^^

* Added a combined video, data, and statistical evaluation launcher restricted
  to the selected checkpoint's saved training gaits, including the X1 Go2-bend
  variant. Use the root ``evaluation.sh --robot x1_go2bend`` launcher, or select
  ``--protocol trained`` in the shared statistical evaluator.

Deprecated
^^^^^^^^^^

* Deprecated the root ``test.sh`` launcher in favor of
  ``evaluation.sh --mode video``. The old entry point forwards to the combined
  launcher and selects only saved training gaits.
