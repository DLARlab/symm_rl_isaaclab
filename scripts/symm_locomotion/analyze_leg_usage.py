# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility entry point for :mod:`evaluation`."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def _load_evaluation_module():
    """Load the replacement evaluation module from the adjacent script."""
    module_name = "_symm_locomotion_evaluation_compat"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_path = Path(__file__).with_name("evaluation.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


warnings.warn(
    "analyze_leg_usage.py is deprecated; use evaluation.py instead.",
    DeprecationWarning,
    stacklevel=2,
)
_evaluation = _load_evaluation_module()
for _name in dir(_evaluation):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_evaluation, _name)

__all__ = [name for name in dir(_evaluation) if not name.startswith("__")]


if __name__ == "__main__":
    print("WARNING: analyze_leg_usage.py is deprecated; use evaluation.py instead.", file=sys.stderr)
    from symm_cli import main

    raise SystemExit(main(["analyze_leg_usage", *sys.argv[1:]]))
