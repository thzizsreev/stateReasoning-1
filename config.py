"""
Compatibility shim for the DSPy rewrite.

Some scripts (e.g. `cognitive_tasks.py`) import `configure_lm` from a top-level
`config.py`. The DSPy implementation lives under `stateReasoning-1/dspy/config.py`,
but that folder name collides with the installed `dspy` library name.

This shim loads the local DSPy config module by file path and re-exports the
public entrypoints.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_local_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_BASE_DIR = Path(__file__).resolve().parent
_DSPY_DIR = _BASE_DIR / "dspy"
_DSPY_CONFIG_PATH = _DSPY_DIR / "config.py"

if not _DSPY_CONFIG_PATH.exists():
    raise ImportError(f"Missing DSPy config at {_DSPY_CONFIG_PATH}")

_local_config = _load_local_module("_local_dspy_config", _DSPY_CONFIG_PATH)

configure_lm = getattr(_local_config, "configure_lm")

__all__ = ["configure_lm"]

