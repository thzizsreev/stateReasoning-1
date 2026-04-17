"""
Compatibility shim for the DSPy rewrite.

Re-exports the DSPy pipeline entrypoints so legacy imports like:

  from pipeline import QueryPipeline, process_query, execute_compute_node

keep working from the `stateReasoning-1/` directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_local_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_BASE_DIR = Path(__file__).resolve().parent
_DSPY_DIR = _BASE_DIR / "dspy"
_DSPY_PIPELINE_PATH = _DSPY_DIR / "pipeline.py"

if not _DSPY_PIPELINE_PATH.exists():
    raise ImportError(f"Missing DSPy pipeline at {_DSPY_PIPELINE_PATH}")

# Ensure local DSPy module directory is importable (it imports `modules`, `schema`, etc.)
import sys as _sys
_dspy_dir_str = str(_DSPY_DIR)
if _dspy_dir_str not in _sys.path:
    _sys.path.insert(0, _dspy_dir_str)

_local_pipeline = _load_local_module("_local_dspy_pipeline", _DSPY_PIPELINE_PATH)

QueryPipeline = getattr(_local_pipeline, "QueryPipeline")
ComputeNodeExecutor = getattr(_local_pipeline, "ComputeNodeExecutor")
process_query = getattr(_local_pipeline, "process_query")
execute_compute_node = getattr(_local_pipeline, "execute_compute_node")

__all__ = [
    "QueryPipeline",
    "ComputeNodeExecutor",
    "process_query",
    "execute_compute_node",
]

