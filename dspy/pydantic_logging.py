"""
Structured JSONL logging for DSPy → Pydantic outputs.

Mirrors `stateReasoning-1/cognitive_tasks.py` logging style, but writes to a
separate file for DSPy runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel


PYDANTIC_LOG_PATH = Path(__file__).resolve().parent / "docs" / "data" / "dspy_pydantic_outputs.jsonl"


def clear_pydantic_log(path: Path = PYDANTIC_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n\n")


def log_pydantic_output(label: str, obj: Any, path: Path = PYDANTIC_LOG_PATH) -> None:
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = obj

    _append(
        path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "type": getattr(obj, "__class__", type(obj)).__name__,
            "data": data,
        },
    )

