from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JsonlRunLogger:
    """
    Minimal run logger designed for repeatable runs.

    - refresh(): no-op (kept for backwards compatibility; logs are append-only)
    - write(): appends one pretty-printed JSON object, never raises
    """

    path: Path

    def refresh(self) -> None:
        # Intentionally a no-op: logs are append-only.
        return

    def write(
        self,
        *,
        stage: str,
        payload: Dict[str, Any],
        domain: Optional[str] = None,
        run_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            record: Dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "payload": payload,
            }
            if domain is not None:
                record["domain"] = domain
            if run_id is not None:
                record["run_id"] = run_id
            if extra:
                record.update(extra)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
        except Exception:
            return

