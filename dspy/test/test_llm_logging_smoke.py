from __future__ import annotations

from pathlib import Path

from llm_utils import parse_with_retries
from log_utils import JsonlRunLogger


def main() -> int:
    log_path = Path(__file__).resolve().parents[1] / "logs" / "smoke_llm_parse_log.jsonl"
    logger = JsonlRunLogger(path=log_path)

    def fake_llm(_: int) -> str:
        return '{"ok": true, "msg": "hello"}'

    parsed = parse_with_retries(
        call_llm=fake_llm,
        parser_name="parse_llm_json_response",
        logger=logger,
        log_stage="smoke",
        request_payload={"purpose": "verify append-only logging"},
        max_attempts=3,
        retry_sleep_s=0.0,
    )
    print(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

