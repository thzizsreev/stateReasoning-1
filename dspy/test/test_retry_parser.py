from __future__ import annotations

import sys

from llm_utils import parse_with_retries


def _run_one(parser_name: str) -> int:
    calls: list[int] = []

    def fake_llm(attempt: int) -> str:
        calls.append(attempt)
        # Intentionally NOT JSON (and includes braces to defeat simple "{...}" matches too)
        return "NOT JSON >>> { broken: true, 'oops': [1,2, } <<< still not json"

    try:
        _ = parse_with_retries(
            call_llm=fake_llm,
            parser_name=parser_name,
            max_attempts=3,
            retry_sleep_s=0.0,  # keep test fast
        )
        print(f"[FAIL] parser={parser_name}: unexpectedly parsed successfully")
        return 1
    except Exception as e:
        ok_calls = calls == [1, 2, 3]
        print(f"[OK] parser={parser_name}: raised after {len(calls)} attempts")
        print(f"     attempts={calls} (expected [1, 2, 3])")
        print(f"     error_type={type(e).__name__}")
        print(f"     error_preview={str(e)[:160]!r}")
        return 0 if ok_calls else 2


def main() -> int:
    exit_codes = [
        _run_one("parse_llm_json_response"),
        _run_one("parse_stage2_response"),
    ]
    return 0 if all(c == 0 for c in exit_codes) else max(exit_codes)


if __name__ == "__main__":
    raise SystemExit(main())

