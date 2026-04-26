from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from openai import OpenAI

try:
    # Optional to avoid import cycles in contexts that don't need logging.
    from log_utils import JsonlRunLogger
except Exception:  # pragma: no cover
    JsonlRunLogger = None  # type: ignore

OPENAI_API_KEY_HARDCODED = ""
OPENROUTER_API_KEY_HARDCODED = "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"

def parse_llm_json_response(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from an LLM response.

    Supports raw JSON, fenced ```json blocks, and mixed text with a JSON object.
    """
    if not text:
        raise ValueError("Empty LLM response; expected JSON.")

    # 1) Prefer fenced JSON blocks.
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fenced:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2) Try to find JSON object candidates in raw text (greedy).
    candidates = re.findall(r"\{[\s\S]*\}", text)
    for cand in candidates:
        cand = cand.strip()
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3) Fallback: brace-balanced scan starting from first '{'.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break

    preview = text[:800]
    raise ValueError(f"Could not extract a valid JSON object. First 800 chars:\n{preview}")

def parse_stage2_response(text: str) -> dict:
    """
    Extract Stage 2 fields directly by key pattern.
    Handles COT responses where JSON is embedded in reasoning text.
    Tries fenced JSON last block first, falls back to key extraction.
    """
    # 1. Try last fenced block first
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in reversed(fenced):
        try:
            parsed = json.loads(block.strip())
            if all(k in parsed for k in ["global_reasoning", "overall_severity", "coverage_ratio"]):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2. Try last raw JSON object that has all required keys
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand.strip())
            if all(k in parsed for k in ["global_reasoning", "overall_severity", "coverage_ratio"]):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Key-based extraction fallback
    result = {}

    # global_reasoning — grab between key and next key
    m = re.search(r'"global_reasoning"\s*:\s*"([\s\S]*?)"(?:\s*,|\s*\})', text)
    if m:
        result["global_reasoning"] = m.group(1)

    # overall_severity
    m = re.search(r'"overall_severity"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        result["overall_severity"] = float(m.group(1))

    # coverage_ratio
    m = re.search(r'"coverage_ratio"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        result["coverage_ratio"] = float(m.group(1))

    # required_but_missing_types — extract array
    m = re.search(r'"required_but_missing_types"\s*:\s*\[([\s\S]*?)\]', text)
    if m:
        types_raw = m.group(1)
        types = re.findall(r'"(FETCH|COMPUTE|DECOMPOSE|EXPAND)"', types_raw)
        result["required_but_missing_types"] = types
    else:
        result["required_but_missing_types"] = []

    if "global_reasoning" in result and "overall_severity" in result:
        return result

    raise ValueError(f"Could not extract Stage 2 fields. First 800 chars:\n{text[:800]}")


TParsed = TypeVar("TParsed")


def _is_retryable_parse_error(exc: BaseException) -> bool:
    return isinstance(exc, (ValueError, json.JSONDecodeError))


def parse_with_retries(
    *,
    call_llm: Callable[[int], str],
    parser_name: str,
    max_attempts: int = 3,
    retry_sleep_s: float = 0.35,
    logger: Optional[Any] = None,
    log_stage: str = "llm_parse",
    run_id: Optional[str] = None,
    domain: Optional[str] = None,
    request_payload: Optional[Dict[str, Any]] = None,
    max_chars: int = 4000,
) -> Any:
    """
    Call the LLM and parse the response, retrying on JSON parse/extraction failures.

    - `call_llm(attempt)` must return the raw model text for that attempt (1-indexed)
    - `parser_name` selects the project's custom parsers:
        - "parse_llm_json_response"
        - "parse_stage2_response"
    - Retries only when the parser raises ValueError / JSONDecodeError
    - If all attempts fail, re-raises the last exception (same behavior as before)
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Optional[BaseException] = None

    def _trunc(v: Any) -> Any:
        if isinstance(v, str) and max_chars > 0 and len(v) > max_chars:
            return v[:max_chars] + f"...[truncated:{len(v) - max_chars}]"
        return v

    def _log(event: str, payload: Dict[str, Any]) -> None:
        if logger is None:
            return
        try:
            # JsonlRunLogger signature: write(stage=..., payload=..., domain=..., run_id=...)
            logger.write(
                stage=f"{log_stage}.{event}",
                payload={k: _trunc(v) for k, v in payload.items()},
                domain=domain,
                run_id=run_id,
            )
        except Exception:
            return

    for attempt in range(1, max_attempts + 1):
        _log("request", {
            "attempt": attempt,
            "parser_name": parser_name,
            "request_payload": request_payload or {},
        })
        raw = call_llm(attempt)
        _log("response_raw", {
            "attempt": attempt,
            "raw": raw,
        })
        
        try:
            if parser_name == "parse_llm_json_response":
                parsed = parse_llm_json_response(raw)
                _log("parse_success", {"attempt": attempt, "parsed": parsed})
                return parsed
            if parser_name == "parse_stage2_response":
                parsed = parse_stage2_response(raw)
                _log("parse_success", {"attempt": attempt, "parsed": parsed})
                return parsed
            raise ValueError(f"Unknown parser_name={parser_name!r}")

        except Exception as e:
            _log("parse_failure", {
                "attempt": attempt,
                "error_type": type(e).__name__,
                "error": str(e),
            })
            last_exc = e
            if not _is_retryable_parse_error(e):
                raise
            if attempt >= max_attempts:
                raise
            if retry_sleep_s > 0:
                time.sleep(retry_sleep_s * attempt)

    # Unreachable, but keeps type-checkers happy.
    if last_exc:
        raise last_exc
    raise RuntimeError("parse_with_retries reached impossible state")


def get_api_key(provider: str, *, hardcoded: str = "", env_var: str = "") -> str:
    key = (hardcoded or os.getenv(env_var, "")).strip()
    if not key:
        raise RuntimeError(f"Missing API key for provider={provider!r}. Set {env_var} or hardcode it.")
    return key


def build_client(
    *,
    provider: str,
    openai_api_key_hardcoded: str = "",
    openrouter_api_key_hardcoded: str = "",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_http_referer: str = "https://localhost",
    openrouter_app_title: str = "stateReasoning-1",
) -> OpenAI:
    provider_norm = provider.lower().strip()

    if provider_norm == "openai":
        key = get_api_key("openai", hardcoded=openai_api_key_hardcoded, env_var="OPENAI_API_KEY")
        if key.startswith("sk-or-"):
            raise RuntimeError("PROVIDER='openai' requires an OpenAI key (sk-...), not an OpenRouter key (sk-or-...).")
        return OpenAI(api_key=key)

    if provider_norm == "openrouter":
        key = get_api_key("openrouter", hardcoded=openrouter_api_key_hardcoded, env_var="OPENROUTER_API_KEY")
        if not key.startswith("sk-or-"):
            raise RuntimeError("PROVIDER='openrouter' requires an OpenRouter key (sk-or-...).")
        return OpenAI(
            base_url=openrouter_base_url,
            api_key=key,
            default_headers={
                "HTTP-Referer": openrouter_http_referer,
                "X-Title": openrouter_app_title,
            },
        )

    raise RuntimeError(f"Unknown provider={provider!r}. Use 'openai' or 'openrouter'.")

