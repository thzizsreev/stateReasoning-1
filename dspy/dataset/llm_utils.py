from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

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

