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

