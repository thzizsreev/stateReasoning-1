from __future__ import annotations

import json
import os
import sys
from typing import Any

import ollama

sys.path.insert(0, os.path.dirname(__file__))

from schemas import ComputeResult, PlannerResponse


def _chat_json(*, model: str, system: str, user: str) -> Any:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0},
        format="json",
    )
    raw = resp["message"]["content"]
    try:
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"LLM did not return valid JSON. Raw output:\n{raw}") from e


def planner_agent(
    phase_text: str,
    available_keys: list[str],
    model: str = "llama3.1:latest",
) -> PlannerResponse:
    system = (
        "Break the Phase into atomic tasks. "
        "Each task MUST be marked as task_type. "
        "For this system, ALWAYS set task_type=COMPUTE (do not output DECOMPOSE tasks). "
        "CRITICAL: 'dependencies' MUST contain ONLY data-node IDs from AVAILABLE_KEYS. "
        "Do NOT reference other task IDs in dependencies. "
        "Also output next_task_id so the orchestrator knows which ready task to run next. "
        "Output JSON."
    )
    user = (
        "PHASE_TEXT:\n"
        f"{phase_text}\n\n"
        "AVAILABLE_KEYS (use only these exact strings for dependencies and target_output):\n"
        f"{json.dumps(available_keys)}\n\n"
        "Rules:\n"
        "- dependencies: only from AVAILABLE_KEYS (no 'Task_*', no 'PHASE_TEXT')\n"
        "- task_type: MUST be \"COMPUTE\"\n"
        "- target_output MUST be a string (e.g. Data_B, Data_D, Data_C, PHASE_1_DONE)\n\n"
        "Return JSON matching:\n"
        '{"tasks":[{"id":"Task_1","task_type":"COMPUTE","dependencies":["Data_A"],"instruction":"...","target_output":"Data_B","status":"PENDING"}],"next_task_id":"Task_1"}'
    )
    payload = _chat_json(model=model, system=system, user=user)
    try:
        return PlannerResponse.model_validate(payload)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PlannerResponse validation failed. Payload:\n{payload}") from e


def worker_agent(
    instruction: str,
    isolated_context: dict,
    model: str = "llama3.1:latest",
) -> ComputeResult:
    system = (
        "You are a precise worker. Your only job is to COMPUTE the final math/logic "
        "from the isolated context and return the computed data. "
        "Do NOT create subtasks. "
        "Output strictly matching the requested JSON schema."
    )

    schema_hint = 'Return exactly: {"computed_data":{...},"target_node_id":"Data_X"}'
    user = (
        "INSTRUCTION:\n"
        f"{instruction}\n\n"
        "ISOLATED_CONTEXT (only these keys exist; do not assume anything else):\n"
        f"{json.dumps(isolated_context, indent=2)}\n\n"
        "CRITICAL:\n"
        "- target_node_id MUST be the exact node id requested by the instruction (e.g. Data_B)\n\n"
        f"{schema_hint}"
    )
    payload = _chat_json(model=model, system=system, user=user)
    try:
        return ComputeResult.model_validate(payload)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"ComputeResult validation failed. Payload:\n{payload}") from e

