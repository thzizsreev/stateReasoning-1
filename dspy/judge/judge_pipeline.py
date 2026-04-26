

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from openai import OpenAI

# Path resolution matching crucible_c1.py pattern
_JUDGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _JUDGE_DIR.parent.parent
_DSPY_DIR = _JUDGE_DIR.parent
for _p in [str(_JUDGE_DIR), str(_DSPY_DIR), str(_PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Log file — always relative to the project root, resolved at import time.
_LOG_FILE = _PROJECT_DIR / "logs" / "pipeline_stage_log.jsonl"

from log_utils import JsonlRunLogger

from llm_utils import (
    build_client,
    parse_llm_json_response,
    parse_stage2_response,
    parse_with_retries,
    OPENAI_API_KEY_HARDCODED,
    OPENROUTER_API_KEY_HARDCODED,
)
from judge_schemas import (
    PipelineInput,
    StructuralResult,
    GlobalReasoningOutput,
    PerTaskOutput,
    MetricScore,
    SemanticOutput,
    SemanticMetricScore,
    ScoreBreakdown,
    TaskNode,
    TypeScores,
    Layer2Weights,
    QueryInput,
)
from judge_structural import run_structural_check
from judge_scorer import assemble_score
from judge_prompts import (
    STAGE_2_GLOBAL_REASONING_PROMPT,
    STAGE_3_PER_TASK_PROMPT,
    STAGE_3S_SEMANTIC_PROMPT,
)

# ============================================================================
# STAGE LOGGING
# ============================================================================

def _make_run_logger() -> tuple[JsonlRunLogger, str]:
    logger = JsonlRunLogger(path=_LOG_FILE)
    run_id = uuid.uuid4().hex
    return logger, run_id


# ============================================================================
# CONFIG
# ============================================================================

PROVIDER        = "openrouter"
MODEL_OPENAI    = "gpt-4o"
MODEL_OPENROUTER = "openai/gpt-oss-120b"

OPENROUTER_BASE_URL     = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = "https://localhost"
OPENROUTER_APP_TITLE    = "stateReasoning-1"

JUDGE_TEMPERATURE = 0.1  # low — deterministic, consistent scoring

# Metrics applicable per task type
APPLICABLE_METRICS: dict[str, list[str]] = {
    "FETCH":     ["M1", "M5", "M7"],
    "COMPUTE":   ["M1", "M6", "M7"],
    "DECOMPOSE": ["M1", "M2", "M5", "M6", "M7"],
    "EXPAND":    ["M1", "M2", "M5", "M6", "M7"],
    "OUTPUT":    ["M1", "M7"],
}


# ============================================================================
# STAGE 2 — GLOBAL REASONING
# ============================================================================

def run_stage_2(
    client: OpenAI,
    pipeline_input: PipelineInput,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
    run_id: Optional[str] = None,
) -> Optional[GlobalReasoningOutput]:
    """
    LLM call: forms holistic impression of the full output.
    No per-metric scoring — purely assessment.
    """
    user_prompt = json.dumps({
        "query": pipeline_input.query.model_dump(),
        "extracted_tasks": [t.model_dump() for t in pipeline_input.extracted_tasks],
    }, indent=2)

    try:
        def _call(attempt: int) -> str:
            messages = [
                {"role": "system", "content": STAGE_2_GLOBAL_REASONING_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 1:
                messages.append({
                    "role": "system",
                    "content": "Retry: return ONLY a valid JSON object. No markdown, no prose.",
                })
            response = client.chat.completions.create(
                model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
                temperature=JUDGE_TEMPERATURE,
                max_tokens=3000,
                messages=messages,
            )
            return response.choices[0].message.content or ""

        data = parse_with_retries(
            call_llm=_call,
            parser_name="parse_stage2_response",
            max_attempts=3,
            logger=run_logger,
            log_stage="judge.stage2",
            run_id=run_id,
            domain=pipeline_input.domain,
            request_payload={"user_prompt": user_prompt},
        )
        print(f"    [Stage 2] Global reasoning output: {data}")
        return GlobalReasoningOutput(**data)

    except Exception as e:
        print(f"  [Stage 2] Global reasoning failed: {e}")
        return None


# ============================================================================
# STAGE 3 — PER-TASK METRIC SCORING
# ============================================================================

def run_stage_3_single_task(
    client: OpenAI,
    pipeline_input: PipelineInput,
    task: TaskNode,
    global_context: GlobalReasoningOutput,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
    run_id: Optional[str] = None,
) -> Optional[PerTaskOutput]:
    applicable = APPLICABLE_METRICS.get(task.task_type, ["M1", "M7"])

    user_prompt = json.dumps({
        "query": pipeline_input.query.model_dump(),
        "global_context": {
            "global_reasoning": global_context.global_reasoning,
            "overall_severity": global_context.overall_severity,
        },
        "task": task.model_dump(),
        "applicable_metrics": applicable,
    }, indent=2)

    try:
        def _call(attempt: int) -> str:
            messages = [
                {"role": "system", "content": STAGE_3_PER_TASK_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 1:
                messages.append({
                    "role": "system",
                    "content": "Retry: output ONLY valid JSON. No markdown, no extra keys.",
                })
            response = client.chat.completions.create(
                model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
                temperature=JUDGE_TEMPERATURE,
                messages=messages,
            )
            return response.choices[0].message.content or ""

        data = parse_with_retries(
            call_llm=_call,
            parser_name="parse_llm_json_response",
            max_attempts=3,
            logger=run_logger,
            log_stage="judge.stage3",
            run_id=run_id,
            domain=pipeline_input.domain,
            request_payload={
                "task_id": task.id,
                "task_type": task.task_type,
                "user_prompt": user_prompt,
            },
        )

        metric_scores = [
            MetricScore(**m) for m in data.get("metric_scores", [])
        ]
        return PerTaskOutput(
            task_id=task.id,
            metric_scores=metric_scores,
        )

    except Exception as e:
        print(f"  ❌ [Stage 3] Task {task.id} scoring failed: {e}")
        return None


def run_stage_3(
    client: OpenAI,
    pipeline_input: PipelineInput,
    global_context: GlobalReasoningOutput,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
    run_id: Optional[str] = None,
) -> list[PerTaskOutput]:
    results = []
    for task in pipeline_input.extracted_tasks:
        print(f"    [Stage 3] Scoring task {task.id} ({task.task_type})...")
        result = run_stage_3_single_task(
            client,
            pipeline_input,
            task,
            global_context,
            run_logger=run_logger,
            run_id=run_id,
        )
        if result:
            results.append(result)
    # Avoid printing full objects (can include unicode that breaks cp1252 consoles on Windows).
    print(f"    [Stage 3] Completed {len(results)} task results.")
    return results


# ============================================================================
# STAGE 3S — SEMANTIC SCORING
# ============================================================================

def run_stage_3s(
    client: OpenAI,
    pipeline_input: PipelineInput,
    global_context: GlobalReasoningOutput,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
    run_id: Optional[str] = None,
) -> Optional[SemanticOutput]:
    """
    LLM call: evaluates holistic semantic dimensions.
    S1 — namespace hallucination
    S2 — inheritance mistakes
    """
    user_prompt = json.dumps({
        "query": pipeline_input.query.model_dump(),
        "global_context": {
            "global_reasoning": global_context.global_reasoning,
            "overall_severity": global_context.overall_severity,
        },
        "extracted_tasks": [t.model_dump() for t in pipeline_input.extracted_tasks],
    }, indent=2)

    try:
        def _call(attempt: int) -> str:
            messages = [
                {"role": "system", "content": STAGE_3S_SEMANTIC_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 1:
                messages.append({
                    "role": "system",
                    "content": "Retry: return ONLY a valid JSON object matching the expected keys.",
                })
            response = client.chat.completions.create(
                model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
                temperature=JUDGE_TEMPERATURE,
                messages=messages,
            )
            return response.choices[0].message.content or ""

        data = parse_with_retries(
            call_llm=_call,
            parser_name="parse_llm_json_response",
            max_attempts=3,
            logger=run_logger,
            log_stage="judge.stage3s",
            run_id=run_id,
            domain=pipeline_input.domain,
            request_payload={"user_prompt": user_prompt},
        )
        print(f"    [Stage 3S] Semantic scoring output: {data}")
        return SemanticOutput(
            S1=SemanticMetricScore(**data["S1"]),
            S2=SemanticMetricScore(**data["S2"]),
        )

    except Exception as e:
        print(f"  [Stage 3S] Semantic scoring failed: {e}")
        return None


# ============================================================================
# PIPELINE ENTRY POINT
# ============================================================================

def run_judge_pipeline(pipeline_input: PipelineInput) -> Optional[ScoreBreakdown]:
    domain = pipeline_input.domain

    # Append-only log file for each run
    run_logger, run_id = _make_run_logger()

    client = build_client(
        provider=PROVIDER,
        openai_api_key_hardcoded=OPENAI_API_KEY_HARDCODED,
        openrouter_api_key_hardcoded=OPENROUTER_API_KEY_HARDCODED,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_http_referer=OPENROUTER_HTTP_REFERER,
        openrouter_app_title=OPENROUTER_APP_TITLE,
    )

    print(f"\n[Judge] Starting pipeline for domain={domain}")
    print(f"[Judge] Tasks: {[t.id for t in pipeline_input.extracted_tasks]}")
    run_logger.write(
        stage="pipeline_start",
        domain=domain,
        run_id=run_id,
        payload={
            "query": pipeline_input.query.model_dump(),
            "extracted_tasks": [t.model_dump() for t in pipeline_input.extracted_tasks],
        },
    )

    # ── Stage 1: Structural check ──────────────────────────────────────────
    print("[Judge] Stage 1: Structural check...")
    structural_result: StructuralResult = run_structural_check(pipeline_input)
    print(f"    [Stage 1] Structural check output: {structural_result}")
    run_logger.write(
        stage="stage1_structural",
        domain=domain,
        run_id=run_id,
        payload=structural_result.model_dump(),
    )
    print(f"  structural_score={structural_result.structural_score}, "
          f"gate_passed={structural_result.gate_passed}")

    if structural_result.violations:
        for v in structural_result.violations:
            print(f"  [!] {v}")

    if not structural_result.gate_passed:
        print("[Judge] Gate failed — skipping LLM stages. Returning early score.")
        from judge_scorer import normalize
        early_score = normalize(structural_result.structural_score * 0.30)
        breakdown = ScoreBreakdown(
            structural_score=structural_result.structural_score,
            per_task_score=0.0,
            semantic_score=0.0,
            coverage_score=0.0,
            raw_final=structural_result.structural_score * 0.30,
            normalized_final=early_score,
            per_task_detail=[],
            type_scores=TypeScores(),
            layer2_weights=Layer2Weights(),
        )
        run_logger.write(
            stage="stage4_breakdown_early",
            domain=domain,
            run_id=run_id,
            payload=breakdown.model_dump(),
        )
        run_logger.write(
            stage="pipeline_end",
            domain=domain,
            run_id=run_id,
            payload={"status": "early_return"},
        )
        return breakdown

    # ── Stage 2: Global reasoning ──────────────────────────────────────────
    print("[Judge] Stage 2: Global reasoning...")
    global_context = run_stage_2(client, pipeline_input, run_logger=run_logger, run_id=run_id)
    if not global_context:
        run_logger.write(
            stage="stage2_global_reasoning_error",
            domain=domain,
            run_id=run_id,
            payload={"error": "run_stage_2 returned None"},
        )
        run_logger.write(
            stage="pipeline_end",
            domain=domain,
            run_id=run_id,
            payload={"status": "error_stage2"},
        )
        return None
    run_logger.write(
        stage="stage2_global_reasoning",
        domain=domain,
        run_id=run_id,
        payload=global_context.model_dump(),
    )
    print(f"  overall_severity={global_context.overall_severity}, "
          f"coverage_ratio={global_context.coverage_ratio}, "
          f"missing_types={global_context.required_but_missing_types}")

    # ── Stage 3: Per-task scoring ──────────────────────────────────────────
    print("[Judge] Stage 3: Per-task metric scoring...")
    per_task_outputs = run_stage_3(client, pipeline_input, global_context, run_logger=run_logger, run_id=run_id)
    if not per_task_outputs:
        run_logger.write(
            stage="stage3_per_task_error",
            domain=domain,
            run_id=run_id,
            payload={"error": "run_stage_3 produced empty results"},
        )
        run_logger.write(
            stage="pipeline_end",
            domain=domain,
            run_id=run_id,
            payload={"status": "error_stage3"},
        )
        return None
    run_logger.write(
        stage="stage3_per_task",
        domain=domain,
        run_id=run_id,
        payload={"per_task_outputs": [p.model_dump() for p in per_task_outputs]},
    )

    # ── Stage 3S: Semantic scoring ─────────────────────────────────────────
    print("[Judge] Stage 3S: Semantic scoring...")
    semantic_output = run_stage_3s(client, pipeline_input, global_context, run_logger=run_logger, run_id=run_id)
    if not semantic_output:
        run_logger.write(
            stage="stage3s_semantic_error",
            domain=domain,
            run_id=run_id,
            payload={"error": "run_stage_3s returned None"},
        )
        run_logger.write(
            stage="pipeline_end",
            domain=domain,
            run_id=run_id,
            payload={"status": "error_stage3s"},
        )
        return None
    run_logger.write(
        stage="stage3s_semantic",
        domain=domain,
        run_id=run_id,
        payload=semantic_output.model_dump(),
    )

    # ── Stage 4: Score assembly ────────────────────────────────────────────
    print("[Judge] Stage 4: Assembling scores...")
    breakdown = assemble_score(
        pipeline_input=pipeline_input,
        structural_result=structural_result,
        global_reasoning=global_context,
        per_task_outputs=per_task_outputs,
        semantic_output=semantic_output,
    )
    run_logger.write(
        stage="stage4_breakdown",
        domain=domain,
        run_id=run_id,
        payload=breakdown.model_dump(),
    )

    print(f"\n[Judge] Final score: {breakdown.normalized_final}")
    print(f"  structural={breakdown.structural_score} | "
          f"per_task={breakdown.per_task_score} | "
          f"semantic={breakdown.semantic_score} | "
          f"coverage={breakdown.coverage_score}")

    run_logger.write(
        stage="pipeline_end",
        domain=domain,
        run_id=run_id,
        payload={"status": "ok"},
    )
    return breakdown



if __name__ == "__main__":
    pipeline_input = PipelineInput(
        domain="test",
        query=QueryInput(
            instruction="test",
            description="test",
            dependencies=[],
        ),
    )