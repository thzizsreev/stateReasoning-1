"""
Judge Pipeline — Single Entry Point.

Orchestrates all stages:
    Stage 0: Inputs assembled externally, passed in
    Stage 1: Structural checker (Python)
    Stage 2: Global reasoning (LLM)
    Stage 3: Per-task metric scoring (LLM, one call per task)
    Stage 3S: Semantic scoring (LLM)
    Stage 4: Score assembly (Python)

Uses llm_utils.build_client() and parse_llm_json_response()
exactly as in crucible files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from openai import OpenAI

# Path resolution matching crucible_c1.py pattern
_JUDGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _JUDGE_DIR.parent
for _p in [str(_JUDGE_DIR), str(_PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_utils import (
    build_client,
    parse_llm_json_response,
    parse_stage2_response,
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
# CONFIG
# ============================================================================

PROVIDER        = "openrouter"
MODEL_OPENAI    = "gpt-4o"
MODEL_OPENROUTER = "anthropic/claude-sonnet-4-5"

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
        response = client.chat.completions.create(
            model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=3000,
            #response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": STAGE_2_GLOBAL_REASONING_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = parse_stage2_response(raw)
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
        response = client.chat.completions.create(
            model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": STAGE_3_PER_TASK_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = parse_llm_json_response(raw)

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
) -> list[PerTaskOutput]:
    results = []
    for task in pipeline_input.extracted_tasks:
        print(f"    [Stage 3] Scoring task {task.id} ({task.task_type})...")
        result = run_stage_3_single_task(client, pipeline_input, task, global_context)
        if result:
            results.append(result)
    print(f"    [Stage 3]  {results}")
    return results


# ============================================================================
# STAGE 3S — SEMANTIC SCORING
# ============================================================================

def run_stage_3s(
    client: OpenAI,
    pipeline_input: PipelineInput,
    global_context: GlobalReasoningOutput,
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
        response = client.chat.completions.create(
            model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=3000,
            
            messages=[
                {"role": "system", "content": STAGE_3S_SEMANTIC_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = parse_llm_json_response(raw)
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
   
    client = build_client(
        provider=PROVIDER,
        openai_api_key_hardcoded=OPENAI_API_KEY_HARDCODED,
        openrouter_api_key_hardcoded=OPENROUTER_API_KEY_HARDCODED,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_http_referer=OPENROUTER_HTTP_REFERER,
        openrouter_app_title=OPENROUTER_APP_TITLE,
    )

    print(f"\n[Judge] Starting pipeline for domain={pipeline_input.domain}")
    print(f"[Judge] Tasks: {[t.id for t in pipeline_input.extracted_tasks]}")

    # ── Stage 1: Structural check ──────────────────────────────────────────
    print("[Judge] Stage 1: Structural check...")
    structural_result: StructuralResult = run_structural_check(pipeline_input)
    print(f"    [Stage 1] Structural check output: {structural_result}")
    print(f"  structural_score={structural_result.structural_score}, "
          f"gate_passed={structural_result.gate_passed}")

    if structural_result.violations:
        for v in structural_result.violations:
            print(f"  ⚠️  {v}")

    if not structural_result.gate_passed:
        print("[Judge] Gate failed — skipping LLM stages. Returning early score.")
        # Return early breakdown with structural score only
        from judge_scorer import normalize
        early_score = normalize(structural_result.structural_score * 0.30)
        return ScoreBreakdown(
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

    # ── Stage 2: Global reasoning ──────────────────────────────────────────
    print("[Judge] Stage 2: Global reasoning...")
    global_context = run_stage_2(client, pipeline_input)
    if not global_context:
        return None
    print(f"  overall_severity={global_context.overall_severity}, "
          f"coverage_ratio={global_context.coverage_ratio}, "
          f"missing_types={global_context.required_but_missing_types}")

    # ── Stage 3: Per-task scoring ──────────────────────────────────────────
    print("[Judge] Stage 3: Per-task metric scoring...")
    per_task_outputs = run_stage_3(client, pipeline_input, global_context)
    if not per_task_outputs:
        return None

    # ── Stage 3S: Semantic scoring ─────────────────────────────────────────
    print("[Judge] Stage 3S: Semantic scoring...")
    semantic_output = run_stage_3s(client, pipeline_input, global_context)
    if not semantic_output:
        return None

    # ── Stage 4: Score assembly ────────────────────────────────────────────
    print("[Judge] Stage 4: Assembling scores...")
    breakdown = assemble_score(
        pipeline_input=pipeline_input,
        structural_result=structural_result,
        global_reasoning=global_context,
        per_task_outputs=per_task_outputs,
        semantic_output=semantic_output,
    )

    print(f"\n[Judge] Final score: {breakdown.normalized_final}")
    print(f"  structural={breakdown.structural_score} | "
          f"per_task={breakdown.per_task_score} | "
          f"semantic={breakdown.semantic_score} | "
          f"coverage={breakdown.coverage_score}")

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