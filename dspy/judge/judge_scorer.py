"""
Judge Scorer — Stage 4.

Pure Python reward math and normalization.
No LLM involvement.

Operations:
1. Apply reward schema per metric (outcome + severity_ratio)
2. Average metrics within each task → task_score[i]
3. Layer 1: soft average within type → type_score[t]
4. Inject -1.5 for required-but-missing types
5. Layer 2: power norm (α=1.5) across types → per_task_score
6. Compute semantic_score from S1, S2
7. Compute coverage_score from coverage_ratio
8. Weighted sum → raw_final
9. Normalize to [0.0, 1.0]
"""

from __future__ import annotations
import math
from typing import Optional
from judge_schemas import (
    PipelineInput,
    StructuralResult,
    GlobalReasoningOutput,
    PerTaskOutput,
    SemanticOutput,
    ScoreBreakdown,
    TypeScores,
    Layer2Weights,
)


# ============================================================================
# CONFIG
# ============================================================================

# Final weighted sum weights
W_STRUCTURAL = 0.30
W_TASK       = 0.40
W_SEMANTIC   = 0.20
W_COVERAGE   = 0.10

# Layer 2 importance weights
IMPORTANCE = {
    "EXPAND":    3.0,
    "DECOMPOSE": 2.5,
    "FETCH":     2.0,
    "COMPUTE":   1.0,
}

POWER_ALPHA = 1.5

# Normalization range: raw scores span [-1.5, +1.0]
RAW_MIN = -1.5
RAW_MAX = +1.0

# Semantic weights
W_S1 = 0.55
W_S2 = 0.45


# ============================================================================
# REWARD SCHEMA
# ============================================================================

def apply_reward(outcome: str, severity_ratio: Optional[float]) -> float:
    """
    Apply reward schema based on outcome and severity_ratio.
    Correct outcomes always get +1.0 regardless of severity.
    """
    if outcome == "correct":
        return +1.0
    elif outcome == "absent_not_required":
        return 0.0
    elif outcome == "wrong":
        return -0.5 * (severity_ratio or 1.0)
    elif outcome == "hallucinated":
        return -1.0 * (severity_ratio or 1.0)
    elif outcome == "unnecessary":
        return -1.5 * (severity_ratio or 1.0)
    else:
        return 0.0


# ============================================================================
# LAYER 1 — SOFT WITHIN-CATEGORY NORMALIZATION
# ============================================================================

def compute_task_score(per_task_output: PerTaskOutput) -> float:
    """Average all metric rewards within one task."""
    scores = [
        apply_reward(m.outcome, m.severity_ratio)
        for m in per_task_output.metric_scores
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_type_scores(
    per_task_outputs: list[PerTaskOutput],
    pipeline_input: PipelineInput,
) -> dict[str, float]:
    """
    Layer 1: soft average within each task type.
    Groups tasks by type, averages task_scores within group.
    Volume within a type cannot inflate the type score.
    """
    type_buckets: dict[str, list[float]] = {}

    task_type_map = {
        t.id: t.task_type for t in pipeline_input.extracted_tasks
    }

    for pto in per_task_outputs:
        task_type = task_type_map.get(pto.task_id, "COMPUTE")
        score = compute_task_score(pto)
        type_buckets.setdefault(task_type, []).append(score)

    return {
        t: sum(scores) / len(scores)
        for t, scores in type_buckets.items()
    }


# ============================================================================
# LAYER 2 — SHARP ACROSS-CATEGORY NORMALIZATION
# ============================================================================

def inject_missing_types(
    type_scores: dict[str, float],
    required_but_missing: list[str],
) -> dict[str, float]:
    """Inject -1.5 penalty for required-but-missing task types."""
    result = type_scores.copy()
    for t in required_but_missing:
        if t not in result:
            result[t] = -1.5
    return result


def power_norm_weights(active_types: list[str]) -> dict[str, float]:
    """
    Power normalization (α=1.5) on importance weights.
    Sharp — preserves importance ratio across categories.
    """
    raw = {t: IMPORTANCE.get(t, 1.0) ** POWER_ALPHA for t in active_types}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}


def compute_per_task_score(type_scores: dict[str, float]) -> float:
    """
    Layer 2: power-normalized weighted sum across type scores.
    """
    active_types = list(type_scores.keys())
    weights = power_norm_weights(active_types)
    return sum(weights[t] * type_scores[t] for t in active_types)


# ============================================================================
# SEMANTIC SCORE
# ============================================================================

def compute_semantic_score(semantic_output: SemanticOutput) -> float:
    s1 = apply_reward(semantic_output.S1.outcome, semantic_output.S1.severity_ratio)
    s2 = apply_reward(semantic_output.S2.outcome, semantic_output.S2.severity_ratio)
    return W_S1 * s1 + W_S2 * s2


# ============================================================================
# COVERAGE SCORE
# ============================================================================

def compute_coverage_score(coverage_ratio: float) -> float:
    """Map [0.0, 1.0] coverage_ratio to [-1.0, +1.0] score."""
    return (2.0 * coverage_ratio) - 1.0


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize(raw: float) -> float:
    """Normalize raw score from [RAW_MIN, RAW_MAX] to [0.0, 1.0]."""
    normalized = (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return max(0.0, min(1.0, normalized))


# ============================================================================
# MAIN SCORER ENTRY POINT
# ============================================================================

def assemble_score(
    pipeline_input: PipelineInput,
    structural_result: StructuralResult,
    global_reasoning: GlobalReasoningOutput,
    per_task_outputs: list[PerTaskOutput],
    semantic_output: SemanticOutput,
) -> ScoreBreakdown:
    """
    Stage 4 entry point.
    Assembles all component scores into final normalized scalar.
    """

    # Layer 1 — soft within type
    type_scores_raw = compute_type_scores(per_task_outputs, pipeline_input)

    # Inject missing type penalties
    type_scores_full = inject_missing_types(
        type_scores_raw,
        global_reasoning.required_but_missing_types,
    )

    # Layer 2 — sharp across types
    per_task_score = compute_per_task_score(type_scores_full)

    # Semantic
    semantic_score = compute_semantic_score(semantic_output)

    # Coverage
    coverage_score = compute_coverage_score(global_reasoning.coverage_ratio)

    # Weighted sum
    raw_final = (
        W_STRUCTURAL * structural_result.structural_score +
        W_TASK       * per_task_score +
        W_SEMANTIC   * semantic_score +
        W_COVERAGE   * coverage_score
    )

    normalized_final = normalize(raw_final)

    # Build weights for logging
    active_types = list(type_scores_full.keys())
    weights = power_norm_weights(active_types)

    # Force every entry into the canonical PerTaskOutput model (from judge_schemas).
    # This avoids failures when test code constructs a same-shaped model from a different import path.
    per_task_detail: list[PerTaskOutput] = []
    for t in (per_task_outputs or []):
        if isinstance(t, PerTaskOutput):
            per_task_detail.append(t)
            continue
        payload = t.model_dump() if hasattr(t, "model_dump") else t
        per_task_detail.append(PerTaskOutput.model_validate(payload))

    return ScoreBreakdown(
        structural_score=round(structural_result.structural_score, 4),
        per_task_score=round(per_task_score, 4),
        semantic_score=round(semantic_score, 4),
        coverage_score=round(coverage_score, 4),
        raw_final=round(raw_final, 4),
        normalized_final=round(normalized_final, 4),
        per_task_detail=per_task_detail,
        type_scores=TypeScores(**{
            t: round(v, 4) for t, v in type_scores_full.items()
        }),
        layer2_weights=Layer2Weights(**{
            t: round(weights.get(t, 0.0), 4) for t in active_types
        }),
    )
