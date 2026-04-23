"""
Judge Schemas — Pydantic validation for all LLM judge outputs.

Covers:
- Stage 0: query and task node shapes
- Stage 2: global reasoning output
- Stage 3: per-task metric scoring output
- Stage 3S: semantic scoring output
- Stage 4: final score breakdown
"""

from __future__ import annotations
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# STAGE 0 — INPUT SHAPES
# ============================================================================

class DependencyNode(BaseModel):
    id: str
    content: Any


class QueryInput(BaseModel):
    instruction: str
    description: str
    dependencies: List[DependencyNode] = Field(default_factory=list)


class TaskNode(BaseModel):
    id: str
    task_type: Literal["FETCH", "COMPUTE", "DECOMPOSE", "EXPAND", "OUTPUT"]
    instruction: str
    dependencies: List[str] = Field(default_factory=list)
    target_output: Optional[str] = None
    description: Optional[str] = None
    reasoning: Optional[str] = None


class PipelineInput(BaseModel):
    """Complete input to the judge pipeline."""
    domain: str
    query: QueryInput
    extracted_tasks: List[TaskNode]


# ============================================================================
# STAGE 1 — STRUCTURAL CHECKER OUTPUT
# ============================================================================

class StructuralResult(BaseModel):
    namespace_score: float = Field(ge=0.0, le=1.0)
    boundary_score: float = Field(ge=0.0, le=1.0)
    structural_score: float = Field(ge=0.0, le=1.0)
    violations: List[str] = Field(default_factory=list)
    gate_passed: bool


# ============================================================================
# STAGE 2 — GLOBAL REASONING OUTPUT
# ============================================================================

class GlobalReasoningOutput(BaseModel):
    global_reasoning: str
    overall_severity: float = Field(ge=0.0, le=1.0)
    required_but_missing_types: List[
        Literal["FETCH", "COMPUTE", "DECOMPOSE", "EXPAND"]
    ] = Field(default_factory=list)
    coverage_ratio: float = Field(ge=0.0, le=1.0)


# ============================================================================
# STAGE 3 — PER-TASK METRIC SCORING OUTPUT
# ============================================================================

MetricName = Literal["M1", "M2", "M5", "M6", "M7"]
OutcomeType = Literal[
    "correct",
    "wrong",
    "hallucinated",
    "unnecessary",
    "absent_not_required",
]


class MetricScore(BaseModel):
    metric: MetricName
    outcome: OutcomeType
    severity_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    local_reasoning: str

    def validate_severity(self) -> None:
        """Severity must be null for correct and absent_not_required outcomes."""
        if self.outcome in ("correct", "absent_not_required"):
            assert self.severity_ratio is None, (
                f"severity_ratio must be null for outcome={self.outcome}"
            )
        else:
            assert self.severity_ratio is not None, (
                f"severity_ratio required for outcome={self.outcome}"
            )


class PerTaskOutput(BaseModel):
    task_id: str
    metric_scores: List[MetricScore]


# ============================================================================
# STAGE 3S — SEMANTIC SCORING OUTPUT
# ============================================================================

class SemanticMetricScore(BaseModel):
    outcome: OutcomeType
    severity_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    local_reasoning: str


class SemanticOutput(BaseModel):
    S1: SemanticMetricScore  # Namespace hallucination
    S2: SemanticMetricScore  # Inheritance mistakes


# ============================================================================
# STAGE 4 — FINAL SCORE BREAKDOWN
# ============================================================================

class TypeScores(BaseModel):
    FETCH: Optional[float] = None
    COMPUTE: Optional[float] = None
    DECOMPOSE: Optional[float] = None
    EXPAND: Optional[float] = None


class Layer2Weights(BaseModel):
    FETCH: Optional[float] = None
    COMPUTE: Optional[float] = None
    DECOMPOSE: Optional[float] = None
    EXPAND: Optional[float] = None


class ScoreBreakdown(BaseModel):
    structural_score: float
    per_task_score: float
    semantic_score: float
    coverage_score: float
    raw_final: float
    normalized_final: float
    per_task_detail: List[PerTaskOutput]
    type_scores: TypeScores
    layer2_weights: Layer2Weights
