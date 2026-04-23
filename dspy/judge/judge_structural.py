"""
Judge Structural Checker — Stage 1.

Pure Python, deterministic, no LLM.
Checks namespace fidelity and capability boundary violations.
Acts as gate — if structural_score < 0.4, skip all LLM judge calls.
"""

from __future__ import annotations
from judge_schemas import PipelineInput, StructuralResult


# Task type capability boundaries
BOUNDARY_RULES = {
    "FETCH": {
        "must_have_empty_dependencies": True,
        "cannot_spawn_types": [],
    },
    "DECOMPOSE": {
        "cannot_spawn_types": ["FETCH"],
    },
    "EXPAND": {
        "cannot_write_orchestration_end": True,
    },
}

STRUCTURAL_GATE_THRESHOLD = 0.4


def build_declared_set(pipeline_input: PipelineInput) -> set[str]:
    """
    Build the complete set of declared node IDs.
    Includes query dependencies + all task target_outputs.
    """
    declared = set()

    for dep in pipeline_input.query.dependencies:
        declared.add(dep.id)

    for task in pipeline_input.extracted_tasks:
        if task.target_output:
            declared.add(task.target_output)

    return declared


def check_namespace(
    pipeline_input: PipelineInput,
    declared_set: set[str],
) -> tuple[float, list[str]]:
    """
    Check all task dependencies against declared_set.
    Returns (namespace_score, list_of_violations).
    """
    violations = []
    total = 0
    passing = 0

    for task in pipeline_input.extracted_tasks:
        for dep in task.dependencies:
            total += 1
            if dep in declared_set:
                passing += 1
            else:
                violations.append(
                    f"Task {task.id}: dependency '{dep}' not in declared_set "
                    f"(phantom dependency — will cause deadlock)"
                )

    score = (passing / total) if total > 0 else 1.0
    return score, violations


def check_boundaries(pipeline_input: PipelineInput) -> tuple[float, list[str]]:
    """
    Check hard capability boundary violations.
    Returns (boundary_score, list_of_violations).
    """
    violations = []
    total = 0
    passing = 0

    for task in pipeline_input.extracted_tasks:

        if task.task_type == "FETCH":
            total += 1
            if task.dependencies:
                violations.append(
                    f"Task {task.id} (FETCH): must have empty dependencies, "
                    f"found {task.dependencies}"
                )
            else:
                passing += 1

        if task.task_type == "EXPAND":
            total += 1
            if task.target_output == "[ORCHESTRATION_END]":
                violations.append(
                    f"Task {task.id} (EXPAND): cannot write to [ORCHESTRATION_END] "
                    f"— only DECOMPOSE/OUTPUT may end orchestration"
                )
            else:
                passing += 1

    score = (passing / total) if total > 0 else 1.0
    return score, violations


def run_structural_check(pipeline_input: PipelineInput) -> StructuralResult:
    """
    Main structural checker entry point.
    Returns StructuralResult with scores, violations, and gate decision.
    """
    declared_set = build_declared_set(pipeline_input)

    namespace_score, namespace_violations = check_namespace(
        pipeline_input, declared_set
    )
    boundary_score, boundary_violations = check_boundaries(pipeline_input)

    all_violations = namespace_violations + boundary_violations

    structural_score = (
        0.60 * namespace_score +
        0.40 * boundary_score
    )

    gate_passed = structural_score >= STRUCTURAL_GATE_THRESHOLD

    return StructuralResult(
        namespace_score=round(namespace_score, 4),
        boundary_score=round(boundary_score, 4),
        structural_score=round(structural_score, 4),
        violations=all_violations,
        gate_passed=gate_passed,
    )
