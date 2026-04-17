"""
DSPy Modules for the Multi-Agent Pipeline.

Each agent wraps a DSPy predictor and:
- Handles tunnel-vision framing (global_context / explicit_target split)
- Enforces the empty-array / null contract on output
- Preserves the log_pydantic_output tracing pattern via structured logging
- Is individually optimizable via dspy.Optimizer
"""

import dspy
import json
from typing import Optional

from signatures import (
    SemanticRouterSignature,
    DataExtractorSignature,
    TaskDecomposerSignature,
    ComputeAgentSignature,
    ExecutionWorkerSignature,
)
from pydantic_logging import log_pydantic_output
from schema import (
    DataNode,
    TaskNode,
    DataNodesList,
    TaskNodesList,
    RouterDecision,
)


class SemanticRouter(dspy.Module):
    """
    Level 0: Evaluates intent and returns a routing decision with scoped snippets.
    Uses ChainOfThought because the Router externalizes scratchpad reasoning.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SemanticRouterSignature)

    def forward(self, user_query: str) -> RouterDecision:
        result = self.predict(user_query=user_query)

        decision = RouterDecision(
            requires_data_extraction=bool(result.requires_data_extraction),
            requires_decomposition=bool(result.requires_decomposition),
            requires_computation=bool(result.requires_computation),
            data_snippet=result.data_snippet or "",
            decomposition_snippet=result.decomposition_snippet or "",
            computation_snippet=result.computation_snippet or "",
        )

        log_pydantic_output("RouterDecision", decision)
        return decision


class DataExtractor(dspy.Module):
    """
    Child 1: Extracts hard-coded facts as DataNodes.
    Uses Predict (no scratchpad — structural recognition).
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(DataExtractorSignature)

    def forward(self, global_context: str, explicit_target: str) -> DataNodesList:
        result = self.predict(global_context=global_context, explicit_target=explicit_target)

        raw_nodes = result.data_nodes if result.data_nodes else []
        validated = []
        for node in raw_nodes:
            try:
                validated.append(
                    DataNode(
                        id=node["id"],
                        content=node.get("content"),
                        status=node.get("status", "READY"),
                    )
                )
            except Exception:
                pass

        output = DataNodesList(data_nodes=validated)
        log_pydantic_output("DataNodesList", output)
        return output


class TaskDecomposer(dspy.Module):
    """
    Child 2: Slices workflow phases into DECOMPOSE/COMPUTE task blueprints.
    Uses Predict (no scratchpad — structural recognition).
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TaskDecomposerSignature)

    def forward(self, global_context: str, explicit_target: str) -> TaskNodesList:
        result = self.predict(global_context=global_context, explicit_target=explicit_target)

        raw_nodes = result.task_nodes if result.task_nodes else []
        validated = []
        for node in raw_nodes:
            try:
                validated.append(
                    TaskNode(
                        id=node["id"],
                        instruction=node["instruction"],
                        dependencies=node.get("dependencies", []),
                        task_type=node.get("task_type", "DECOMPOSE"),
                        target_output=node.get("target_output"),
                        status=node.get("status", "PENDING"),
                    )
                )
            except Exception:
                pass

        output = TaskNodesList(task_nodes=validated)
        log_pydantic_output("TaskNodesList.decomposer", output)
        return output


class ComputeAgent(dspy.Module):
    """
    Child 3: Builds atomic COMPUTE task blueprints.
    Uses Predict (no scratchpad — structural recognition).
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ComputeAgentSignature)

    def forward(self, global_context: str, explicit_target: str) -> TaskNodesList:
        result = self.predict(global_context=global_context, explicit_target=explicit_target)

        raw_nodes = result.task_nodes if result.task_nodes else []
        validated = []
        for node in raw_nodes:
            try:
                validated.append(
                    TaskNode(
                        id=node["id"],
                        instruction=node["instruction"],
                        dependencies=node.get("dependencies", []),
                        task_type="COMPUTE",
                        target_output=node.get("target_output"),
                        status=node.get("status", "PENDING"),
                    )
                )
            except Exception:
                pass

        output = TaskNodesList(task_nodes=validated)
        log_pydantic_output("TaskNodesList.compute", output)
        return output


class ExecutionWorker(dspy.Module):
    """
    Leaf node executor: stateless, receives pre-fetched deps, writes one DataNode.
    Uses ChainOfThought because the Worker externalizes reasoning as audit trail.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ExecutionWorkerSignature)

    def forward(self, task: dict, dependencies_data: dict) -> Optional[DataNode]:
        task_id = task["id"]
        instruction = task["instruction"]
        target_id = task.get("target_output")

        if not target_id:
            print(f"  ⚠️ Task {task_id} has no target_output. Aborting.")
            return None

        result = self.predict(
            dependencies_json=json.dumps(dependencies_data, indent=2),
            instruction=instruction,
        )

        reasoning = getattr(result, "reasoning", None) or "No reasoning provided."
        content = getattr(result, "content", None)

        print(f"  -> Reasoning: {reasoning}")

        if content is None:
            print(f"  ⚠️ Worker returned null content for {task_id}. Halting cascade.")
            return None

        output_node = DataNode(
            id=target_id,
            content=content,
            status="READY",
        )
        log_pydantic_output("DataNode", output_node)
        return output_node

