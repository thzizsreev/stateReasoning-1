"""
DSPy Pipeline Orchestrator.

Replaces: process_query() and execute_compute_node() from the original agents.py.

Architecture preserved:
- Fan-out: Router → [DataExtractor, TaskDecomposer, ComputeAgent] in parallel
- Fan-in: results merged into FinalPipelinePayload
- Tunnel-vision framing: global_context / explicit_target always kept separate
- Empty-array / null contract: no bad data propagates downstream
- ExecutionWorker is invoked separately by the Neo4j orchestrator loop
"""

import asyncio
from typing import Optional

import dspy

from modules import (
    SemanticRouter,
    DataExtractor,
    TaskDecomposer,
    ComputeAgent,
    ExecutionWorker,
)
from schema import (
    DataNode,
    DataNodesList,
    TaskNodesList,
    FinalPipelinePayload,
    RouterDecision,
)
from pydantic_logging import log_pydantic_output


class QueryPipeline(dspy.Module):
    """
    Top-level DSPy module: routes a query and fans out to the appropriate agents.

    This is the DSPy equivalent of process_query(). It is itself a dspy.Module,
    so the entire pipeline is optimizable end-to-end with a single optimizer call.
    """

    def __init__(self):
        super().__init__()
        self.router = SemanticRouter()
        self.data_extractor = DataExtractor()
        self.task_decomposer = TaskDecomposer()
        self.compute_agent = ComputeAgent()

    def forward(self, user_query: str) -> FinalPipelinePayload:
        """
        Synchronous DSPy forward pass.
        Fan-out to agents is done sequentially here but individually async-compatible
        via aforward() below for drop-in replacement.
        """
        decision: RouterDecision = self.router(user_query=user_query)
        print(
            f"  [Router Flags] "
            f"Data: {decision.requires_data_extraction} | "
            f"Decompose: {decision.requires_decomposition} | "
            f"Compute: {decision.requires_computation}"
        )

        final_payload = FinalPipelinePayload()

        if decision.requires_data_extraction:
            print(f"  -> Passing to Data Extractor: '{decision.data_snippet}'")
            result: DataNodesList = self.data_extractor(
                global_context=user_query,
                explicit_target=decision.data_snippet,
            )
            if result.data_nodes:
                final_payload.data_nodes = (final_payload.data_nodes or []) + result.data_nodes

        if decision.requires_decomposition:
            result: TaskNodesList = self.task_decomposer(
                global_context=user_query,
                explicit_target=decision.decomposition_snippet,
            )
            if result.task_nodes:
                final_payload.task_nodes = (final_payload.task_nodes or []) + result.task_nodes

        if decision.requires_computation:
            result: TaskNodesList = self.compute_agent(
                global_context=user_query,
                explicit_target=decision.computation_snippet,
            )
            if result.task_nodes:
                final_payload.task_nodes = (final_payload.task_nodes or []) + result.task_nodes

        return final_payload

    async def aforward(self, user_query: str) -> FinalPipelinePayload:
        """
        Async forward: fan-out agents run concurrently (matching original asyncio.gather pattern).
        """
        decision: RouterDecision = self.router(user_query=user_query)
        print(
            f"  [Router Flags] "
            f"Data: {decision.requires_data_extraction} | "
            f"Decompose: {decision.requires_decomposition} | "
            f"Compute: {decision.requires_computation}"
        )

        loop = asyncio.get_event_loop()
        pending = []

        if decision.requires_data_extraction:
            pending.append(
                loop.run_in_executor(
                    None,
                    lambda: self.data_extractor(
                        global_context=user_query,
                        explicit_target=decision.data_snippet,
                    ),
                )
            )

        if decision.requires_decomposition:
            pending.append(
                loop.run_in_executor(
                    None,
                    lambda: self.task_decomposer(
                        global_context=user_query,
                        explicit_target=decision.decomposition_snippet,
                    ),
                )
            )

        if decision.requires_computation:
            pending.append(
                loop.run_in_executor(
                    None,
                    lambda: self.compute_agent(
                        global_context=user_query,
                        explicit_target=decision.computation_snippet,
                    ),
                )
            )

        final_payload = FinalPipelinePayload()
        if pending:
            results = await asyncio.gather(*pending)
            for result in results:
                if isinstance(result, DataNodesList) and result.data_nodes:
                    final_payload.data_nodes = (final_payload.data_nodes or []) + result.data_nodes
                elif isinstance(result, TaskNodesList) and result.task_nodes:
                    final_payload.task_nodes = (final_payload.task_nodes or []) + result.task_nodes

        return final_payload


class ComputeNodeExecutor(dspy.Module):
    """
    Leaf node executor — DSPy wrapper around ExecutionWorker.
    Replaces execute_compute_node() from the original agents.py.
    """

    def __init__(self):
        super().__init__()
        self.worker = ExecutionWorker()

    def forward(self, task: dict, dependencies_data: dict, db) -> None:
        task_id = task["id"]
        target_id = task.get("target_output")

        print(f"\n⚙️ [COMPUTE WORKER] Executing {task_id}")

        if not target_id:
            print(f"  ⚠️ No target_output on {task_id}. Marking complete, skipping write.")
            db.mark_task_complete(task_id)
            return

        output_node: Optional[DataNode] = self.worker(
            task=task,
            dependencies_data=dependencies_data,
        )

        if output_node is None:
            print(f"  ⚠️ Worker returned null. Cascade halted for {task_id}.")
            return

        db.upsert_data_node(data=output_node)
        db.mark_task_complete(task_id)
        print(f"🟢 [COMPUTE WORKER] Written to {target_id}: {output_node}")


async def process_query(user_query: str, pipeline: Optional[QueryPipeline] = None) -> FinalPipelinePayload:
    """
    Drop-in async replacement for the original process_query(user_query, llm).
    The `llm` argument is no longer needed — DSPy's configured LM is used globally.
    """
    if pipeline is None:
        pipeline = QueryPipeline()
    payload = await pipeline.aforward(user_query)
    log_pydantic_output("FinalPipelinePayload", payload)
    return payload


async def execute_compute_node(
    task: dict,
    dependencies_data: dict,
    db,
    executor: Optional[ComputeNodeExecutor] = None,
) -> None:
    """
    Drop-in async replacement for the original execute_compute_node().
    """
    if executor is None:
        executor = ComputeNodeExecutor()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, executor.forward, task, dependencies_data, db)

