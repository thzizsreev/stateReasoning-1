from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from graph_engine import LocalGraph
from llm_routers import planner_agent, worker_agent
from prompts2 import PROMPTS


def _pick_next_task_id_from_plan(graph: LocalGraph, plan_task_ids: list[str]) -> str | None:
    ready = set(graph.get_ready_tasks())
    for tid in plan_task_ids:
        task = graph.task_nodes.get(tid)
        if task is None:
            continue
        if task.status != "PENDING":
            continue
        if tid in ready:
            return tid
    return None


def run() -> None:
    graph = LocalGraph()

    # Pre-load immutable knowledge as READY
    graph.insert_data("DEF_PRIMITIVES", PROMPTS["DEF_PRIMITIVES"], "READY")
    graph.insert_data("RULE_R1", PROMPTS["RULE_R1"], "READY")
    graph.insert_data("RULE_R2", PROMPTS["RULE_R2"], "READY")
    graph.insert_data("RULE_R3", PROMPTS["RULE_R3"], "READY")

    # Inject inputs
    graph.insert_data("Data_A", {"row": 2, "col": 3}, "READY")
    graph.insert_data("Data_L", 4, "READY")

    # Initial Plan
    print("Planning initial tasks...", flush=True)
    plan = planner_agent(PROMPTS["PHASE_1"], available_keys=list(graph.data_nodes.keys()))
    for task in plan.tasks:
        graph.insert_task(task)
    plan_task_ids = [t.id for t in plan.tasks]
    print(f"Planner returned {len(plan.tasks)} tasks. Next: {plan.next_task_id}", flush=True)

    # Execution Loop
    while True:
        ready_tasks = graph.get_ready_tasks()

        # Check for Completion or Deadlock
        if not ready_tasks:
            done = graph.data_nodes.get("PHASE_1_DONE")
            if done is not None and done.status == "READY":
                print("✅ Phase Complete!")
                break
            print("❌ Deadlock: No tasks ready, target not reached.")
            break

        # Execute exactly the "next task" chosen by the planner (no intermixing)
        task_id = plan.next_task_id if plan.next_task_id in ready_tasks else (  # type: ignore[assignment]
            _pick_next_task_id_from_plan(graph, plan_task_ids) or ready_tasks[0]
        )
        task = graph.task_nodes[task_id]
        isolated_context = graph.get_isolated_context(task_id)

        if task.task_type == "COMPUTE":
            print(f"Worker computing {task.id} -> {task.target_output}", flush=True)
            result = worker_agent(task.instruction, isolated_context)
            target_id = task.target_output or result.target_node_id
            graph.insert_data(target_id, result.computed_data, "READY")
            task.status = "COMPLETE"
            print(f"Computed {target_id}.", flush=True)

        elif task.task_type == "DECOMPOSE":
            print(f"Planner decomposing {task.id}", flush=True)
            # Decomposition is handled only by the planner (not the worker).
            # We treat this task's instruction as the "phase text" to plan from.
            subplan = planner_agent(task.instruction, available_keys=list(graph.data_nodes.keys()))
            for new_task in subplan.tasks:
                graph.insert_task(new_task)
            task.status = "COMPLETE"
            plan = subplan
            plan_task_ids = [t.id for t in subplan.tasks]
            print(f"Subplan returned {len(subplan.tasks)} tasks. Next: {subplan.next_task_id}", flush=True)
            continue

        # After compute, advance within the planner's task list deterministically (no worker-driven decomposition).
        nxt = _pick_next_task_id_from_plan(graph, plan_task_ids)
        if nxt is not None:
            plan.next_task_id = nxt

    print("\n=== FINAL data_nodes STATE ===")
    print(
        json.dumps(
            {k: v.model_dump() for k, v in graph.data_nodes.items()},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    run()

