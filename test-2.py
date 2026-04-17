"""
DSPy test runner (mirrors `cognitive_tasks.py` behavior).

This script:
- Configures the global DSPy LM from `stateReasoning-1/.env`
- Runs the DSPy pipeline (`process_query`) to produce a `FinalPipelinePayload`
- Executes one sample COMPUTE task using `execute_compute_node`
  against a tiny in-memory mock DB (no Neo4j required)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List


def _prioritize_site_packages_over_local_dspy_folder() -> None:
    """
    Your repo contains a local folder named `stateReasoning-1/dspy/` which is NOT
    the DSPy library. The actual DSPy library is also imported as `dspy`.

    To ensure `import dspy` resolves to the installed DSPy library (site-packages),
    we move the script directory entry to the *end* of sys.path.

    Then we add `stateReasoning-1/dspy/` directly to sys.path so that modules like
    `pipeline.py` can be imported as `import pipeline`.
    """
    base_dir = Path(__file__).resolve().parent
    dspy_dir = base_dir / "dspy"

    # 1) Move the script directory (and '' = CWD) to end so it won't shadow site-packages `dspy`.
    while "" in sys.path:
        sys.path.remove("")
    sys.path.append("")

    base_str = str(base_dir)
    while base_str in sys.path:
        sys.path.remove(base_str)
    sys.path.append(base_str)

    # 2) Put the local pipeline module directory first.
    dspy_str = str(dspy_dir)
    if dspy_str not in sys.path:
        sys.path.insert(0, dspy_str)


class MockDB:
    """
    Minimal DB adapter implementing exactly what `ComputeNodeExecutor` needs.
    """

    def __init__(self) -> None:
        self.data_nodes: Dict[str, Any] = {}
        self.completed_tasks: List[str] = []

    def upsert_data_node(self, data) -> None:
        # `data` is a Pydantic model (DataNode); store by id for inspection
        self.data_nodes[getattr(data, "id", "UNKNOWN")] = data
        print(f"[MockDB] upsert_data_node: {data}")

    def mark_task_complete(self, task_id: str) -> None:
        self.completed_tasks.append(task_id)
        print(f"[MockDB] mark_task_complete: {task_id}")


def main() -> None:
    _prioritize_site_packages_over_local_dspy_folder()

    try:
        # Local DSPy-based pipeline modules (stateReasoning-1/dspy/*.py)
        from config import configure_lm  # type: ignore
        from pipeline import QueryPipeline, process_query, execute_compute_node  # type: ignore
    except Exception as e:
        print("Failed to import the local DSPy pipeline modules.")
        print("If the error mentions missing `dspy`, install DSPy in your venv:")
        print(r"  .\venv\Scripts\python.exe -m pip install -U dspy-ai")
        raise

    # 1) Configure LM once (loads OPENROUTER_* from stateReasoning-1/.env)
    configure_lm(backend="openrouter")

    # 2) Run pipeline (router + fan-out agents)
    pipeline = QueryPipeline()
    query2 = """Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE]."""

    result = asyncio.run(process_query(query2, pipeline))
    print("\n=== FINAL PAYLOAD (process_query) ===")
    print(result.model_dump_json(indent=2))

    # 3) Execute a sample compute node (mock DB, no Neo4j)
    db = MockDB()
    sample_task = {
        "id": "[Task_Phase_1_Step_1]",
        "dependencies": ["[Data_A]", "[Data_L]", "[RULE_R1]"],
        "instruction": "Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].",
        "target_output": "[Data_B]",
        "task_type": "COMPUTE",
        "status": "PENDING",
    }
    sample_dependencies = {
        "[Data_A]": {"row": 0, "col": 0},
        "[Data_L]": 4,
        "[RULE_R1]": "INPUT: A Point P, an Integer Length L. OUTPUT: A new Point Q = (P.row, P.col + L).",
    }

    asyncio.run(execute_compute_node(sample_task, sample_dependencies, db))

    print("\n=== MOCK DB STATE ===")
    print(f"data_nodes keys: {list(db.data_nodes.keys())}")
    print(f"completed_tasks: {db.completed_tasks}")


if __name__ == "__main__":
    main()