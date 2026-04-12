from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from schemas import DataNode, TaskNode


class LocalGraph:
    def __init__(self) -> None:
        self.data_nodes: dict[str, DataNode] = {}
        self.task_nodes: dict[str, TaskNode] = {}

    def insert_data(self, id: str, content: Any, status: str = "READY") -> None:
        self.data_nodes[id] = DataNode(id=id, content=content, status=status)  # type: ignore[arg-type]

    def insert_task(self, task_node: TaskNode) -> None:
        if task_node.id in self.task_nodes:
            i = 2
            new_id = f"{task_node.id}_{i}"
            while new_id in self.task_nodes:
                i += 1
                new_id = f"{task_node.id}_{i}"
            task_node = task_node.model_copy(update={"id": new_id})

        self.task_nodes[task_node.id] = task_node
        if not task_node.target_output:
            return
        if task_node.target_output not in self.data_nodes:
            self.data_nodes[task_node.target_output] = DataNode(
                id=task_node.target_output,
                content=None,
                status="UNRESOLVED",
            )

    def get_ready_tasks(self) -> list[str]:
        ready: list[str] = []
        for task_id, task in self.task_nodes.items():
            if task.status != "PENDING":
                continue
            deps_ok = True
            for dep_id in task.dependencies:
                dep = self.data_nodes.get(dep_id)
                if dep is None or dep.status != "READY":
                    deps_ok = False
                    break
            if deps_ok:
                ready.append(task_id)
        return ready

    def get_isolated_context(self, task_id: str) -> dict:
        task = self.task_nodes[task_id]
        ctx: dict[str, Any] = {}
        for dep_id in task.dependencies:
            if dep_id in self.data_nodes:
                ctx[dep_id] = self.data_nodes[dep_id].content
        return ctx

