from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class DataNode(BaseModel):
    id: str
    content: Any = None
    status: Literal["UNRESOLVED", "READY"]


class TaskNode(BaseModel):
    id: str
    task_type: Literal["COMPUTE", "DECOMPOSE"]
    dependencies: list[str]
    instruction: str
    target_output: str | None = None
    status: Literal["PENDING", "COMPLETE"] = "PENDING"

    @model_validator(mode="after")
    def _enforce_target_output(self) -> "TaskNode":
        if self.task_type == "COMPUTE" and not self.target_output:
            raise ValueError("COMPUTE tasks must set a non-empty target_output")
        return self


class PlannerResponse(BaseModel):
    tasks: list[TaskNode]
    next_task_id: str


class ComputeResult(BaseModel):
    computed_data: dict
    target_node_id: str

