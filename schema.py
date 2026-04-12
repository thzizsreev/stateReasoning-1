import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Any, Literal, Optional, Dict

# ==========================================
# 1. DEFINE THE MODELS
# ==========================================

class DataNode(BaseModel):
    id: str = Field(
        description="Unique identifier using bracket notation, e.g., '[Data_A]' or '[RULE_R1]'"
    )
    content: Any = Field(
        default=None, 
        description="The dynamic payload. Can be an int, dict, string, etc."
    )
    status: Literal["UNRESOLVED", "READY"] = "UNRESOLVED"

# ==========================================
# 2. THE TASK NODE
# ==========================================
class TaskNode(BaseModel):
    id: str = Field(
        description="Unique identifier for the task, e.g., 'Task_1_1'"
    )
    dependencies: List[str] = Field(
        description="Exact bracketed IDs of inputs required (e.g., ['[Data_A]', '[RULE_R1]'])."
    )
    instruction: str = Field(
        description="The isolated prompt for the Worker LLM, preserving bracketed IDs."
    )
    target_output: Optional[str] = Field(
        default=None,
        description=(
            "Bracketed ID of the DataNode this task creates; null allowed for DECOMPOSE "
            "or tasks that do not materialize a single output node."
        ),
    )
    task_type: Literal["COMPUTE", "DECOMPOSE", "OUTPUT"] = Field(
        description="'COMPUTE' if a Worker can solve this. 'DECOMPOSE' if it needs breakdown."
    )
    
    # --- ORCHESTRATION FIELDS (Invisible to LLM) ---
    
    # By setting a default, the LLM NEVER has to write "status": "PENDING" in its JSON.
    # Pydantic will automatically inject it for you when it parses the dictionary.
    
    status: Literal["PENDING", "COMPLETE"] = Field(default="PENDING")
    
    # Optional field: If a task gets DECOMPOSED later, you can track where the sub-tasks came from.
    parent_task: Optional[str] = Field(default=None)

# ==========================================
# 3. THE MASTER PLAN
# ==========================================
class PhasePlan(BaseModel):
    """This validates the clean {'tasks': [...]} JSON output from the Planner."""
    tasks: List[TaskNode]

class query_plan(BaseModel):
    data_nodes: Optional[List[DataNode]] = Field(default=None)
    task_nodes: Optional[List[TaskNode]] = Field(default=None)
# ==========================================
# 2. ENFORCEMENT DEMONSTRATION
# ==========================================
if __name__ == "__main__":
    
    # Scenario A: The LLM follows instructions perfectly.
    perfect_llm_json = """
    {
      "tasks": [
        {
          "id": "Task_Derive_B",
          "dependencies": ["Data_A", "RULE_R1", "Data_L"],
          "instruction": "Apply RULE_R1 to Data_A using Data_L.",
          "target_output": "Data_B",
          "task_type": "COMPUTE",
          "status": "PENDING"
        }
      ]
    }
    """
    
    print("--- Testing Perfect LLM Output ---")
    try:
        # This acts as the gatekeeper. It parses the JSON and builds the Python objects.
        validated_plan = PhasePlan.model_validate_json(perfect_llm_json)
        print("✅ Validation Successful!")
        print(f"Task Type is: {validated_plan.tasks[0].task_type}")
    except ValidationError as e:
        print("❌ Failed:", e)


    # Scenario B: The LLM Hallucinates a task_type ("CALCULATE" instead of "COMPUTE")
    hallucinated_llm_json = """
    {
      "tasks": [
        {
          "id": "Task_Derive_C",
          "dependencies": ["Data_B"],
          "instruction": "Calculate Point C.",
          "target_output": "Data_C",
          "task_type": "CALCULATE", 
          "status": "PENDING"
        }
      ]
    }
    """
    
    print("\n--- Testing Hallucinated LLM Output ---")
    try:
        validated_plan = PhasePlan.model_validate_json(hallucinated_llm_json)
    except ValidationError as e:
        print("🚨 VALIDATION CAUGHT AN ERROR (This is good!):")
        
        # Pydantic tells you exactly which field failed and why.
        for error in e.errors():
            print(f"Field '{error['loc'][0]}': {error['msg']}")