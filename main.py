import json
from typing import List, Any, Dict
from pydantic import BaseModel, ValidationError
#from base import Neo4jOrchestrator
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from llm import OllamaClient, OpenRouterClient
from prompts import PROMPTS
from schema import PhasePlan, DataNode, TaskNode, query_plan
from base import Neo4jOrchestrator


task_list : List[TaskNode] = []


def _normalize_id(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return x
    if x.startswith("[") and x.endswith("]"):
        return x
    return f"[{x}]"
# ==========================================
# 3. THE TASK DIVISION FUNCTION (PLANNER)
# ==========================================
# def divide_phase_into_tasks(phase_text: str, llm) -> dict:
#     print("\nLLM Planner: Extracting Graph Topology from Text...")
    
#     # ---------------------------------------------------------
#     # THE DOMAIN-AGNOSTIC SYSTEM PROMPT
#     # ---------------------------------------------------------
#     system_prompt = """You are the Master Orchestrator for a General Purpose DAG Engine. 
# Your objective is to translate human-readable workflow instructions into atomic computational tasks and extract explicitly provided facts.

# COGNITIVE PROCESSING STEPS:
# 1. INTENT CLASSIFICATION (CRITICAL): Determine if the user is making an ACTIONABLE REQUEST (asking you to calculate, build, or process something) or a FACTUAL DECLARATION (just providing data, variables, or context). If it is a FACTUAL DECLARATION with no explicit command, you MUST skip task isolation and output an empty `task_nodes` array `[]`.
# 2. TASK ISOLATION: Create ONE task block per explicit action requested.
# 3. DEPENDENCY EXTRACTION (CRITICAL): You must extract EVERY SINGLE bracketed ID present in the task's instruction, EXCEPT the target output. You must treat EVERYTHING in brackets as a dependency, regardless of whether it represents raw data (e.g., [Data_A]), a physical constraint (e.g., [Length_L]), or a method/rule (e.g., [RULE_R1]). If it is in brackets, it goes in the dependencies array.
# 4. TARGET EXTRACTION: Identify the bracketed ID for the output. If the task does not explicitly generate a new data artifact (e.g., a "DECOMPOSE" task), you must set this to `null`.
# 5. INSTRUCTIONS: Rewrite the isolated task as a simple command. Do not use the bracketed IDs in the rewritten instruction if they are already listed in the dependencies.
# 6. ROUTING: Evaluate the complexity of the task:
#    - "COMPUTE": The task is a single, atomic action (e.g., summarizing, extracting, calculating).
#    - "DECOMPOSE": The task is a massive, multi-step project (e.g., "build an app", "design an architecture"). 
#      CRITICAL RULE FOR DECOMPOSE: Do NOT break the massive project into smaller tasks yourself. Output a SINGLE task block for the massive project and label it "DECOMPOSE". Let the downstream Orchestrator handle the breakdown.
#    - "OUTPUT": The task is ONLY setting a final status flag.

# STRICT SCHEMA ENFORCEMENT (CRITICAL):
# - OPTIONAL TARGET OUTPUT: `target_output` can be `null` if the task does not generate a specific piece of data. `id` and `instruction` must always be valid strings. 
# - NO INVENTED TYPES: `task_type` must be EXACTLY one of: "COMPUTE", "DECOMPOSE", or "OUTPUT".
# - EMPTY ARRAYS ARE VALID: If the intent is a FACTUAL DECLARATION, you MUST output `"task_nodes": []`. DO NOT infer implicit goals. DO NOT invent tasks like "Decompose the project" or "Store the data" just to be helpful. If the user didn't ask for an action, do not create one.
# - JSON ONLY: Output perfectly formatted JSON. No decoration or markdown comments like ```json. 

# MULTI-DOMAIN EXAMPLES (Focus strictly on the structure, not the subject matter):
# {
#   "data_nodes": [
#     {
#       "id": "[Example_Known_Data]",
#       "content": 42
#     }
#   ],
#   "task_nodes": [
#     {
#       "id": "[Task_Kinematics]",
#       "dependencies": ["[Initial_Velocity]", "[Time]", "[Gravity_Constant]", "[Formula_Distance]"],
#       "instruction": "Calculate the total distance traveled using the provided formula.",
#       "target_output": "[Final_Distance]",
#       "task_type": "COMPUTE"
#     },
#     {
#       "id": "[Task_Build_Backend]",
#       "dependencies": ["[System_Architecture_Doc]"],
#       "instruction": "Develop the complete microservice backend infrastructure according to the architecture document.",
#       "target_output": null,
#       "task_type": "DECOMPOSE"
#     },
#     {
#       "id": "[Task_Phase_1_Gate]",
#       "dependencies": ["[Derived_Point_B]", "[Derived_Point_C]", "[Derived_Point_D]"],
#       "instruction": "Mark Phase 1 as successfully completed.",
#       "target_output": "[STATE_PHASE_1_DONE]",
#       "task_type": "OUTPUT"
#     }
#   ]
# }

# ANTI-BIAS EXAMPLE (FACTUAL DECLARATION ONLY):
# USER CONTEXT: "The starting coordinates for Point A are x=5, y=10. The length of the side is 4."
# {
#   "data_nodes": [
#     {"id": "[Point_A]", "content": {"x": 5, "y": 10}},
#     {"id": "[Side_Length]", "content": 4}
#   ],
#   "task_nodes": [] 
# }
# """

#     user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{phase_text}"
#     print("Sending prompt to LLM...")
    
#     try:
#         raw_json_dict = llm.generate_json(system_prompt, user_prompt)
#         print(f"Raw JSON Dictionary: {raw_json_dict}")
#         return raw_json_dict
            
#     except Exception as e:
#         print(f"LLM Execution Failed: {e}")
#         exit(1)

def Extract_datanodes(text: str, llm) -> dict:
    print("\nLLM Planner: Extracting Graph Topology from Text...")
    
    # ---------------------------------------------------------
    # THE DOMAIN-AGNOSTIC SYSTEM PROMPT
    # ---------------------------------------------------------
    system_prompt = """You are the Master Orchestrator for a General Purpose DAG Engine. 
Your objective is to translate human-readable workflow instructions into atomic computational tasks and extract explicitly provided facts.

COGNITIVE PROCESSING STEPS:
1. INTENT CLASSIFICATION (CRITICAL): Determine if the user is making an ACTIONABLE REQUEST (asking you to calculate, build, or process something) or a FACTUAL DECLARATION (just providing data, variables, or context). If it is a FACTUAL DECLARATION with no explicit command, you MUST skip task isolation and output an empty `task_nodes` array `[]`.
2. TASK ISOLATION: Create ONE task block per explicit action requested.
3. DEPENDENCY EXTRACTION (CRITICAL): You must extract EVERY SINGLE bracketed ID present in the task's instruction, EXCEPT the target output. You must treat EVERYTHING in brackets as a dependency, regardless of whether it represents raw data (e.g., [Data_A]), a physical constraint (e.g., [Length_L]), or a method/rule (e.g., [RULE_R1]). If it is in brackets, it goes in the dependencies array.
4. TARGET EXTRACTION: Identify the bracketed ID for the output. If the task does not explicitly generate a new data artifact (e.g., a "DECOMPOSE" task), you must set this to `null`.
5. INSTRUCTIONS: Rewrite the isolated task as a simple command. Do not use the bracketed IDs in the rewritten instruction if they are already listed in the dependencies.
6. ROUTING: Evaluate the complexity of the task:
   - "COMPUTE": The task is a single, atomic action (e.g., summarizing, extracting, calculating).
   - "DECOMPOSE": The task is a massive, multi-step project (e.g., "build an app", "design an architecture"). 
     CRITICAL RULE FOR DECOMPOSE: Do NOT break the massive project into smaller tasks yourself. Output a SINGLE task block for the massive project and label it "DECOMPOSE". Let the downstream Orchestrator handle the breakdown.
   - "OUTPUT": The task is ONLY setting a final status flag.

STRICT SCHEMA ENFORCEMENT (CRITICAL):
- OPTIONAL TARGET OUTPUT: `target_output` can be `null` if the task does not generate a specific piece of data. `id` and `instruction` must always be valid strings. 
- NO INVENTED TYPES: `task_type` must be EXACTLY one of: "COMPUTE", "DECOMPOSE", or "OUTPUT".
- EMPTY ARRAYS ARE VALID: If the intent is a FACTUAL DECLARATION, you MUST output `"task_nodes": []`. DO NOT infer implicit goals. DO NOT invent tasks like "Decompose the project" or "Store the data" just to be helpful. If the user didn't ask for an action, do not create one.
- JSON ONLY: Output perfectly formatted JSON. No decoration or markdown comments like ```json. 

MULTI-DOMAIN EXAMPLES (Focus strictly on the structure, not the subject matter):
{
  "data_nodes": [
    {
      "id": "[Example_Known_Data]",
      "content": 42
    }
  ],
  "task_nodes": [
    {
      "id": "[Task_Kinematics]",
      "dependencies": ["[Initial_Velocity]", "[Time]", "[Gravity_Constant]", "[Formula_Distance]"],
      "instruction": "Calculate the total distance traveled using the provided formula.",
      "target_output": "[Final_Distance]",
      "task_type": "COMPUTE"
    },
    {
      "id": "[Task_Build_Backend]",
      "dependencies": ["[System_Architecture_Doc]"],
      "instruction": "Develop the complete microservice backend infrastructure according to the architecture document.",
      "target_output": null,
      "task_type": "DECOMPOSE"
    },
    {
      "id": "[Task_Phase_1_Gate]",
      "dependencies": ["[Derived_Point_B]", "[Derived_Point_C]", "[Derived_Point_D]"],
      "instruction": "Mark Phase 1 as successfully completed.",
      "target_output": "[STATE_PHASE_1_DONE]",
      "task_type": "OUTPUT"
    }
  ]
}

ANTI-BIAS EXAMPLE (FACTUAL DECLARATION ONLY):
USER CONTEXT: "The starting coordinates for Point A are x=5, y=10. The length of the side is 4."
{
  "data_nodes": [
    {"id": "[Point_A]", "content": {"x": 5, "y": 10}},
    {"id": "[Side_Length]", "content": 4}
  ],
  "task_nodes": [] 
}
"""


    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{text}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(system_prompt, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        return raw_json_dict
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)


# def process_plan_output(llm_response_dict: dict) -> PhasePlan:
#     """
#     Takes the parsed dictionary from the LLM, applies Pydantic validation, 
#     and returns the structured PhasePlan object.
#     """
#     print("⚙️  Parsing and Validating LLM Output...\n")
#     try:
#         # Use model_validate() because we are passing a Python dictionary, NOT a string
#         plan = PhasePlan.model_validate(llm_response_dict)

#         # Normalize IDs for consistent retrieval
#         normalized_tasks: List[TaskNode] = []
#         for t in plan.tasks:
#             normalized_tasks.append(
#                 t.model_copy(
#                     update={
#                         "id": _normalize_id(t.id),
#                         "dependencies": [_normalize_id(d) for d in t.dependencies],
#                         "target_output": (
#                             _normalize_id(t.target_output) if t.target_output else None
#                         ),
#                         "parent_task": _normalize_id(t.parent_task) if t.parent_task else None,
#                     }
#                 )
#             )
#         plan = plan.model_copy(update={"tasks": normalized_tasks})

#         # If there is a DECOMPOSE root, make it the parent for its children (unless already set)
#         decompose_roots = [t for t in plan.tasks if t.task_type == "DECOMPOSE"]
#         if decompose_roots:
#             root_id = decompose_roots[0].id
#             updated: List[TaskNode] = []
#             for t in plan.tasks:
#                 if t.id == root_id:
#                     updated.append(t)
#                     continue
#                 if t.parent_task is None:
#                     updated.append(t.model_copy(update={"parent_task": root_id}))
#                 else:
#                     updated.append(t)
#             plan = plan.model_copy(update={"tasks": updated})

#         return plan
#     except ValidationError as e:
#         print("❌ VALIDATION FAILED! The LLM broke the schema rules.")
#         # Safely handle the error location tuple (in case it's a root-level error)
#         for error in e.errors():
#             # Join the tuple into a readable path, e.g., "tasks -> 0 -> instruction"
#             error_location = " -> ".join([str(loc) for loc in error.get('loc', [])]) if error.get('loc') else "Root Payload"
#             print(f" - Field '{error_location}': {error['msg']}")
#         exit(1)
#     except Exception as e:
#         print(f"❌ CRITICAL ERROR: Could not parse. {e}")
#         exit(1)

def process_datanodes_output(llm_response_dict: dict) -> dict:
    """
    Takes the parsed dictionary from the LLM, applies Pydantic validation, 
    and returns the structured PhasePlan object.
    """
    print("⚙️  Parsing and Validating LLM Output...\n")
    try:
        # Use model_validate() because we are passing a Python dictionary, NOT a string
        plan = query_plan.model_validate(llm_response_dict)
        print(f"Plan: {plan}")
        return plan
    except ValidationError as e:
        print("❌ VALIDATION FAILED! The LLM broke the schema rules.")
        for error in e.errors():
            error_location = " -> ".join(str(x) for x in error.get("loc", ()))
            print(f" - Field '{error_location}': {error['msg']}")
        exit(1)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not parse. {e}")
        exit(1)


# def print_graph_nodes(plan: PhasePlan):
#     """Iterates through the validated Pydantic models and prints them."""
#     print(f"✅ Successfully extracted {len(plan.tasks)} Task Nodes:\n")

#     graph_dump: List[Dict[str, Any]] = []
#     for i, task in enumerate(plan.tasks, 1):
#         print(f"--- Task {i}: {task.id} ---")
#         print(f"  Type         : [{task.task_type}]")
#         print(f"  Dependencies : {task.dependencies}")
#         print(f"  Instruction  : \"{task.instruction}\"")
#         print(f"  Target Output: {task.target_output}")
#         print(f"  Status       : {task.status}")
#         print("-" * 40 + "\n")
#         task_list.append(task)
#         graph_dump.append(
#             {
#                 "index": i,
#                 "id": task.id,
#                 "task_type": task.task_type,
#                 "dependencies": task.dependencies,
#                 "instruction": task.instruction,
#                 "target_output": task.target_output,
#                 "status": task.status,
#                 "parent_task": task.parent_task,
#             }
#         )

#     out_dir = os.path.join(os.path.dirname(__file__), "docs", "data")
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, "graph_nodes.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump({"tasks": graph_dump}, f, indent=2, ensure_ascii=False)
#     print(f"Wrote task graph (same order as above) to: {out_path}")

def print_datanodes_nodes(plan: query_plan):
    """Iterates through the validated Pydantic models and prints them."""
    if(plan.data_nodes is not None and isinstance(plan.data_nodes, list)):
        for i, data_node in enumerate(plan.data_nodes, 1):
            print(f"--- Data Node {i}: {data_node.id} ---")
            print(f"  Content: {data_node.content}")
            print(f"  Status: {data_node.status}")
            print("-" * 40 + "\n")

    if(plan.task_nodes is not None and isinstance(plan.task_nodes, list)):
        for i, task_node in enumerate(plan.task_nodes, 1):
            print(f"--- Task Node {i}: {task_node.id} ---")
            print(f"  Dependencies: {task_node.dependencies}")
            print(f"  Instruction: {task_node.instruction}")
            print(f"  Target Output: {task_node.target_output}")
            print(f"  Status: {task_node.status}")
            print("-" * 40 + "\n")
        
    

# def insert_tasks_into_graph(plan: PhasePlan, db: Neo4jOrchestrator):
#     #db._clear_db()
#     db.ensure_semaphore()
#     for task in plan.tasks:
#         db.upsert_task(task)
#     db.recompute_semaphore()

def insert_datanodes_into_graph(plan: query_plan, db: Neo4jOrchestrator):
    db.ensure_semaphore()
    if plan.data_nodes is not None and isinstance(plan.data_nodes, list):
        for data_node in plan.data_nodes:
            db.upsert_data_node(data_node)
        db.recompute_semaphore()
    # else:
    #     db.upsert_data_node(plan.data_nodes)
    #     db.recompute_semaphore()

    if plan.task_nodes is not None and isinstance(plan.task_nodes, list):
            for task_node in plan.task_nodes:
                db.upsert_task(task_node)
            db.recompute_semaphore()
    # else:
    #     db.upsert_task(plan.task_nodes)
    #     db.recompute_semaphore()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":    

    my_llm = OpenRouterClient()
    db = Neo4jOrchestrator(password="password")
    # Mock data
    known_keys = ["DEF_PRIMITIVES", "RULE_R1", "RULE_R2", "RULE_R3"]
    phase_text = PROMPTS["TASK_PHASE_1"]

    predef_datanodes = {
        "data_nodes": [
            {
        "id": "[DEF_PRIMITIVES]",
        "content": PROMPTS["DEF_PRIMITIVES"]
    }, {
        "id": "[RULE_R1]",
        "content": PROMPTS["RULE_R1"]
    }, {
        "id": "[RULE_R2]",
        "content": PROMPTS["RULE_R2"]
    }, {
        "id": "[RULE_R3]",
        "content": PROMPTS["RULE_R3"]
    }, {
        "id": "[RULE_R4_HORIZONTAL]",
        "content": PROMPTS["RULE_R4_HORIZONTAL"]
    }, {
        "id": "[RULE_R4_VERTICAL]",
        "content": PROMPTS["RULE_R4_VERTICAL"]
    }
    ],
    "task_nodes": []
    }
    temp = process_datanodes_output(predef_datanodes)
    insert_datanodes_into_graph(temp, db)


    query = """construct a square with the point A(0,0) and side length 4. name the nodes as Data_A and Data_L for point A and side length respectively"""
    datanodes = Extract_datanodes(query, my_llm)
    datanodes = process_datanodes_output(datanodes)
    print_datanodes_nodes(datanodes)
    insert_datanodes_into_graph(datanodes, db)
    # phase_text = """First, analyze the [Client_Requirements_Doc] to extract a core [Feature_List]. Next, take that 
    # [Feature_List] and use it to build a full, production-ready [ECommerce_Web_Application]. Building a web application is
    #  a massive undertaking that requires designing  a database, writing frontend React code, and configuring backend APIs. 
    #  Once the web application is fully built, deploy it to AWS to generate a [Live_Production_URL]. Finally, trigger a 
    #  [STATE_DEPLOYMENT_COMPLETE]."""
    plan = Extract_datanodes(phase_text, my_llm)
    plan = process_datanodes_output(plan)
    print_datanodes_nodes(plan)
    insert_datanodes_into_graph(plan, db)
    
    # Step 3: Loop and Create Nodes
    # print("\n--- INJECTING TASK GRAPH ---")
    # for task in plan.tasks:
    #     db.insert_task(task)
        
    print("\n✅ Setup Complete. Open Neo4j Browser and run: MATCH (n) RETURN n")