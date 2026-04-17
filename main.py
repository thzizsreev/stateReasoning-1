import json
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, ValidationError
import asyncio
import sys
#from base import Neo4jOrchestrator
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Ensure UTF-8 output on Windows consoles (prevents emoji print crashes)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from llm import OllamaClient, OpenRouterClient
from prompts import PROMPTS
from schema import PhasePlan, DataNode, TaskNode, query_plan, FinalPipelinePayload
from base import Neo4jOrchestrator
from cognitive_tasks import run_data_extractor, run_task_decomposer, run_compute_agent, process_query, _clear_pydantic_log, execute_compute_node
from pathlib import Path

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

def insert_datanodes_into_graph(plan: FinalPipelinePayload, db: Neo4jOrchestrator):
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

"----------------------------- execution engine -----------------------------"

def check_terminal_state(db: Neo4jOrchestrator, terminal_id: str) -> bool:
        """Checks if the overarching program has reached its halting condition."""
        query = """
        MATCH (d:DataNode {id: $id, status: 'COMPLETE'})
        RETURN count(d) > 0 AS is_done
        """
        with db.driver.session() as session:
            result = session.run(query, id=db._normalize_id(terminal_id))
            return result.single()["is_done"]

def fetch_executable_tasks(db: Neo4jOrchestrator) -> list:
        """Pulls the frontier nodes where the semaphore has reached 0."""
        query = """
        MATCH (t:TaskNode {status: 'PENDING', unresolved_deps: 0})
        OPTIONAL MATCH (t)-[:GENERATES]->(out:DataNode)
        RETURN t.id AS id, 
               t.instruction AS instruction, 
               t.task_type AS task_type, 
               out.id AS target_output
        """
        with db.driver.session() as session:
            result = session.run(query)
            return [
                {
                    "id": record["id"], 
                    "instruction": record["instruction"], 
                    "task_type": record["task_type"],
                    "target_output": record["target_output"]
                } for record in result
            ]

async def process_and_upsert_query(query: str, llm, db, parent_task_id: Optional[str] = None):
    """
    End-to-end pipeline: Extracts nodes via the cognitive pipeline, 
    inserts them blindly, then dynamically draws top-down hierarchy edges in Neo4j.
    """
    print(f"\n--- PROCESSING QUERY ---")
    print(f"Text: {query[:80]}...") 
    
    # 1. THE UNIFIED COGNITIVE PIPELINE
    payload = await process_query(query, llm)
    
    # 2. GRAPH INSERTION (Schema stays clean, no parent attributes)
    insert_datanodes_into_graph(payload, db)
    
    # 3. TOP-DOWN HIERARCHY ENFORCEMENT
    if parent_task_id and payload.task_nodes:
        print(f"  [Graph Update] Drawing [:CREATES] edges from Parent: {parent_task_id}")
        with db.driver.session() as session:
            for task in payload.task_nodes:
                session.run("""
                    MATCH (parent:TaskNode {id: $parent_id})
                    MATCH (child:TaskNode {id: $child_id})
                    MERGE (parent)-[:CREATES]->(child)
                """, parent_id=db._normalize_id(parent_task_id), child_id=db._normalize_id(task.id))
    
    print("--- UPSERT COMPLETE ---\n")
    return payload


def fetch_resolved_dependencies_for_task(db: Neo4jOrchestrator, task_id: str) -> List[Dict[str, Any]]:
    """
    Fetches COMPLETE dependency DataNodes for a task.
    Returns a JSON-serializable list of {id, content}.
    """
    query = """
    MATCH (d:DataNode)-[:REQUIRES]->(t:TaskNode {id: $task_id})
    WHERE coalesce(d.status, 'UNRESOLVED') = 'COMPLETE'
    RETURN d.id AS id, d.content AS content
    ORDER BY d.id
    """
    with db.driver.session() as session:
        result = session.run(query, task_id=db._normalize_id(task_id))
        deps: List[Dict[str, Any]] = []
        for record in result:
            content_raw = record["content"]
            try:
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except Exception:
                content = content_raw
            deps.append({"id": record["id"], "content": content})
        return deps


def fetch_and_print_fill_outputs(db: Neo4jOrchestrator) -> Dict[str, Any]:
    """
    Fetch contents of fill output nodes and print a combined JSON output.
    """
    output_ids = ["[Data_Fill_AB]", "[Data_Fill_BC]", "[Data_Fill_CD]", "[Data_Fill_DA]"]
    combined: Dict[str, Any] = {}

    query = """
    MATCH (d:DataNode)
    WHERE d.id IN $ids
    RETURN d.id AS id, d.content AS content, d.status AS status
    """
    with db.driver.session() as session:
        result = session.run(query, ids=output_ids)
        for record in result:
            node_id = record["id"]
            content_raw = record["content"]
            try:
                content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
            except Exception:
                content = content_raw
            combined[node_id] = {"status": record["status"], "content": content}

    # Ensure keys exist even if missing in DB
    for oid in output_ids:
        combined.setdefault(oid, {"status": None, "content": None})

    print("\n=== COMBINED FILL OUTPUTS ===")
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return combined


async def run_execution_engine(db: Neo4jOrchestrator, llm, terminal_node: str):
    print("\n🚀 [ENGINE] Booting Runtime Execution Engine...")
    
    while True:
        # 1. The Halting Problem (Target Satisfaction)
        if check_terminal_state(db, terminal_node):
            print(f"\n🏁 [ENGINE] HALTING: Terminal state {terminal_node} successfully resolved.")
            break

        # 2. Fetch the Frontier
        executable_tasks = fetch_executable_tasks(db)

        if not executable_tasks:
            # Prevent CPU thrashing while waiting for external LLM/Compute calls to finish
            print("  [ENGINE] Graph is blocked or processing. Waiting for semaphore updates...")
            await asyncio.sleep(2)
            continue

        # 3. Process the Frontier Concurrently
        for task in executable_tasks:
            print(f"\n⚡ [ENGINE] Executing Node: {task['id']} (Type: {task['task_type']})")
            
            if task['task_type'] == 'DECOMPOSE':
                # Route to Cognitive Pipeline
                print(f"  -> Passing instruction to Compiler...")
                # Note: This is the recursive expansion of the graph
                await process_and_upsert_query(task['instruction'], llm, db, parent_task_id=task['id'])
                db.mark_task_complete(task['id'])
                
            elif task['task_type'] == 'COMPUTE':
                # Route to Physical Execution Worker
                print(f"  -> Passing to Math Worker...")
                
                deps = fetch_resolved_dependencies_for_task(db, task["id"])
                await execute_compute_node(task, deps, db, llm)

        # Sleep briefly to let the graph transactions settle and recompute semaphores
        await asyncio.sleep(1)    



# def mark_task_complete(db: Neo4jOrchestrator, task_id: str) -> None:
#         """Advances the state machine after execution."""
#         query = """
#         MATCH (t:TaskNode {id: $id})
#         SET t.status = 'COMPLETE'
#         """
#         with db.driver.session() as session:
#             session.run(query, id=db._normalize_id(task_id))
# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":  

    def mock_RAG(instruction: str, db) -> str:
        """
        Simulates querying the Vector Database. 
        Fetches the required dependencies (Primitives), injects them into the graph, 
        and returns the execution template for the orchestrator to chunk.
        """
        print(f"\n[Experiential RAG] Searching memory for: '{instruction[:50]}...'")
        
        # 1. Fetch & Load Dependencies (Primitives & Rules)
        print("  -> Match found. Loading required dependencies into memory...")
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
        
        # 2. Fetch & Return the Template
        print("  -> Dependencies loaded. Returning execution template.")
        phase_text = PROMPTS["SQUARE_CONSTRUCTION"]
        
        return phase_text  
    "-------------- real code---------------------------"
    my_llm = OpenRouterClient()
    db = Neo4jOrchestrator(password="password")
    
    PYDANTIC_LOG_PATH = Path(__file__).resolve().parent / "docs" / "data" / "pydantic_outputs.jsonl"
    _clear_pydantic_log(PYDANTIC_LOG_PATH)
    # Mock data
    known_keys = ["DEF_PRIMITIVES", "RULE_R1", "RULE_R2", "RULE_R3"]
    phase_text = PROMPTS["TASK_PHASE_1"]


    db._clear_db()
    query = """construct a square with the point A(0,0) and side length 4. name the nodes as Data_A and Data_L for point A and side length respectively"""
    datanodes = asyncio.run(process_query(query, my_llm))
    #datanodes = process_datanodes_output(datanodes)
    ##print_datanodes_nodes(datanodes)
    insert_datanodes_into_graph(datanodes, db)
    # phase_text = """First, analyze the [Client_Requirements_Doc] to extract a core [Feature_List]. Next, take that 
    # [Feature_List] and use it to build a full, production-ready [ECommerce_Web_Application]. Building a web application is
    #  a massive undertaking that requires designing  a database, writing frontend React code, and configuring backend APIs. 
    #  Once the web application is fully built, deploy it to AWS to generate a [Live_Production_URL]. Finally, trigger a 
    #  [STATE_DEPLOYMENT_COMPLETE]."""


    # plan = Extract_datanodes(phase_text, my_llm)
    # plan = process_datanodes_output(plan)
    # print_datanodes_nodes(plan)
    # insert_datanodes_into_graph(plan, db)
    
    # Step 3: Loop and Create Nodes
    # print("\n--- INJECTING TASK GRAPH ---")
    # for task in plan.tasks:
    #     db.insert_task(task)
        
    # ========================================================================
    # 2. RAG RETRIEVAL & PRIMITIVE INJECTION
    # ========================================================================
    print("\n=== STAGE 2: FETCHING EXPERIENTIAL MEMORY ===")
    # Pass the instruction to the mock RAG. It handles injecting the primitives 
    # internally and returns the template we need to decompose.
    template_text = mock_RAG(datanodes.task_nodes[0].instruction, db)

    # ========================================================================
    # 3. TEMPLATE CHUNKING & HIERARCHY CONJUNCTION
    # ========================================================================
    print("\n=== STAGE 3: DECOMPOSING TEMPLATE & BUILDING HIERARCHY ===")
    # Process the returned template, passing the root_task_id so the function 
    # natively draws the [:CREATES] edges from the root to these new sub-tasks.
    root_task_id = datanodes.task_nodes[0].id
    plan_payload = asyncio.run(process_and_upsert_query(template_text, my_llm, db, parent_task_id=root_task_id))
    
    # ========================================================================
    # 4. ADVANCE STATE
    # ========================================================================
    with db.driver.session() as session:
        session.run("""
            MATCH (t:TaskNode {id: $id}) 
            SET t.status = 'COMPLETE'
        """, id=db._normalize_id(root_task_id))
        print(f"\n🟢 ROOT TASK '{root_task_id}' COMPLETED (Successfully decomposed).")

    print("\n✅ Setup Complete. Open Neo4j Browser and run: MATCH (n) RETURN n")
    asyncio.run(run_execution_engine(db, my_llm, terminal_node="[STATE_PHASE_3_DONE]"))
    fetch_and_print_fill_outputs(db)