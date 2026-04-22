from llm import OllamaClient, OpenRouterClient
import asyncio
import json
from pydantic import BaseModel, Field
from typing import Optional
from schema import DataNode, TaskNode, query_plan, DataNodesList, TaskNodesList, FinalPipelinePayload, RouterDecision
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from prompts import PROMPTS
from base import Neo4jOrchestrator
PYDANTIC_LOG_PATH = Path(__file__).resolve().parent / "docs" / "data" / "pydantic_outputs.jsonl"
#from config import configure_lm  # type: ignore
#from pipeline import QueryPipeline, process_query, execute_compute_node

def _clear_pydantic_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def _append_pydantic_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n\n")


def log_pydantic_output(label: str, obj: Any) -> None:
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = obj

    _append_pydantic_log(
        PYDANTIC_LOG_PATH,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "type": getattr(obj, "__class__", type(obj)).__name__,
            "data": data,
        },
    )

ROUTER_PROMPT = PROMPTS["ROUTER_PROMPT"] 
DECOMPOSER_PROMPT = PROMPTS["DECOMPOSER_PROMPT"]
DATA_EXTRACTOR_PROMPT = PROMPTS["DATA_EXTRACTOR_PROMPT"] 
COMPUTE_AGENT_PROMPT = PROMPTS["COMPUTE_AGENT_PROMPT"]

# def Extract_datanodes(query: str, llm) -> dict:
#     print("\nLLM Planner: Extracting Graph Topology from Text...")
    
#     # ---------------------------------------------------------
#     # THE DOMAIN-AGNOSTIC SYSTEM PROMPT
#     # ---------------------------------------------------------
#     system_prompt = COMPUTE_AGENT_PROMPT

#     user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{query}"
#     print("Sending prompt to LLM...")
    
#     try:
#         raw_json_dict = llm.generate_json(system_prompt, user_prompt)
#         print(f"Raw JSON Dictionary: {raw_json_dict}")
#         return raw_json_dict
#     except Exception as e:
#         print(f"LLM Execution Failed: {e}")
#         exit(1)

# ============================================================================
# 3. THE MICRO-AGENTS
# ============================================================================

async def run_semantic_router(user_query: str, llm) -> RouterDecision:
    """Level 0: Evaluates intent and returns a boolean routing matrix."""
    # print(f"\n[Router] Analyzing: '{user_query}'")
    # response = await client.beta.chat.completions.parse(
    #     model=MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": ROUTER_PROMPT},
    #         {"role": "user", "content": user_query}
    #     ],
    #     response_format=RouterDecision,
    #     temperature=0.0
    # )
    # return response.choices[0].message.parsed
    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{user_query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(ROUTER_PROMPT, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        decision = RouterDecision.model_validate(raw_json_dict)
        log_pydantic_output("RouterDecision", decision)
        return decision
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

async def run_data_extractor(user_query: str, llm) -> DataNodesList:
    """Child 1: Extracts hard facts."""
    # print("  -> [Fact Extractor] Scanning for explicit values...")
    # response = await client.beta.chat.completions.parse(
    #     model=MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": DATA_EXTRACTOR_PROMPT},
    #         {"role": "user", "content": user_query}
    #     ],
    #     response_format=DataNodesList,
    #     temperature=0.0
    # )
    # print("  <- [Fact Extractor] Complete.")
    # return response.choices[0].message.parsed
    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{user_query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(DATA_EXTRACTOR_PROMPT, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        data_nodes = DataNodesList.model_validate(raw_json_dict)
        log_pydantic_output("DataNodesList", data_nodes)
        return data_nodes
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

async def run_task_decomposer(user_query: str, llm) -> TaskNodesList:
    """Child 2: Slices instructions into DECOMPOSE chunks."""
    # print("  -> [Task Decomposer] Slicing workflow milestones...")
    # response = await client.beta.chat.completions.parse(
    #     model=MODEL_NAME, # Consider using a stronger model here (e.g., gpt-4o) if logic gets complex
    #     messages=[
    #         {"role": "system", "content": DECOMPOSER_PROMPT},
    #         {"role": "user", "content": user_query}
    #     ],
    #     response_format=TaskNodesList,
    #     temperature=0.0
    # )
    # print("  <- [Task Decomposer] Complete.")
    # return response.choices[0].message.parsed
    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{user_query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(DECOMPOSER_PROMPT, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        task_nodes = TaskNodesList.model_validate(raw_json_dict)
        log_pydantic_output("TaskNodesList.decomposer", task_nodes)
        return task_nodes
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

async def run_compute_agent(user_query: str, llm) -> TaskNodesList:
    """Child 3: Builds atomic COMPUTE tasks."""
    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{user_query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(COMPUTE_AGENT_PROMPT, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        task_nodes = TaskNodesList.model_validate(raw_json_dict)
        log_pydantic_output("TaskNodesList.compute", task_nodes)
        return task_nodes
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

async def process_query(user_query: str, llm) -> FinalPipelinePayload:
    # 1. Get the routing decision and snippets
    decision = await run_semantic_router(user_query, llm)
    print(f"  [Router Flags] Data: {decision.requires_data_extraction} | Decompose: {decision.requires_decomposition} | Compute: {decision.requires_computation}")

    pending_tasks = []
    final_payload = FinalPipelinePayload()

    # 2. Queue the required agents WITH THEIR SPECIFIC SNIPPETS
    if decision.requires_data_extraction:
        print(f"  -> Passing to Data Extractor: '{decision.data_snippet}'")
#         framed_prompt = f"""--- GLOBAL CONTEXT (For reference only. DO NOT evaluate your rules against this) ---
# {user_query}

# --- YOUR EXPLICIT TARGET (You MUST apply your rules ONLY to this text) ---
# {decision.decomposition_snippet}"""
        pending_tasks.append(run_data_extractor(user_query, llm))
        
    if decision.requires_decomposition:
        # Build the Tunnel Vision Prompt
        framed_prompt = f"""--- GLOBAL CONTEXT (For reference only. DO NOT evaluate your rules against this) ---
{user_query}

--- YOUR EXPLICIT TARGET (You MUST apply your rules ONLY to this text) ---
{decision.decomposition_snippet}"""
        
        pending_tasks.append(run_task_decomposer(framed_prompt, llm))
        
    if decision.requires_computation:
        # Build the Tunnel Vision Prompt
        framed_prompt = f"""--- GLOBAL CONTEXT (For reference only. DO NOT evaluate your rules against this) ---
{user_query}

--- YOUR EXPLICIT TARGET (You MUST apply your rules ONLY to this text) ---
{decision.computation_snippet}"""
        
        pending_tasks.append(run_compute_agent(framed_prompt, llm))

    # 3. Execute concurrently (Fan-Out)
    if pending_tasks:
        results = await asyncio.gather(*pending_tasks)
        
        # 4. Aggregate results (Fan-In)
        for result in results:
            if isinstance(result, DataNodesList):
                if final_payload.data_nodes is None:
                    final_payload.data_nodes = []
                if result.data_nodes:
                    final_payload.data_nodes.extend(result.data_nodes)
            elif isinstance(result, TaskNodesList):
                if final_payload.task_nodes is None:
                    final_payload.task_nodes = []
                if result.task_nodes:
                    final_payload.task_nodes.extend(result.task_nodes)

    log_pydantic_output("FinalPipelinePayload", final_payload)
    return final_payload
    
async def execute_compute_node(task: dict, dependencies_data: list | dict, db, llm) -> None:
    """
    Leaf Node Execution: Acts as a pure, stateless worker. 
    Takes the task and pre-fetched dependencies directly from arguments, 
    computes the math via LLM, and writes the result back to the graph.
    """
    task_id = task['id']
    instruction = task['instruction']
    target_id = task.get('target_output')

    print(f"\n⚙️ [COMPUTE WORKER] Initializing execution for {task_id}")

    if not target_id:
        print(f"  ⚠️ WARNING: Task {task_id} has no target_output. Execution aborted to prevent data loss.")
        # Mark complete to prevent an infinite loop, but skip DB insertion
        db.mark_task_complete(task_id)
        return

    print(f"  -> Received {len(dependencies_data)} dependencies from orchestrator.")

    # 1. Construct the Tunnel-Vision Prompt
    # We place the pre-fetched dependencies at the top as ground truth.
    user_prompt = f"DEPENDENCIES:\n{json.dumps(dependencies_data, indent=2)}\n\nINSTRUCTION:\n{instruction}"
    print("  -> Sending prompt to LLM Compute Agent...")

    # 2. Execute using your exact standard pattern
    try:
        raw_json_dict = llm.generate_json(PROMPTS["COMPUTE_EXECUTION_PROMPT"], user_prompt)
        print(f"  -> Raw JSON Dictionary: {raw_json_dict}")
        
        # Extract the keys we hardcoded the LLM to provide
        reasoning = raw_json_dict.get("reasoning", "No reasoning provided.")
        computed_content = raw_json_dict.get("content", None)
        
    except Exception as e:
        print(f"  ❌ LLM Execution Failed: {e}")
        return

    print(f"  -> Reasoning: {reasoning}")

    if computed_content is None:
        print(f"  ⚠️ [COMPUTE WORKER] Agent returned null content. Halting cascade.")
        return

    # 3. Map to existing schema and Upsert
    output_node = DataNode(
        id=target_id,
        content=computed_content,
        status="READY" 
    )
    log_pydantic_output("DataNode", output_node)
    db.upsert_data_node(data=output_node)
    db.mark_task_complete(task_id)
    
    print(f"🟢 [COMPUTE WORKER] Data successfully written to {target_id} output node: {output_node}")


if __name__ == "__main__":
#   PYDANTIC_LOG_PATH = Path(__file__).resolve().parent / "docs" / "data" / "pydantic_outputs.jsonl"
#   _clear_pydantic_log(PYDANTIC_LOG_PATH)
  #my_llm = OpenRouterClient()
  #configure_lm(backend="openrouter")
  #pipeline = QueryPipeline()
  
  query = """PHASE 1 — POINT DERIVATION (MANDATORY)

Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE].

PHASE 2 — LINE DERIVATION (MANDATORY)

Task 2.1: Create a Horizontal Line from [Data_A] to [Data_B]. Save output to node ID: [Data_Line_AB].
Task 2.2: Create a Vertical Line from [Data_B] to [Data_C]. Save output to node ID: [Data_Line_BC].
Task 2.3: Create a Horizontal Line from [Data_C] to [Data_D]. Save output to node ID: [Data_Line_CD].
Task 2.4: Create a Vertical Line from [Data_D] to [Data_A]. Save output to node ID: [Data_Line_DA].

Once all 4 lines are successfully derived, output a final status node named: [STATE_PHASE_2_DONE].

PHASE 3 — LINE FILLING (STRICT)

Task 3.1: Apply [RULE_R4_HORIZONTAL] to [Data_Line_AB]. Save the List of Filled objects to node ID: [Data_Fill_AB].
Task 3.2: Apply [RULE_R4_VERTICAL] to [Data_Line_BC]. Save the List of Filled objects to node ID: [Data_Fill_BC].
Task 3.3: Apply [RULE_R4_HORIZONTAL] to [Data_Line_CD]. Save the List of Filled objects to node ID: [Data_Fill_CD].
Task 3.4: Apply [RULE_R4_VERTICAL] to [Data_Line_DA]. Save the List of Filled objects to node ID: [Data_Fill_DA].

Once all 4 fill lists are successfully derived, output a final status node named: [STATE_PHASE_3_DONE]."""

  #result = asyncio.run(process_query(query, pipeline))
  print("\n=== FINAL PAYLOAD (process_query) ===")
  #print(result.model_dump_json(indent=2))
#   queries = [
#   """The starting coordinates are [Origin_X]=0 and [Origin_Y]=0. Phase 1: Plot the [Base_Line] using [Rule_Horizontal]. Phase 2: Extrude the square to a 3D cube by applying [Rule_Z_Axis_Depth].""",
#   #"""The radius of the circle is [Radius_R]=15cm. Calculate the total circumference using [Pi_Constant] and save to [Circle_Circumference].""",
#   #"""Calculate the hypotenuse of [Triangle_A] and save to [Hypotenuse_Length]. Separately, execute the polygon rendering sequence: first render the [Vertex_List], then apply the [Fill_Shader].""",
#   #"""Set the [Stroke_Width] to 5px. Calculate the interior angles of the [Hexagon_Shape] and save to [Interior_Angle_Value]. Also, run the fractal generation sequence: first duplicate the [Seed_Triangle], then scale it by [Scale_Factor_Half]."""
#     ]
#   test_task = {
#         "id": "[Task_Phase_1_Step_1]",
#         "dependencies": ["[Data_A]", "[Data_L]"],
#         "instruction": "Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B]",
#         "target_output": "[Data_B]",
#         "task_type": "COMPUTE",   
#         "status": "PENDING"
#     }

#   test_dependencies = {
#         "[Data_A]": {"row": 0, "col": 0},
#         "[Data_L]": 4,
#         "[RULE_R1]": "INPUT: A Point P, an Integer Length L. OUTPUT: A new Point Q = (P.row, P.col + L)."
#     }
#   test_task_node = TaskNode(
#         id="[Task_Phase_1_Step_1]",
#         dependencies=["[Data_A]", "[Data_L]"],
#         instruction="Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B]",
#         target_output="[Data_B]",
#         task_type="COMPUTE",
#         status="PENDING"
#     )
#   #db = Neo4jOrchestrator(password="password")
# #   db.upsert_task(test_task_node)
# #   result = asyncio.run(execute_compute_node(test_task, test_dependencies, db, my_llm))
# #   print(result)
# #   print("done")
# #   db.close()
  
#   query2= """Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
# Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
# Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

# Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE]."""
#   asyncio.run(process_query(query2, my_llm))