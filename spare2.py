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

PYDANTIC_LOG_PATH = Path(__file__).resolve().parent / "docs" / "data" / "pydantic_outputs.jsonl"


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





def Extract_datanodes(query: str, llm) -> dict:
    print("\nLLM Planner: Extracting Graph Topology from Text...")
    
    # ---------------------------------------------------------
    # THE DOMAIN-AGNOSTIC SYSTEM PROMPT
    # ---------------------------------------------------------
    system_prompt = PROMPTS["COMPUTE_AGENT_PROMPT"]

    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(system_prompt, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        return raw_json_dict
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

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
        raw_json_dict = llm.generate_json(PROMPTS["ROUTER_PROMPT"], user_prompt)
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
        raw_json_dict = llm.generate_json(PROMPTS["DATA_EXTRACTOR_PROMPT"], user_prompt)
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
        raw_json_dict = llm.generate_json(PROMPTS["DECOMPOSER_PROMPT"], user_prompt)
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
        raw_json_dict = llm.generate_json(PROMPTS["COMPUTE_AGENT_PROMPT"], user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        task_nodes = TaskNodesList.model_validate(raw_json_dict)
        log_pydantic_output("TaskNodesList.compute", task_nodes)
        return task_nodes
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)

async def process_user_query(user_query: str, llm) -> FinalPipelinePayload:
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

if __name__ == "__main__":

  my_llm = OpenRouterClient()

  query = """construct a square with the point A(0,0) and side length 4. name the nodes as Data_A and Data_L for point A and side length respectively"""


  queries = [
  """The starting coordinates are [Origin_X]=0 and [Origin_Y]=0. Phase 1: Plot the [Base_Line] using [Rule_Horizontal]. Phase 2: Extrude the square to a 3D cube by applying [Rule_Z_Axis_Depth].""",
  #"""The radius of the circle is [Radius_R]=15cm. Calculate the total circumference using [Pi_Constant] and save to [Circle_Circumference].""",
  #"""Calculate the hypotenuse of [Triangle_A] and save to [Hypotenuse_Length]. Separately, execute the polygon rendering sequence: first render the [Vertex_List], then apply the [Fill_Shader].""",
  #"""Set the [Stroke_Width] to 5px. Calculate the interior angles of the [Hexagon_Shape] and save to [Interior_Angle_Value]. Also, run the fractal generation sequence: first duplicate the [Seed_Triangle], then scale it by [Scale_Factor_Half]."""
]

  _clear_pydantic_log(PYDANTIC_LOG_PATH)
  path = PYDANTIC_LOG_PATH
  # datanodes = asyncio.run(process_user_query(query, my_llm))
  for i, query in enumerate(queries):
    with open(path, "a", encoding="utf-8") as f:
      f.write(f"Query {i+1}: {query}\n")
    datanodes = asyncio.run(process_user_query(query, my_llm))
  
# datanodes = Extract_datanodes(query, my_llm)
  print("done")