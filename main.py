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