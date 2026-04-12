import json
import sys
from typing import List
from string import Template
from pydantic import ValidationError
import ollama

# Imports from your local files
from base import DynamicTaskGraph, TaskDecomposition
from prompts import PROMPTS

# ==========================================
# 1. LLM AGENT ROUTERS
# ==========================================
def call_llm(system_prompt: str, user_prompt: str, target_model: str = "llama3.1") -> dict:
    try:
        response = ollama.chat(
            model=target_model,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            options={"temperature": 0.0, "seed": 42},
            format="json"
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"\n❌ LLM ERROR: {e}")
        sys.exit(1)

def agent_decompose_phase(phase_text: str, current_data_nodes: List[str], available_skills: List[str]) -> TaskDecomposition:
    print("\n🧠 [DECOMPOSER AGENT] Analyzing Phase Instructions...")
    
    system = f"""You are a graph compilation engine. Break the Phase into independent logical steps.
    You MUST map the 'required_skills' array using ONLY these valid keys: {available_skills}
    
    OUTPUT STRICTLY IN JSON matching this schema:
    {{
      "phase_name": "String",
      "tasks": [
        {{
          "task_id": "Unique_Task_Name",
          "instruction": "What specifically to calculate/do",
          "required_skills": ["VALID_SKILL_KEY"],
          "dependencies": ["Data_A"], 
          "output_key": "Data_B"
        }}
      ]
    }}"""
    
    user = f"KNOWN DATA NODES IN GRAPH: {current_data_nodes}\n\nPHASE INSTRUCTIONS TO DECOMPOSE:\n{phase_text}"
    
    raw_json = call_llm(system, user)
    try:
        return TaskDecomposition(**raw_json)
    except ValidationError as e:
        print(f"❌ Decomposer schema failed: {e}")
        sys.exit(1)

def agent_execute_worker(task_id: str, graph: DynamicTaskGraph) -> dict:
    task_attr = graph.G.nodes[task_id]
    
    # Fetch exact Context from Graph
    deps = [u for u, v, d in graph.G.in_edges(task_id, data=True) if d['relation'] == 'REQUIRES']
    context = "\n".join([f"{dep} = {json.dumps(graph.G.nodes[dep]['content'])}" for dep in deps])
    
    # Fetch exact Skills from PROMPTS dictionary
    skills = "\n\n".join([PROMPTS.get(s, "") for s in task_attr['skills']])
    
    system = f"""{PROMPTS['SYS_001_PERSONA']}
    {PROMPTS['DEF_PRIMITIVES']}
    
    Read the skills, apply them to the context, and output the resultant data.
    OUTPUT STRICTLY IN JSON FORMAT. Do not nest the JSON inside markdown.
    """
    
    user = f"AVAILABLE SKILLS:\n{skills}\n\nKNOWN CONTEXT:\n{context}\n\nYOUR TASK:\n{task_attr['instruction']}\n\nGenerate the JSON output for {task_attr['output_key']}."
    
    print(f"\n⚙️ [WORKER AGENT] Executing {task_id}...")
    return call_llm(system, user)

# ==========================================
# 2. THE MULTI-PHASE PIPELINE
# ==========================================
def run_autonomous_pipeline():
    print("🚀 Initializing Dynamic Agentic Compiler...")
    graph = DynamicTaskGraph()
    
    # 1. Inject Initial State Variables
    dynamic_vars = {
        "START_NODE": "A",
        "LENGTH": "4"
    }
    graph.add_data_node("Data_A", {"row": 2, "col": 3})
    graph.add_data_node("Data_L", 4)
    
    # 2. Define execution sequence
    phases_to_execute = ["TASK_PHASE_1", "TASK_PHASE_2", "TASK_PHASE_3"]
    valid_skill_keys = list(PROMPTS.keys())
    
    # 3. Iterate through each Phase
    for phase_key in phases_to_execute:
        print(f"\n{'='*50}\n▶️ INITIATING {phase_key}\n{'='*50}")
        
        # Template Substitution
        raw_phase_text = PROMPTS[phase_key]
        substituted_phase_text = Template(raw_phase_text).safe_substitute(**dynamic_vars)
        
        known_nodes = [n for n, attr in graph.G.nodes(data=True) if attr.get("type") == "DataNode"]
        
        # Decompose Phase into DAG
        decomposition = agent_decompose_phase(substituted_phase_text, known_nodes, valid_skill_keys)
        graph.spawn_tasks(decomposition)
        
        # Resolve the DAG
        loop_count = 1
        while True:
            ready_tasks = graph.get_executable_tasks()
            
            if not ready_tasks:
                print(f"\n✅ DAG RESOLVED FOR {phase_key}: No pending executable tasks remain.")
                break
                
            print(f"\n--- {phase_key} EXECUTION LOOP {loop_count} ---")
            print(f"Found {len(ready_tasks)} executable tasks: {ready_tasks}")
            
            for task_id in ready_tasks:
                result = agent_execute_worker(task_id, graph)
                graph.complete_task(task_id, result)
                
            loop_count += 1

    print("\n" + "="*50)
    print("=== FINAL GRAPH DATA STATE ===")
    for n, attr in graph.G.nodes(data=True):
        if attr.get("type") == "DataNode":
            print(f"{n} -> {attr.get('content')} | Status: {attr.get('status')}")

if __name__ == "__main__":
    run_autonomous_pipeline()