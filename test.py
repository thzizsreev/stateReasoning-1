from llm import OllamaClient, OpenRouterClient

def Extract_datanodes(phase_text: str, llm) -> dict:
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


    user_prompt = f"INSTRUCTIONS TO DECOMPOSE:\n{query}"
    print("Sending prompt to LLM...")
    
    try:
        raw_json_dict = llm.generate_json(system_prompt, user_prompt)
        print(f"Raw JSON Dictionary: {raw_json_dict}")
        return raw_json_dict
    except Exception as e:
        print(f"LLM Execution Failed: {e}")
        exit(1)


my_llm = OpenRouterClient()

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
datanodes = Extract_datanodes(query, my_llm)