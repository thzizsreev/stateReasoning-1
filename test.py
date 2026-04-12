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
2. FACT EXTRACTION: Look for explicitly stated values (e.g., "Length is 4", "Point A is x=5"). Extract these into the `data_nodes` array. If a variable or rule is mentioned (e.g., "[Data_A]") but NO explicit value is given, DO NOT place it in `data_nodes`.
3. TASK CHUNKING (LOOSE COUPLING): Do NOT blindly unroll complex workflows into dozens of microscopic tasks. You must group related sub-actions into broad "Logical Milestones" or "Phases". Create ONE task block per major milestone.
4. DEPENDENCY EXTRACTION: For each chunk, extract EVERY SINGLE bracketed ID present in that specific part of the instruction. You must treat EVERYTHING in brackets as a dependency (data, constraints, rules, or methods).
5. TARGET EXTRACTION: Identify the bracketed ID for the output. If the task does not explicitly generate a single, concrete artifact (e.g., a "DECOMPOSE" task representing a broad phase), you MUST set this to `null`.
6. INSTRUCTION SLICING (CRITICAL): For DECOMPOSE tasks, do NOT summarize, abstract, or rewrite the user's request. You must act as a precise text slicer. Copy the exact relevant sentences/semantics from the user's prompt into the `instruction` field verbatim. You MUST preserve the exact operational rules, constraints, and even the bracketed IDs within the text so the downstream engine has 100% of the original context to break it down later.
7. ROUTING: Evaluate the complexity of the task:
   - "COMPUTE": The task is a single, indivisible atomic action.
   - "DECOMPOSE": The task groups multiple logical steps or sub-actions to achieve a milestone. Output a SINGLE task block for this entire chunk. 
   - "OUTPUT": The task is ONLY setting a final status flag.

STRICT SCHEMA ENFORCEMENT (CRITICAL):
- NO EMPTY DATA SHELLS: Every object in `data_nodes` MUST have a `content` field. Never extract an ID into `data_nodes` if you do not know its exact numerical/textual value. 
- OPTIONAL TARGET OUTPUT: `target_output` can be `null`. `id` and `instruction` must always be valid strings. 
- NO SCHEMA FITTING: If the intent is a FACTUAL DECLARATION, output `"task_nodes": []`. If no explicit facts are provided, output `"data_nodes": []`. Do not hallucinate nodes just to fill the JSON.
- NO INVENTED TYPES: `task_type` must be EXACTLY one of: "COMPUTE", "DECOMPOSE", or "OUTPUT".
- JSON ONLY: Output perfectly formatted JSON. No decoration or markdown comments.

MULTI-DOMAIN EXAMPLES (Focus strictly on the structure, not the subject matter):

EXAMPLE 1 (LOOSELY COUPLED DECOMPOSITION WITH SLICING):
USER CONTEXT: "Construct a square. Phase 1: Apply [RULE_R1] to [Data_A] using length [Data_L] to derive [Data_B], and [RULE_R2] to derive [Data_D]. Phase 2: Connect lines."
{
  "data_nodes": [],
  "task_nodes": [
    {
      "id": "[Task_Phase_1]",
      "dependencies": ["[RULE_R1]", "[Data_A]", "[Data_L]", "[Data_B]", "[RULE_R2]", "[Data_D]"],
      "instruction": "Phase 1: Apply [RULE_R1] to [Data_A] using length [Data_L] to derive [Data_B], and [RULE_R2] to derive [Data_D].",
      "target_output": null,
      "task_type": "DECOMPOSE"
    },
    {
      "id": "[Task_Phase_2]",
      "dependencies": ["[Data_B]", "[Data_D]"],
      "instruction": "Phase 2: Connect lines.",
      "target_output": null,
      "task_type": "DECOMPOSE"
    }
  ]
}

EXAMPLE 2 (ANTI-BIAS / FACTUAL DECLARATION ONLY):
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