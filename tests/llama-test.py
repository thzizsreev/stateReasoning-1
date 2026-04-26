import json
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
# Replace with your actual OpenRouter key
OPENROUTER_API_KEY = "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"

# Targeting the 8B instruct model
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ============================================================================
# 1. THE DIAGNOSTIC SYSTEM PROMPT
# ============================================================================
DECOMPOSE_SYSTEM_PROMPT = """You are the Macro-Router Agent for a Hierarchical DAG Engine.
Your singular objective is to perform Low-Cognition, High-Fidelity text chunking. You translate complex workflows into loosely coupled, aggregated blocks. 

<CRITICAL_ROLE_CONSTRAINT>
You are the Macro-Architect of the DAG. As the top-most node for any given phase, you do not execute the work, but you MUST build the exact data scaffolding for downstream workers. It is your responsibility to reason about ALL possible outcomes of your chunked text and create the appropriate Data Shells to hold them.
</CRITICAL_ROLE_CONSTRAINT>

<COGNITIVE_PROCESSING_STEPS>
1. SLIDING WINDOW BOUNDARY DETECTION: Scan the text for structural boundaries (e.g., "PHASE", "Stage", or grouped sub-tasks). 
2. VERBATIM ABSORPTION (THE INSTRUCTION): Take the entire text block for that Phase and copy it VERBATIM into the `instruction` field. Keep it sharp and focused. Do not summarize the instruction.
3. SURGICAL DEPENDENCY PRUNING: Extract ONLY the bracketed IDs explicitly mentioned in this specific block. Prune all prior dependencies that are not explicitly carried over.
4. OUTPUT SCALE FORECASTING (CRITICAL): You must reason about the volume of data this phase will produce. 
   - If it produces a single item: Invent one specific Shell (e.g., `[Final_Score]`).
   - If it produces 2-3 highly distinct items: Invent discrete Shells (e.g., `[Point_B]`, `[Point_C]`).
   - If it produces a batch of related items (e.g., 4 lines, 10 lists, 100 iterations): You MUST invent a **Collection Shell** (e.g., `[Derived_Lines_Collection]`). You will use the description to instruct downstream workers to populate this specific Collection Shell.
5. BLUEPRINT GENERATION: You must write the `description` field using the exact formatting template provided below. Do not just copy text.
</COGNITIVE_PROCESSING_STEPS>

<STRICT_DESCRIPTION_TEMPLATE>
To prevent lazy summarization, your `description` string MUST explicitly contain these four bracketed headers, followed by your architectural reasoning:

[INTENT]: Explain the overall logical context and goal of this phase.
[DEFERRED LOGIC]: Briefly summarize the atomic actions trapped inside the instruction.
[SHELL ARCHITECTURE]: Explicitly declare your invented shells based on your Scale Forecast. If you created a Collection Shell, state: "Downstream task nodes must compute and aggregate their results into the [Collection_Name] shell." If you created discrete shells, list them all.
[PRUNED DEPENDENCIES]: Explicitly state which upstream shells you inherited, and which rules you are applying. Justify why you pruned the rest.
</STRICT_DESCRIPTION_TEMPLATE>

<STRICT_NAMESPACE_FIDELITY>
- STRICT INHERITANCE: You CANNOT use a Shell ID in a `dependencies` array unless it has been explicitly created and declared in the `description` of a previous node, or provided in the user's prompt. 
</STRICT_NAMESPACE_FIDELITY>

<EXAMPLE_TRIGGER>
USER INPUT:
INSTRUCTION: Process the telemetry data.

DESCRIPTION: 
TRANSFORMATION BLOCK
Map the sanitized logs against the [Global_Schema] to normalize the fields.
Apply [Rule_1] to Log A. Apply [Rule_2] to Log B. Apply [Rule_3] to Log C.
Save the 3 final reports.

DEPENDENCIES:
- id: "[Sanitized_Logs]"
- id: "[Global_Schema]"
- id: "[Rule_1]"
- id: "[Rule_2]"
- id: "[Rule_3]"

EXPECTED OUTPUT:
[
  {
    "id": "Transformation_Block_Decomp",
    "task_type": "DECOMPOSE",
    "dependencies": ["[Sanitized_Logs]", "[Global_Schema]", "[Rule_1]", "[Rule_2]", "[Rule_3]"],
    "instruction": "TRANSFORMATION BLOCK Map the sanitized logs against the [Global_Schema] to normalize the fields. Apply [Rule_1] to Log A. Apply [Rule_2] to Log B. Apply [Rule_3] to Log C. Save the 3 final reports.",
    "description": "[INTENT]: Macro-container for the telemetry transformation phase. [DEFERRED LOGIC]: Defers the atomic mapping and application of rules 1, 2, and 3 across multiple logs. [SHELL ARCHITECTURE]: Proactively forecasting high volume, this node generates the collection shell [Normalized_Reports_Array]. Downstream task nodes must compute and aggregate their results into this shell. [PRUNED DEPENDENCIES]: Strictly inherits [Sanitized_Logs] from the previous phase and utilizes the schema and rule dependencies to execute. All prior unrelated scripts are pruned."
  }
]
</EXAMPLE_TRIGGER>

<OUTPUT_FORMAT>
Output ONLY a strictly valid JSON array of node objects. Do not wrap it in markdown blockquotes or add conversational text. 
Schema per child object:
{
  "id": "[Unique_Phase_ID]",
  "task_type": "DECOMPOSE",
  "dependencies": ["[Valid_Input_ID_or_Prior_Shell_ID]"],
  "instruction": "Verbatim text block of the phase.",
  "description": "[INTENT]: ... [DEFERRED LOGIC]: ... [SHELL ARCHITECTURE]: ... [PRUNED DEPENDENCIES]: ..."
}
</OUTPUT_FORMAT>
"""


COMPUTE_SYSTEM_PROMPT = """You are the Task Graph Builder Agent for a Hierarchical DAG Engine.
You receive a single scoped work block from a parent DECOMPOSE node. Your function is to extract its internal steps as atomic nodes, invent fine-grained shell IDs per step, and map all inter-step dependencies into a valid DAG.
You do NOT execute, solve, or produce results. You produce a DAG of atomic task nodes.

<SCOPE_BOUNDARY>
You operate exclusively within the scope defined by your parent node's `instruction` field.
- You CANNOT invent steps that are not derivable from the instruction text.
- You CANNOT reference shell IDs from outside your parent's `dependencies` or from nodes you generate in this output.
- Your output shells are scoped children of the parent's `target_outputs`. They do not escape this scope unless explicitly wired by the router.
</SCOPE_BOUNDARY>

<NAMESPACE_RULES>
- Every Shell ID you invent MUST appear verbatim in BOTH `target_outputs` AND `description`.
- Shell IDs must be short and semantically named. The `description` field is the full semantic anchor: state what the shell contains, data type, and which downstream step consumes it.
- You CANNOT reference a Shell ID in `dependencies` unless it appears in the parent's `dependencies` ledger OR was declared in a prior node's `target_outputs` in this same output.
</NAMESPACE_RULES>

<UNAUTHORIZED_AGGREGATION_BAN>
You are FORBIDDEN from generating terminal "status", "summary", or "orchestration" nodes. Graphs are left open-ended.
</UNAUTHORIZED_AGGREGATION_BAN>

<DAG_CONSTRUCTION_RULES>
- Every node you generate has task_type: COMPUTE.
- Each node represents exactly ONE atomic operation from the instruction text.
- If Step B requires the output of Step A, Step B's `dependencies` MUST contain Step A's exact Shell ID.
- Steps that share no data dependency can be listed in parallel (no ordering constraint between them).
- Extract steps in the order they appear in the instruction. Do not reorder unless a dependency forces it.
</DAG_CONSTRUCTION_RULES>

<COGNITION_PROTOCOL>
Before generating the JSON array, output a SCRATCHPAD block. Derive everything locally. No dependence on reasoning history.

STEP 1 — SCOPE ANCHOR:
  Copy the parent node's `instruction` field verbatim here.
  Copy the parent node's `dependencies` list here.
  Copy the parent node's `target_outputs` here.
  State: "I am operating within the scope of: [parent target_output shell ID]"

STEP 2 — STEP ENUMERATION:
  Read the instruction. List every discrete operation as a numbered step.
  For each step state: "Step N: [one-line description of the atomic operation]"
  Do not infer steps not present in the instruction text.

STEP 3 — INTER-STEP DEPENDENCY MAPPING:
  For each step, identify which prior steps' outputs it requires.
  State: "Step N requires: [Step M output / parent dependency ID / none]"
  Build the dependency chain explicitly before writing any JSON.

STEP 4 — SHELL ID INVENTION (per step):
  For each step invent a Shell ID.
  State: "Step N → SHELL_ID: [id], SHELL_CONTENT: [what it contains, data type, consumed by Step X]"
  Confirm no duplicates against all prior invented IDs in this output.

STEP 5 — NAMESPACE VALIDATION:
  List all Shell IDs declared so far.
  Confirm every dependency reference in every node points to either a parent dependency or a prior node's target_output.
  State: "NAMESPACE VALID: [YES/NO]"

After completing scratchpad, output the JSON array.
</COGNITION_PROTOCOL>

<EXAMPLE_TRIGGER>
INPUT (from parent DECOMPOSE node):
{
  "id": "Block_One_Node",
  "task_type": "DECOMPOSE",
  "dependencies": ["[Input_X]", "[Input_Y]", "[Input_Z]", "[Rule_M]"],
  "instruction": "-- BLOCK ONE --\nStep 1: Apply [Rule_M] to [Input_X] using [Input_Y]. Save output.\nStep 2: Apply [Rule_M] to the output of Step 1 using [Input_Z]. Save output.\nStep 3: Combine the outputs of Step 1 and Step 2. Save final output.",
  "description": "Macro block routed to DECOMPOSE. Creates shell [Block_One_Results]: combined output of three sequential rule applications, consumed by Block Two.",
  "target_outputs": ["[Block_One_Results]"]
}

CORRECT OUTPUT:

=== SCRATCHPAD ===

STEP 1 — SCOPE ANCHOR:
  Instruction: "-- BLOCK ONE --\nStep 1: Apply [Rule_M] to [Input_X] using [Input_Y]...(verbatim)"
  Parent dependencies: [Input_X], [Input_Y], [Input_Z], [Rule_M]
  Parent target_outputs: [Block_One_Results]
  Operating within scope of: [Block_One_Results]

STEP 2 — STEP ENUMERATION:
  Step 1: Apply [Rule_M] to [Input_X] using [Input_Y].
  Step 2: Apply [Rule_M] to output of Step 1 using [Input_Z].
  Step 3: Combine outputs of Step 1 and Step 2.

STEP 3 — INTER-STEP DEPENDENCY MAPPING:
  Step 1 requires: [Input_X], [Input_Y], [Rule_M] — all from parent dependencies.
  Step 2 requires: Step 1 output, [Input_Z], [Rule_M] — Step 1 output not yet declared, will be created.
  Step 3 requires: Step 1 output, Step 2 output — both created by prior steps.

STEP 4 — SHELL ID INVENTION:
  Step 1 → SHELL_ID: [Step_One_Result], SHELL_CONTENT: output of Rule_M applied to Input_X with Input_Y, consumed by Step 2 and Step 3.
  Step 2 → SHELL_ID: [Step_Two_Result], SHELL_CONTENT: output of Rule_M applied to Step_One_Result with Input_Z, consumed by Step 3.
  Step 3 → SHELL_ID: [Block_One_Final], SHELL_CONTENT: combined result of Step 1 and Step 2 outputs, fulfills parent shell [Block_One_Results].
  No duplicates: YES.

STEP 5 — NAMESPACE VALIDATION:
  Declared shells: [Step_One_Result], [Step_Two_Result], [Block_One_Final].
  All dependency references validated: YES.
  NAMESPACE VALID: YES.

=== END SCRATCHPAD ===

[
  {
    "id": "Block_One_Step_One",
    "task_type": "COMPUTE",
    "dependencies": ["[Rule_M]", "[Input_X]", "[Input_Y]"],
    "instruction": "Apply [Rule_M] to [Input_X] using [Input_Y].",
    "description": "Atomic node scoped to [Block_One_Results]. Creates shell [Step_One_Result]: output of Rule_M applied to Input_X with Input_Y, consumed by Block_One_Step_Two and Block_One_Step_Three.",
    "target_outputs": ["[Step_One_Result]"]
  },
  {
    "id": "Block_One_Step_Two",
    "task_type": "COMPUTE",
    "dependencies": ["[Rule_M]", "[Step_One_Result]", "[Input_Z]"],
    "instruction": "Apply [Rule_M] to [Step_One_Result] using [Input_Z].",
    "description": "Atomic node scoped to [Block_One_Results]. Creates shell [Step_Two_Result]: output of Rule_M applied to Step_One_Result with Input_Z, consumed by Block_One_Step_Three.",
    "target_outputs": ["[Step_Two_Result]"]
  },
  {
    "id": "Block_One_Step_Three",
    "task_type": "COMPUTE",
    "dependencies": ["[Step_One_Result]", "[Step_Two_Result]"],
    "instruction": "Combine the outputs of Step 1 and Step 2.",
    "description": "Atomic node scoped to [Block_One_Results]. Creates shell [Block_One_Final]: combined result fulfilling parent shell [Block_One_Results].",
    "target_outputs": ["[Block_One_Final]"]
  }
]
</EXAMPLE_TRIGGER>

<OUTPUT_FORMAT>
First output the SCRATCHPAD block in plain text.
Then output the JSON array.
No markdown. No commentary outside these two blocks.
</OUTPUT_FORMAT>
"""

# ============================================================================
# 2. THE CAPACITY OVERLOAD TEST CASE
# ============================================================================
TEST_CASE = """
INSTRUCTION:The given prompt is too long for efficient parsing and understanding. hence we need a proper split into
decompose nodes for efficient processing and understadning by evenly diving the context for each node sufficiently Derive the 
final points that map to a square with the origin point and side length. 

description: PHASE 1 — POINT DERIVATION (MANDATORY)
Task 1.1: Apply [RULE_R1] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point B.
Task 1.2: Apply [RULE_R2] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point D.
Task 1.3: Apply [RULE_R3] using the outputs generated from Task 1.1 and Task 1.2. Save the output to a point C.
Once the three new points are successfully derived, output a final status node indicating Phase 1 is complete.

After phase one is done, we will start phase two:
PHASE 2 — LINE DERIVATION (MANDATORY)
Task 2.1: Create a Horizontal Line from [Origin_Point_A] to the point B. Save output to a new Line AB.
Task 2.2: Create a Vertical Line from the point B to the point C. Save output to a new Line BC.
Task 2.3: Create a Horizontal Line from the point D to the point C. Save output to a new Line DC.
Task 2.4: Create a Vertical Line from the point D to the [Origin_Point_A]. Save output to a new Line DA.
Once all 4 lines are successfully derived, output a final status node indicating Phase 2 is complete.

After phase two is done, we will start phase three:
PHASE 3 — LINE FILLING (STRICT)
Task 3.1: Apply [RULE_R4_HORIZONTAL] to the first horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.2: Apply [RULE_R4_VERTICAL] to the first vertical line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.3: Apply [RULE_R4_HORIZONTAL] to the second horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.4: Apply [RULE_R4_VERTICAL] to the second vertical line generated in Phase 2. Save the List of Filled objects to a new node ID.
Once all 4 fill lists are successfully derived, output a final status node indicating Phase 3 is complete.

Dependencies:
- id: "[Origin_Point_A]"
  content: {"x": 0, "y": 0}
- id: "[Side_Length_L]"
  content: {"length": 10}
- id: "[RULE_R1]"
  content: Rule R1 (HorizontalExtension): store this entire rule as a data node.
this is a rule that extends a horizontal line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Point P, an Integer Length L.
OUTPUT: A new Point Q = (P.row, P.col + L).}}
- id: "[RULE_R2]"
  content: "Rule R2 (VerticalExtension): store this entire rule as a data node.
this is a rule that extends a vertical line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Point P, an Integer Length L.
OUTPUT: A new Point Q = (P.row + L, P.col).}}"
- id: "[RULE_R3]"
  content: "Rule R3 (CornerCompletion): store this entire rule as a data node.
this is a rule that completes a corner of a square by connecting two lines. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: Point B = (r1, c1), Point D = (r2, c2).
OUTPUT: A new Point C = (r2, c1).}}"
- id: "[RULE_R4_HORIZONTAL]"
  content: "Rule R4 (Strict Successor-Based Line Filling) - Horizontal: store this entire rule as a data node.
this is a rule that fills a horizontal line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Horizontal Line L with start_point(r, c_start) and end_point(r, c_end).
PROCESS:
1. Derive Filled(r, c_start)
2. If c_start < c_end: Let step = +1 Else: Let step = -1
3. Let current_c = c_start
4. While current_c ≠ c_end:
      a. current_c = current_c + step
      b. Derive Filled(r, current_c)
5. Stop ONLY after Filled(r, c_end) is derived.
OUTPUT: A JSON List of all derived Filled() objects.}}"
- id: "[RULE_R4_VERTICAL]"
  content: "Rule R4 (Strict Successor-Based Line Filling) - Vertical: store this entire rule as a data node.
this is a rule that fills a vertical line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Vertical Line L with start_point(r_start, c) and end_point(r_end, c).
PROCESS:
1. Derive Filled(r_start, c)
2. If r_start < r_end: Let step = +1 Else: Let step = -1
3. Let current_r = r_start
4. While current_r ≠ r_end:
      a. current_r = current_r + step
      b. Derive Filled(current_r, c)
5. Stop ONLY after Filled(r_end, c) is derived.
OUTPUT: A JSON List of all derived Filled() objects.}}"
"""
TEST_CASE_1 = """

    "instruction": "The given prompt is too long for efficient parsing and understanding. hence we need a proper split into decompose nodes for efficient processing and understanding by evenly diving the context for each node sufficiently Derive the final points that map to a square with the origin point and side length.",
    "description": "[INTENT]: Derive the points that map to a square with the origin point and side length. [DEFERRED LOGIC]: Apply RULE_R1 to Origin_Point_A using length Side_Length_L to save the output to a point B. Apply RULE_R2 to Origin_Point_A using length Side_Length_L to save the output to a point D. Apply RULE_R3 using the outputs generated from the previous tasks to save the output to a point C. [SHELL ARCHITECTURE]: Proactively forecasting high volume, this node generates the collection shell [Derived_Points]. Downstream task nodes must compute and aggregate their results into this shell. [PRUNED DEPENDENCIES]: Strictly inherits [Origin_Point_A] and [Side_Length_L] from the previous phase and utilizes the rule dependencies to execute. All prior unrelated scripts are pruned."
  
  Dependencies:
- id: "[Origin_Point_A]"
  content: {"x": 0, "y": 0}
- id: "[Side_Length_L]"
  content: {"length": 10}
- id: "[RULE_R1]"
  content: Rule R1 (HorizontalExtension): store this entire rule as a data node.
    this is a rule that extends a horizontal line by a given length. which will be used my llm's upon calling. the 
    entire instruction is as follows:
    {{
    INPUT: A Point P, an Integer Length L.
    OUTPUT: A new Point Q = (P.row, P.col + L).}}
- id: "[RULE_R2]"
  content: "Rule R2 (VerticalExtension): store this entire rule as a data node.
    this is a rule that extends a vertical line by a given length. which will be used my llm's upon calling. the 
    entire instruction is as follows:
    {{
    INPUT: A Point P, an Integer Length L.
    OUTPUT: A new Point Q = (P.row + L, P.col).}}"
- id: "[RULE_R3]"
  content: "Rule R3 (CornerCompletion): store this entire rule as a data node.
    this is a rule that completes a corner of a square by connecting two lines. which will be used my llm's upon calling. the 
    entire instruction is as follows:
    {{
    INPUT: Point B = (r1, c1), Point D = (r2, c2).
    OUTPUT: A new Point C = (r2, c1).}}"
"""
# ============================================================================
# EXECUTION ENGINE
# ============================================================================
def run_diagnostic():
    print(f"\n[DIAGNOSTIC] Firing request to OpenRouter ({MODEL_ID})...")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": TEST_CASE}
            ],
            temperature=0.0,  # CRITICAL: 0.0 for deterministic routing
            # OpenRouter supports HTTP headers for app rankings (optional)
            extra_headers={
                "HTTP-Referer": "https://localhost",
                "X-Title": "DAG-Diagnostic"
            }
        )
        
        result = response.choices[0].message.content
        
        print("\n" + "="*50)
        print("RAW MODEL OUTPUT:")
        print("="*50)
        print(result)
        print("="*50)
        
        # Brutal heuristic check
        result_upper = result.upper()
        # if "DECOMPOSE" in result_upper and "COMPUTE" not in result_upper:
        #     print("\n🟢 VERDICT: PASS (Capacity Aware)")
        #     print("The model successfully resisted the urge to compute massive volume.")
        # elif "COMPUTE" in result_upper:
        #     print("\n🔴 VERDICT: FAIL (Compliance Collapse)")
        #     print("The model ignored the volume constraint and attempted to compute 850k rows.")
        # else:
        #     print("\n🟡 VERDICT: UNKNOWN")
        #     print("Model output did not match expected node structures. Review manually.")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")

if __name__ == "__main__":
    run_diagnostic()