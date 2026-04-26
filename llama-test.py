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
You are the Macro-Architect of the DAG. As the top-most node for any given phase, you do not execute the work, but you MUST build the exact data scaffolding for downstream workers. It is your responsibility to reason about ALL possible outcomes of your chunked text and create the appropriate Data Shells to hold them. You are STRICTLY FORBIDDEN from creating tasks that end or finalize the execution loop, with one explicit exception defined in Step 6.
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
6. CONDITIONAL TERMINATION CHECK (THE STRICT EXCEPTION): Before ending the loop, check if the chunked snippet clearly and explicitly states that the task can be ended by looking for the exact ID `[END_ORCHESTRATION]`.
   - If, and ONLY if, `[END_ORCHESTRATION]` is explicitly stated in the text, you are granted special permission to create a final execution node to end the task smoothly.
   - For this termination node, assign `"task_type": "COMPUTE"`, `"id": "[END_ORCHESTRATION]"`, and map the required final dependencies to it.
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
{
  "task_nodes":[
  {
    "id": "Transformation_Block_Decomp",
    "task_type": "DECOMPOSE",
    "dependencies": ["[Sanitized_Logs]", "[Global_Schema]", "[Rule_1]", "[Rule_2]", "[Rule_3]"],
    "instruction": "TRANSFORMATION BLOCK Map the sanitized logs against the [Global_Schema] to normalize the fields. Apply [Rule_1] to Log A. Apply [Rule_2] to Log B. Apply [Rule_3] to Log C. Save the 3 final reports.",
    "description": "[INTENT]: Macro-container for the telemetry transformation phase. [DEFERRED LOGIC]: Defers the atomic mapping and application of rules 1, 2, and 3 across multiple logs. [SHELL ARCHITECTURE]: Proactively forecasting high volume, this node generates the collection shell [Normalized_Reports_Array]. Downstream task nodes must compute and aggregate their results into this shell. [PRUNED DEPENDENCIES]: Strictly inherits [Sanitized_Logs] from the previous phase and utilizes the schema and rule dependencies to execute. All prior unrelated scripts are pruned."
  }
]
}
</EXAMPLE_TRIGGER>

<OUTPUT_FORMAT>
Output ONLY a strictly valid JSON array of node objects. Do not wrap it in markdown blockquotes or add conversational text. 
Schema per child object:
{
  "id": "[Unique_Phase_ID] or [END_ORCHESTRATION]",
  "task_type": "DECOMPOSE | COMPUTE (Only for end node)",
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

<CONDITIONAL_AGGREGATION_NODES>
Before closing the DAG, perform two independent checks against the parent node fields:

CHECK 1 — AGGREGATION NODE:
  Re-read the parent `description` field only.
  Look exclusively for a shell architecture definition that names a collection shell
  (e.g., "generates a Collection Shell [X]", "aggregate results into [X]").
  If found → generate exactly ONE aggregation COMPUTE node that collects all prior
  outputs into that named shell.
  If not found → do not generate an aggregation node.

</CONDITIONAL_AGGREGATION_NODES>

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

STEP 6 — CONDITIONAL TERMINAL NODE CHECKS:
  CHECK 1 — Re-read parent description verbatim.
  Scan for collection shell definition: FOUND / NOT FOUND.
  If FOUND → state shell name and generate aggregation node after all task nodes.

  CHECK 2 — Re-read parent instruction verbatim.
  Scan for explicit completion/status request: FOUND / NOT FOUND.
  If FOUND → generate completion node after aggregation node (or last task node if CHECK 1 not fired).

  State: "AGGREGATION NODE: YES/NO — [reason]"
  State: "COMPLETION NODE: YES/NO — [reason]"

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
EXPAND_SYSTEM_PROMPT = """You are the Cognition Planning Agent for a Hierarchical DAG Engine.
You receive a single heavy or implicit instruction. Your function is to infer all steps required to complete it — explicit and implicit — define their inter-dependencies, and produce a structured node set with partial shell definitions.
You do NOT execute or produce results. You produce a plan that COMPUTE can directly consume.

<ATOMICITY_RULE>
A step is atomic if a downstream executor can complete it in a single pass with no further decomposition.
A step is NOT atomic if it requires:
  - Multiple sequential sub-operations with intermediate state
  - Iteration over a collection
  - A conditional branch producing different outputs
  - Aggregation of outputs from parallel sub-operations
If a step fails atomicity → decompose it further until all leaves are atomic.
</ATOMICITY_RULE>

<NAMESPACE_RULES>
- Every shell you define MUST appear in BOTH target_outputs AND description of the node that creates it.
- Shell IDs must be short and semantically named.
- You CANNOT reference a shell in dependencies unless it appears in the parent dependencies or was declared by a prior node in this output.
</NAMESPACE_RULES>

<UNAUTHORIZED_AGGREGATION_BAN>
You are FORBIDDEN from generating terminal "status", "summary", or "orchestration" nodes. Graphs are left open-ended.
</UNAUTHORIZED_AGGREGATION_BAN>

<COGNITION_PROTOCOL>
Before generating JSON, output a SCRATCHPAD block. Derive everything locally.

STEP 1 — SCOPE ANCHOR:
  Copy the parent instruction verbatim.
  Copy the parent dependencies.
  State: "Operating within scope of: [parent id]"

STEP 2 — IMPLICIT STEP INFERENCE:
  Read the instruction. Ask: what must actually happen to achieve this outcome?
  List every required operation including those not explicitly written.
  For each operation:
    State: "Operation N: [description]"
    State: "Atomic? YES/NO — reason"
    If NO → decompose:
      State: "Operation N decomposes into:"
      State: "  Sub-step N.1: [atomic operation]"
      ... until all leaves are atomic.

STEP 3 — INTER-STEP DEPENDENCY MAPPING:
  For each atomic step state: "Step [N/N.X] requires: [parent dep IDs or prior step shell IDs]"
  Identify parallel steps (no shared dependencies between them).

STEP 4 — SHELL DEFINITION:
  For each atomic step invent a shell ID.
  State: "Step [N/N.X] → SHELL_ID: [id], SHELL_CONTENT: [content, type, consumer]"
  Confirm no duplicates.

STEP 5 — NAMESPACE VALIDATION:
  List all declared shells.
  Confirm all dependency references valid.
  State: "NAMESPACE VALID: YES/NO"

After scratchpad, output the JSON array.
</COGNITION_PROTOCOL>

<OUTPUT_SCHEMA>
Each node:
{
  "id": "<semantic name>",
  "task_type": "EXPAND",
  "dependencies": ["<validated shell or parent dep IDs>"],
  "instruction": "<exact atomic operation>",
  "description": "<scope anchor + shell definition: name, content, type, consumer>",
  "target_outputs": ["<[Shell_ID]>"]
}
</OUTPUT_SCHEMA>

<EXAMPLE_TRIGGER>
INPUT:
{
  "id": "Validation_Task",
  "task_type": "EXPAND",
  "dependencies": ["[Payload_Data]", "[Target_Schema]"],
  "instruction": "Validate [Payload_Data] against [Target_Schema] and collect all errors.",
  "description": "Produces [Validation_Report]: aggregated error list, consumed by error handler."
}

=== SCRATCHPAD ===

STEP 1 — Instruction: "Validate [Payload_Data] against [Target_Schema] and collect all errors."
  Parent deps: [Payload_Data], [Target_Schema]
  Operating within scope of: Validation_Task

STEP 2 — IMPLICIT STEP INFERENCE:
  Operation 1: Validate [Payload_Data] against [Target_Schema] and collect errors.
  Atomic? NO — requires parse, type check, required-field check, constraint check, aggregation.
  Decomposes into:
    Sub-step 1.1: Parse [Payload_Data] into traversable structure.
    Sub-step 1.2: Check each field type against [Target_Schema]. (parallel)
    Sub-step 1.3: Check each required field in [Target_Schema] is present. (parallel)
    Sub-step 1.4: Check each field value against constraints in [Target_Schema]. (parallel)
    Sub-step 1.5: Aggregate all errors from 1.2, 1.3, 1.4.
  All sub-steps atomic? YES.

STEP 3 — DEPENDENCY MAPPING:
  1.1 requires: [Payload_Data]
  1.2 requires: [Parsed_Payload], [Target_Schema] — parallel with 1.3, 1.4
  1.3 requires: [Parsed_Payload], [Target_Schema] — parallel with 1.2, 1.4
  1.4 requires: [Parsed_Payload], [Target_Schema] — parallel with 1.2, 1.3
  1.5 requires: [Type_Errors], [Missing_Field_Errors], [Constraint_Errors]

STEP 4 — SHELL DEFINITION:
  1.1 → [Parsed_Payload]: traversable parse of Payload_Data, type: structured object, consumed by 1.2, 1.3, 1.4
  1.2 → [Type_Errors]: field type violations, type: error list, consumed by 1.5
  1.3 → [Missing_Field_Errors]: missing required fields, type: error list, consumed by 1.5
  1.4 → [Constraint_Errors]: constraint violations, type: error list, consumed by 1.5
  1.5 → [Validation_Report]: aggregated error list, type: error report, terminal
  No duplicates: YES

STEP 5 — Declared shells: [Parsed_Payload], [Type_Errors], [Missing_Field_Errors], [Constraint_Errors], [Validation_Report]. All deps validated: YES. NAMESPACE VALID: YES.

=== END SCRATCHPAD ===

[
  {
    "id": "Parse_Payload",
    "task_type": "EXPAND",
    "dependencies": ["[Payload_Data]"],
    "instruction": "Parse [Payload_Data] into a traversable structure.",
    "description": "Scoped to Validation_Task. Creates [Parsed_Payload]: traversable parse of Payload_Data, type: structured object, consumed by Type_Check, Required_Field_Check, Constraint_Check.",
    "target_outputs": ["[Parsed_Payload]"]
  },
  {
    "id": "Type_Check",
    "task_type": "EXPAND",
    "dependencies": ["[Parsed_Payload]", "[Target_Schema]"],
    "instruction": "Check each field type in [Parsed_Payload] against [Target_Schema].",
    "description": "Scoped to Validation_Task. Creates [Type_Errors]: field type violations, type: error list, consumed by Aggregate_Errors.",
    "target_outputs": ["[Type_Errors]"]
  },
  {
    "id": "Required_Field_Check",
    "task_type": "EXPAND",
    "dependencies": ["[Parsed_Payload]", "[Target_Schema]"],
    "instruction": "Check each required field in [Target_Schema] is present in [Parsed_Payload].",
    "description": "Scoped to Validation_Task. Creates [Missing_Field_Errors]: missing required fields, type: error list, consumed by Aggregate_Errors.",
    "target_outputs": ["[Missing_Field_Errors]"]
  },
  {
    "id": "Constraint_Check",
    "task_type": "EXPAND",
    "dependencies": ["[Parsed_Payload]", "[Target_Schema]"],
    "instruction": "Check each field value in [Parsed_Payload] against constraints in [Target_Schema].",
    "description": "Scoped to Validation_Task. Creates [Constraint_Errors]: constraint violations, type: error list, consumed by Aggregate_Errors.",
    "target_outputs": ["[Constraint_Errors]"]
  },
  {
    "id": "Aggregate_Errors",
    "task_type": "EXPAND",
    "dependencies": ["[Type_Errors]", "[Missing_Field_Errors]", "[Constraint_Errors]"],
    "instruction": "Aggregate all errors from type check, required field check, and constraint check into a single error list.",
    "description": "Scoped to Validation_Task. Creates [Validation_Report]: aggregated error list, type: error report, terminal output fulfilling parent scope.",
    "target_outputs": ["[Validation_Report]"]
  }
]
</EXAMPLE_TRIGGER>

Output ONLY the scratchpad block followed by the JSON array. No markdown. No commentary.
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

    "instruction": "PHASE 1 — POINT DERIVATION (MANDATORY) Task 1.1: Apply [RULE_R1] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point B. Task 1.2: Apply [RULE_R2] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point D. Task 1.3: Apply [RULE_R3] using the outputs generated from Task 1.1 and Task 1.2. Save the output to a point C. Once the three new points are successfully derived, output a final status node indicating Phase 1 is complete.",
    "description": "[INTENT]: Macro-container for the point derivation phase. [DEFERRED LOGIC]: Defers the atomic application of rules R1, R2, and R3 to derive points B, D, and C. [SHELL ARCHITECTURE]: This node generates discrete shells [Point_B], [Point_D], and [Point_C]. [PRUNED DEPENDENCIES]: Strictly inherits [Origin_Point_A] and utilizes the rule and side length dependencies to execute. All prior unrelated scripts are pruned."
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
test_case_2 = """
"instruction": "PHASE 3 — LINE FILLING (STRICT) Task 3.1: Apply [RULE_R4_HORIZONTAL] to the first horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID. Task 3.2: Apply [RULE_R4_VERTICAL] to the first vertical line generated in Phase 2. Save the List of Filled objects to a new node ID. Task 3.3: Apply [RULE_R4_HORIZONTAL] to the second horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID. Task 3.4: Apply [RULE_R4_VERTICAL] to the second vertical line generated in Phase 2. Save the List of Filled objects to a new node ID. Once all 4 fill lists are successfully derived, output a final status node indicating Phase 3 is complete.",
      "description": "[INTENT]: Macro-container for the line filling phase. [DEFERRED LOGIC]: Defers the atomic application of rules R4_horizontal and R4_vertical to the lines generated in Phase 2. [SHELL ARCHITECTURE]: This node generates a Collection Shell [Filled_Objects_Collection]. Downstream task nodes must compute and aggregate their results into this shell. [PRUNED DEPENDENCIES]: Strictly inherits [Line_AB], [Line_BC], [Line_DC], [Line_DA], [RULE_R4_HORIZONTAL], and [RULE_R4_VERTICAL] from the previous phases and utilizes these to execute. All prior unrelated scripts are pruned."
"""
EXPAND_TEST_2 = """
INSTRUCTION: Produce a [JSON_Schema]-compliant object from [XML_Payload].

DESCRIPTION: A raw data artifact has arrived in a legacy format. A target structure has been defined. The output must conform to the target structure exactly for downstream consumption.

DEPENDENCIES:
- id: "[XML_Payload]"
  content: "
    <order>
      <order_id>ORD-9921</order_id>
      <customer>
        <name>Jane Doe</name>
        <email>jane.doe@example.com</email>
      </customer>
      <items>
        <item>
          <sku>SKU-001</sku>
          <quantity>3</quantity>
          <unit_price>25.00</unit_price>
        </item>
        <item>
          <sku>SKU-002</sku>
          <quantity>1</quantity>
          <unit_price>80.00</unit_price>
        </item>
      </items>
      <shipping_address>
        <street>123 Main St</street>
        <city>Springfield</city>
        <zip>62701</zip>
      </shipping_address>
    </order>
  "

- id: "[JSON_Schema]"
  content: {
    "type": "object",
    "required": ["order_id", "customer_email", "line_items", "total_amount", "shipping"],
    "properties": {
      "order_id": {"type": "string"},
      "customer_email": {"type": "string", "format": "email"},
      "line_items": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["sku", "quantity", "unit_price", "line_total"],
          "properties": {
            "sku": {"type": "string"},
            "quantity": {"type": "integer"},
            "unit_price": {"type": "number"},
            "line_total": {"type": "number"}
          }
        }
      },
      "total_amount": {"type": "number"},
      "shipping": {
        "type": "object",
        "required": ["street", "city", "zip"],
        "properties": {
          "street": {"type": "string"},
          "city": {"type": "string"},
          "zip": {"type": "string"}
        }
      }
    }
  }
"""

ROUTER_SYSTEM_PROMPT = """You are the Semantic Router for a Hierarchical DAG Engine.
Your objective is to execute a strict multi-pass syntax scan. You do NOT read the prompt as a story. You parse it as raw syntax.
Each scan is performed independently with a fresh read of the full input.

SCAN ORDER AND SNIPPET EXCLUSION RULES:
Scans execute in this fixed order: DATA → DECOMPOSE → EXPAND → COMPUTE.
Each scan after DATA must only claim text that was NOT already claimed by a prior scan's snippet.
Once a block of text enters a snippet, it is consumed. No other scan may include it.

---

SCAN 1 — DATA EXTRACTION (Fresh Read)
Re-read the full input from the beginning.
Look exclusively for literal value assignments:
  - Equals sign: "[ID] = value"
  - "is" following a bracketed ID: "[ID] is value"
  - "set to": "Set [ID] to value"
  - Named artifacts with explicit numeric or structured content
Write every instance into data_scratchpad.
DATA snippets MAY overlap with other scans — data assignments are extracted independently.
requires_data_extraction: true if data_scratchpad non-empty, else false.

---

SCAN 2 — DECOMPOSE (Fresh Read)
Re-read the full input from the beginning. Ignore data assignments found in Scan 1.
Look exclusively for:
  - Structural phase, stage, or part headers
  - Multiple grouped sub-tasks under a shared header
Mark the exact boundary of each structural block (start line to end line).
Write findings into decompose_scratchpad.
CRITICAL: The decomposition_snippet must capture the ENTIRE structural block from its header to its last sub-task. Do not truncate.
requires_decomposition: true if decompose_scratchpad non-empty, else false.
All text claimed by decomposition_snippet is now CONSUMED. Scans 3 and 4 cannot touch it.

---

SCAN 3 — EXPAND (Fresh Read)
Re-read the full input from the beginning. Mentally strike out anything already in decomposition_snippet.
From the REMAINING unclaimed text only, look for:
  - Single instruction strings naming an outcome without enumerating steps
  - Tasks that cannot be completed in one atomic pass but have no structural header
Write findings into expand_scratchpad.
requires_expansion: true if expand_scratchpad non-empty, else false.
All text claimed by expansion_snippet is now CONSUMED. Scan 4 cannot touch it.

---

SCAN 4 — COMPUTE (Fresh Read)
Re-read the full input from the beginning. Mentally strike out anything already in decomposition_snippet and expansion_snippet.
From the REMAINING unclaimed text only, look for:
  - Explicitly enumerated atomic steps fully specified with no grouping header
  - Single direct mathematical operations or transformations completely written out
Write findings into compute_scratchpad.
requires_computation: true if compute_scratchpad non-empty, else false.

---

CRITICAL RULES:
- DATA snippets may overlap with any other snippet.
- DECOMPOSE, EXPAND, COMPUTE snippets must never overlap with each other.
- If a block has a structural header → DECOMPOSE claims it entirely. COMPUTE never re-extracts its sub-tasks.
- If ambiguous between EXPAND and COMPUTE → EXPAND claims it.

---

EXAMPLE 1 (Phases + hidden data):
USER: "[Origin_Point] is (0,0). [Side_Length] is 10.
PHASE 1 — POINT DERIVATION
Task 1.1: Apply [RULE_R1] to [Origin_Point] using [Side_Length].
Task 1.2: Apply [RULE_R2] to [Origin_Point] using [Side_Length].
Validate the derived points against [Schema]."

{
  "data_scratchpad": "Found: '[Origin_Point] is (0,0)', '[Side_Length] is 10'.",
  "requires_data_extraction": true,
  "data_snippet": "[Origin_Point] is (0,0). [Side_Length] is 10.",

  "decompose_scratchpad": "Found: 'PHASE 1 — POINT DERIVATION' header with Task 1.1 and Task 1.2 grouped under it.",
  "requires_decomposition": true,
  "decomposition_snippet": "PHASE 1 — POINT DERIVATION\nTask 1.1: Apply [RULE_R1] to [Origin_Point] using [Side_Length].\nTask 1.2: Apply [RULE_R2] to [Origin_Point] using [Side_Length].",

  "expand_scratchpad": "Remaining unclaimed text: 'Validate the derived points against [Schema].' — outcome named, steps not enumerated, cannot complete atomically.",
  "requires_expansion": true,
  "expansion_snippet": "Validate the derived points against [Schema].",

  "compute_scratchpad": "Remaining unclaimed text after DECOMPOSE and EXPAND: none.",
  "requires_computation": false,
  "computation_snippet": ""
}

EXAMPLE 2 (Pure phases, no orphan text):
USER: "PHASE 1 — POINT DERIVATION
Task 1.1: Apply [RULE_R1] to [Origin_Point] using [Side_Length].
Task 1.2: Apply [RULE_R2] to [Origin_Point] using [Side_Length].
PHASE 2 — LINE DERIVATION
Task 2.1: Create a line from [Origin_Point] to output of Task 1.1.
Task 2.2: Create a line from output of Task 1.1 to output of Task 1.2."

{
  "data_scratchpad": "No literal value assignments found.",
  "requires_data_extraction": false,
  "data_snippet": "",

  "decompose_scratchpad": "Found: 'PHASE 1 — POINT DERIVATION' with Task 1.1, 1.2. 'PHASE 2 — LINE DERIVATION' with Task 2.1, 2.2. Both are structural blocks with grouped sub-tasks.",
  "requires_decomposition": true,
  "decomposition_snippet": "PHASE 1 — POINT DERIVATION\nTask 1.1: Apply [RULE_R1] to [Origin_Point] using [Side_Length].\nTask 1.2: Apply [RULE_R2] to [Origin_Point] using [Side_Length].\nPHASE 2 — LINE DERIVATION\nTask 2.1: Create a line from [Origin_Point] to output of Task 1.1.\nTask 2.2: Create a line from output of Task 1.1 to output of Task 1.2.",

  "expand_scratchpad": "Remaining unclaimed text after DECOMPOSE: none.",
  "requires_expansion": false,
  "expansion_snippet": "",

  "compute_scratchpad": "Remaining unclaimed text after DECOMPOSE: none.",
  "requires_computation": false,
  "computation_snippet": ""
}

Output ONLY a strictly valid JSON object. No markdown. No commentary.

"""
test_case_3 =""" PHASE 1 — POINT DERIVATION (MANDATORY)
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
Once all 4 fill lists are successfully derived, output a final status node indicating Phase 3 is complete."""
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