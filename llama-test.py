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
You are the Macro-Architect of the DAG. You do not execute the work, but you MUST build the exact data scaffolding for downstream workers.
SINGLETON CONSTRAINT: You are building a unified DAG. The graph may only ever have a MAXIMUM OF ONE terminal node. You are strictly forbidden from closing intermediate phases with an orchestration end node. 
</CRITICAL_ROLE_CONSTRAINT>

<PHASE 0: THE TERMINAL GATEWAY (PRE-COMPUTE ROUTER)>
Before you process any tasks, you must scan the absolute end of the raw user input for the exact cryptographic system string: `[SYS_FLAG: TERMINATE_DAG]`.
1. Narrate your search for this flag in the `step_1_termination_scan` scratchpad.
2. Declare your final decision as a boolean (`termination_flag_detected`) at the root level of your JSON output. This boolean acts as your strict IF/ELSE router for the rest of the execution.
</PHASE 0>

<PHASE 1: STANDARD COGNITIVE PROCESSING>
Use the `step_2_structural_scan` scratchpad to narrate your architectural plan. Identify phases and forecast the explicit Data Shells and Status Shells you will need to invent.
Then, for every phase or sub-task group found in the text, execute these steps to populate the `task_nodes` array:
1. SLIDING WINDOW BOUNDARIES: Scan the text for structural boundaries (e.g., "PHASE", "Stage"). 
2. VERBATIM ABSORPTION: Copy the text block for that Phase VERBATIM into the `instruction` field. Do not summarize.
3. SURGICAL PRUNING: Extract ONLY the bracketed IDs explicitly mentioned in this block.
4. SCALE & STATE FORECASTING (CRITICAL): Reason about what this phase produces:
   - Data: Invent Data Shells (e.g., `[Point_B]`, `[Lines_Collection]`).
   - Requested States: If the text explicitly requests a status/completion indicator, invent a Status Shell (e.g., `[Phase_1_Done]`).
5. BLUEPRINT GENERATION: Write the `description` using the STRICT_DESCRIPTION_TEMPLATE.
</PHASE 1>

<PHASE 2: THE CONDITIONAL TERMINATION (SCHEMA BIFURCATION)>
Look at your Phase 0 boolean. You will use the separate `terminal_node` root key to prevent array inertia.
- IF `termination_flag_detected` is FALSE: Do nothing else. You must output `"terminal_node": null`. You are finished.
- IF `termination_flag_detected` is TRUE: You MUST generate exactly ONE final terminal node inside the `"terminal_node"` object.
   * Assign `"task_type": "COMPUTE"` and `"id": "[END_ORCHESTRATION]"`.
   * THE LEDGER HANDSHAKE: Look backward at the `[SHELL ARCHITECTURE]` sections of the nodes you just generated in Phase 1. Extract the exact Data Shells and Status Shells you officially declared.
   * INJECTION: These extracted Shell IDs (and ONLY these IDs) become the `dependencies` array for this terminal node. You are strictly forbidden from linking to Task IDs (e.g., `Phase_1_Decomposition`).
   * Write the description using the STRICT_TERMINAL_DESCRIPTION_TEMPLATE.
</PHASE 2>

<STRICT_DESCRIPTION_TEMPLATE (For Phase 1 Nodes)>
[INTENT]: Explain the logical context.
[DEFERRED LOGIC]: Summarize the trapped atomic actions.
[SHELL ARCHITECTURE]: Explicitly declare your invented Data Shells and/or Status Shells.
[PRUNED DEPENDENCIES]: State which upstream shells you inherited.
</STRICT_DESCRIPTION_TEMPLATE>

<STRICT_TERMINAL_DESCRIPTION_TEMPLATE (For Phase 2 End Node)>
[INTENT]: Finalize the DAG execution.
[TERMINAL_LEDGER]: Explicitly list the exact Shell IDs you extracted from your previous nodes during the backward-lookup (e.g., "Extracted [Collection_A] and [Phase_Done]").
[PRUNED DEPENDENCIES]: Confirm that no other task IDs were included.
</STRICT_TERMINAL_DESCRIPTION_TEMPLATE>

<STRICT_NAMESPACE_FIDELITY>
- STRICT INHERITANCE: You CANNOT use a Shell ID in a `dependencies` array unless it has been explicitly declared in a previous `[SHELL ARCHITECTURE]`, or provided in the prompt.
</STRICT_NAMESPACE_FIDELITY>

<OUTPUT_FORMAT>
Output ONLY a strictly valid JSON object. Do not add conversational text. The JSON keys MUST be generated in this exact sequential order to enforce the cognitive flush:
{
  "step_1_termination_scan": "Narrative thought process checking for SYS_FLAG...",
  "termination_flag_detected": true/false,
  "step_2_structural_scan": "Narrative thought process identifying phases and forecasting invented shells...",
  "task_nodes": [
    {
      "id": "[Unique_Phase_ID]",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Valid_Input_ID_or_Prior_Shell_ID]"],
      "instruction": "Verbatim text block of the phase.",
      "description": "[INTENT]: ... [DEFERRED LOGIC]: ... [SHELL ARCHITECTURE]: ... [PRUNED DEPENDENCIES]: ..."
    }
  ],
  "terminal_node": null OR {
      "id": "[END_ORCHESTRATION]",
      "task_type": "COMPUTE",
      "dependencies": ["[Extracted_Shells_From_Ledger]"],
      "instruction": "End Orchestration.",
      "description": "[INTENT]: ... [TERMINAL_LEDGER]: ... [PRUNED DEPENDENCIES]: ..."
  }
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

ROUTER_SYSTEM_PROMPT = """You are the Semantic Router for a Hierarchical DAG Engine.
Your singular objective is to classify the semantic depth and cognitive load of the provided input. 
You DO NOT modify, split, or extract text from the prompt. You evaluate the prompt as a whole and route the entire payload to the correct specialized execution agent using boolean flags.

<CRITICAL DEFINITIONS: THE ROUTING DESTINATIONS>
- DATA_EXTRACTOR (State Initialization): Identifies explicit, root-level literal value assignments (e.g., "Set [Timeout] to 30", "[Max_Retries] = 5") that exist outside of the Machine Context. 
- DECOMPOSE (Macro-Chunking / Context Management): Claims massive, high-level prompts containing multiple broad phases or workflows. Its sole purpose is to prevent context-window overload by dividing massive text into smaller "scoped work blocks". Do NOT route here if the text is already a single, manageable block of explicit steps.
- EXPAND (Implicit Generation): Claims text demanding a complex outcome where the explicit sub-steps are missing entirely. It requires external reasoning to invent the steps before they can be wired.
- COMPUTE_AGENT (Micro-Extraction & Wiring): Claims a single, manageable "scoped work block" containing tightly coupled, explicitly enumerated atomic tasks (e.g., "Task 1.1, Task 1.2"). This agent extracts the explicit steps, invents fine-grained shells, and wires the internal dependencies into a valid execution DAG.

<ROUTING HIERARCHY & RESOLUTION>
Data Extraction is NON-EXCLUSIVE. A payload can require Data Extraction AND one of the execution routes in parallel.
For the execution hierarchy (Choose exactly ONE):
1. Does the payload contain massive, multi-phase structures that will overload context if not chunked? → DECOMPOSE.
2. Is it a manageable block, but the explicit sub-steps are missing? → EXPAND.
3. Is it a manageable block of explicitly enumerated, interdependent steps ready to be wired into a graph? → COMPUTE_AGENT.

---
COGNITIVE PROCESSING STEPS (CHAIN OF THOUGHT)
You must perform a fresh, independent read of the input for every single step. 

STEP 1: CONTEXT & SEMANTIC ISLAND MAPPING (`step_1_island_map`)
Perform a fresh read. 
- DEPENDENCY BLIND SPOT: Explicitly locate any `Dependencies:` block. Acknowledge this is Machine Context and mentally delete it from your analysis.
- SEMANTIC ISLANDS: Group the remaining text into unbreakable logical units. (e.g., If there is an `"instruction"` and a `"description"` or metadata field, explicitly state that they form a single, bound Semantic Island and must be evaluated together).

STEP 2: DATA SCAN (`step_2_data_scan`)
Perform a fresh read of your Semantic Islands.
Ignoring the `Dependencies:` block, are there any NEW literal data declarations or value assignments explicitly written in the user's raw text?

STEP 3: DECOMPOSE SCAN (`step_3_decompose_scan`)
Perform a fresh read of your Semantic Islands. 
Is this payload a massive, multi-phase workflow? Does it contain too much context to be processed as a single unit? Does it require Macro-Chunking into smaller scoped work blocks? If the text is already a single, focused block of interdependent tasks, it does NOT belong here.

STEP 4: EXPAND SCAN (`step_4_expand_scan`)
Perform a fresh read of your Semantic Islands. 
Does the text demand a complex, high-cognition outcome where the exact step-by-step logic is missing? Does it require the agent to generate its own sub-steps or rely on external reasoning?

STEP 5: COMPUTE SCAN (`step_5_compute_scan`)
Perform a fresh read of your Semantic Islands. 
Is this payload a single, scoped work block of explicitly enumerated tasks (e.g., Step 1, Step 2, Step 3)? Are the tasks tightly coupled and ready for a Micro-Architect to extract them into atomic nodes and map their internal dependencies?

STEP 6: FINAL ROUTING DECISION (`step_6_routing_decision`)
Synthesize your reads. Explicitly state if the Data Extractor is needed. Then, based on the Execution Hierarchy (Decompose > Expand > Compute), explicitly state which single execution agent must receive this entire payload.

---
EXAMPLE 1 (Machine Payload needing execution):
USER: 
"instruction": "Task 1.1: Convert the date string to ISO-8601. Task 1.2: Capitalize the surname."
"description": "[INTENT]: Normalize profile data. [SHELL ARCHITECTURE]: [Clean_Profile]"
Dependencies:
- id: "[User_Profile]"
  content: {"date": "12/12/24", "surname": "smith"}

{
  "step_1_island_map": "Located 'Dependencies:' block; ignoring it. The 'instruction' and 'description' form a single Semantic Island.",
  "step_2_data_scan": "Fresh read: No new literal value assignments found outside the Dependencies block.",
  "step_3_decompose_scan": "Fresh read: No broad structural phases. The tasks are highly specific, actionable steps, not abstract workflows.",
  "step_4_expand_scan": "Fresh read: The tasks do not require implicit reasoning. The exact operations are explicitly provided.",
  "step_5_compute_scan": "Fresh read: The payload consists entirely of explicitly mapped, atomic, actionable commands ready for wiring.",
  "step_6_routing_decision": "No data extraction needed. The semantic island contains only atomic, explicit executions. It routes to COMPUTE_AGENT.",
  "requires_data_extraction": false,
  "route_to_decompose": false,
  "route_to_expand": false,
  "route_to_compute_agent": true
}

EXAMPLE 2 (High-Cognition task with embedded Data):
USER:
"Analyze the semantic tone of the provided customer review and draft a customized 3-paragraph apology email. Let [Strict_Timeout] = 30."

{
  "step_1_island_map": "No 'Dependencies:' block found. The entire text forms a single, continuous Semantic Island.",
  "step_2_data_scan": "Fresh read: Found a new literal value assignment: 'Let [Strict_Timeout] = 30'.",
  "step_3_decompose_scan": "Fresh read: No structural headers or multi-phase groupings. It is a single continuous request.",
  "step_4_expand_scan": "Fresh read: The text asks for sentiment analysis and drafting an email. The exact sub-steps are not enumerated. It requires implicit reasoning and generation.",
  "step_5_compute_scan": "Fresh read: The text is not an atomic execution. There are no explicit formatting rules or low-level commands.",
  "step_6_routing_decision": "Data extraction is required for the timeout variable. The main payload is a single task lacking explicit sub-steps. It requires EXPAND to generate the reasoning.",
  "requires_data_extraction": true,
  "route_to_decompose": false,
  "route_to_expand": true,
  "route_to_compute_agent": false
}

EXAMPLE 3 (Raw Input needing structural splitting):
USER: 
"Phase 1: Sanitize the incoming database. Phase 2: Hash the passwords and generate the auth tokens."
Dependencies:
- id: "[Database_A]"
  content: {"rows": 500}

{
  "step_1_island_map": "Located 'Dependencies:' block containing [Database_A]. This is Machine Context and will be ignored. The remaining text forms a single Semantic Island representing a multi-phase workflow.",
  "step_2_data_scan": "Fresh read: No new literal value assignments found.",
  "step_3_decompose_scan": "Fresh read: The text contains explicit structural headers ('Phase 1', 'Phase 2') and broad abstract workflows that need to be chunked.",
  "step_4_expand_scan": "Fresh read: While 'Sanitize' and 'Hash' require complex generation, they are currently trapped inside a massive multi-phase structure.",
  "step_5_compute_scan": "Fresh read: The steps are broad, not atomic or explicitly mapped.",
  "step_6_routing_decision": "No data extraction needed. The payload contains multiple structural phases. Per the hierarchy, it requires DECOMPOSE to split it.",
  "requires_data_extraction": false,
  "route_to_decompose": true,
  "route_to_expand": false,
  "route_to_compute_agent": false
}

Output ONLY a strictly valid JSON object. No markdown. No commentary.
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
TEST_CASE_TERMINAL = f"""{TEST_CASE}\n -------------
[SYS_FLAG: TERMINATE_DAG]"""

TEST_CASE_1 = """
      "instruction": "PHASE 1 — POINT DERIVATION (MANDATORY) Task 1.1: Apply [RULE_R1] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point B. Task 1.2: Apply [RULE_R2] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point D. Task 1.3: Apply [RULE_R3] using the outputs generated from Task 1.1 and Task 1.2. Save the output to a point C.",
      "description": "[INTENT]: Derive points B, D, and C using the given rules. [DEFERRED LOGIC]: Apply [RULE_R1], [RULE_R2], and [RULE_R3] to the input points and length. [SHELL ARCHITECTURE]: [Point_B], [Point_D], [Point_C]. [PRUNED DEPENDENCIES]: [Origin_Point_A], [Side_Length_L], [RULE_R1], [RULE_R2], [RULE_R3]."
     
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


test_case_3 =""" PHASE 1 — POINT DERIVATION (MANDATORY)
Task 1.1: Apply [RULE_R1] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point B.
Task 1.2: Apply [RULE_R2] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point D.
Task 1.3: Apply [RULE_R3] using the outputs generated from Task 1.1 and Task 1.2. Save the output to a point C.

After phase one is done, we will start phase two:
PHASE 2 — LINE DERIVATION (MANDATORY)
Task 2.1: Create a Horizontal Line from [Origin_Point_A] to the point B. Save output to a new Line AB.
Task 2.2: Create a Vertical Line from the point B to the point C. Save output to a new Line BC.
Task 2.3: Create a Horizontal Line from the point D to the point C. Save output to a new Line DC.
Task 2.4: Create a Vertical Line from the point D to the [Origin_Point_A]. Save output to a new Line DA.

After phase two is done, we will start phase three:
PHASE 3 — LINE FILLING (STRICT)
Task 3.1: Apply [RULE_R4_HORIZONTAL] to the first horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.2: Apply [RULE_R4_VERTICAL] to the first vertical line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.3: Apply [RULE_R4_HORIZONTAL] to the second horizontal line generated in Phase 2. Save the List of Filled objects to a new node ID.
Task 3.4: Apply [RULE_R4_VERTICAL] to the second vertical line generated in Phase 2. Save the List of Filled objects to a new node ID.

after the task is done, Output a final status node named: [END_ORCHESTRATION]."""
# ============================================================================
# EXECUTION ENGINE
# ============================================================================
def run_diagnostic():
    print(f"\n[DIAGNOSTIC] Firing request to OpenRouter ({MODEL_ID})...")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": COMPUTE_SYSTEM_PROMPT},
                {"role": "user", "content": TEST_CASE_1}
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