import dspy
from typing import List, Literal, Optional, Any
from schema import DataNode, TaskNode, RouterDecision
from pydantic import BaseModel, Field
#from schema import RouterDecision
# 1. Define the exact JSON shape using Pydantic


# 2. The Signature
class RouterDecision(BaseModel):
    # ---------------------------------------------------------
    # 1. THE WORK (Snippets are generated FIRST)
    # ---------------------------------------------------------
    decomposition_snippet: str = Field(
        description="Verbatim extraction of the ENTIRE macro-workflow. Must include all phases and steps without truncation. If not a workflow, output ''."
    )
    computation_snippet: str = Field(
        description=(
            "Verbatim extraction of an ATOMIC compute task. "
            "CRITICAL PRIORITY RULE: If you extracted text into decomposition_snippet, this field MUST be an empty string ''. "
            "Do not extract compute tasks from macro-workflows."
        )
    )
    data_snippet: str = Field(
        description="Verbatim extraction of static facts or node initializations. Empty string '' if there are no explicit static facts."
    )

    # ---------------------------------------------------------
    # 2. THE CLASSIFICATION (Booleans are generated LAST)
    # ---------------------------------------------------------
    requires_decomposition: bool = Field(
        description="MUST be True if decomposition_snippet contains text. MUST be False if it is empty."
    )
    requires_computation: bool = Field(
        description="MUST be True if computation_snippet contains text. MUST be False if decomposition_snippet contains text."
    )
    requires_data_extraction: bool = Field(
        description="MUST be True if data_snippet contains text. MUST be False if empty."
    )

class SemanticRouterSignature(dspy.Signature):
    """
    You are the Metacognitive Router for an Autonomous HTN DAG Engine.
    
    Your sole purpose is to parse complex, multi-phase user workflows and route the exact text 
    to three cognitive buckets: Data Extraction, Decomposition, and Computation.
    
    ENVIRONMENTAL CONTEXT (ONTOLOGY):
    - DAG Engine: A system that executes mathematical and logical tasks in strict chronological order.
    - Nodes (e.g., [Data_A], [STATE_DONE]): Discrete memory units.
    - Rules (e.g., [RULE_R1]): Functions applied to Nodes.
    - Decomposition: The act of defining phases and task sequences.
    
    THE ANTI-LAZINESS PROTOCOL (MANDATORY):
    1. NO TRUNCATION: If a workflow contains 3 phases, your extracted snippets MUST contain the text for all 3 phases.
    2. VERBATIM COPYING: You must copy/paste directly from the input. Do not paraphrase. Do not add commas or brackets that are not in the original text.
    3. CONTINUOUS BLOCKS: A snippet must be a continuous, unbroken sequence of characters directly from the source text.
    """
    user_query: str = dspy.InputField(desc="The massive, multi-phase workflow text provided by the environment.")
    decision: RouterDecision = dspy.OutputField()



"-------------------------------Decomposer Agent Signature-------------------------------"

ORCHESTRATION_END_ID = "[ORCHESTRATION_END]"
 
 
# ============================================================================
# OUTPUT SCHEMA
# ============================================================================
 
class DecomposerTaskNode(BaseModel):
    id: str = Field(
        description=(
            "Unique bracketed identifier derived from what this task does. "
            "Format: [Task_<verb>_<noun>]. Example: [Task_Analyze_Contract]. "
            "Exception: the final terminal task must always use the reserved ID [Task_Finalize_Orchestration]."
        )
    )
    instruction: str = Field(
        description=(
            "A self-contained instruction with enough context to execute independently. "
            "Reference dependency outputs by their [BracketedID] where needed. "
            
        )
    )
    dependencies: List[str] = Field(
        description=(
            "Bracketed IDs this task must wait for before executing. "
            "Include only what this task genuinely cannot start without. "
            "it must not fire until all results are ready."
            "Format: [BracketedID]. Empty list if no prerequisites."
        )
    )
    task_type: Literal["COMPUTE", "DECOMPOSE", "EXPAND", "OUTPUT"] = Field(
        description=(
            "COMPUTE: single atomic action with a deterministic output. "
            "DECOMPOSE: too broad — needs further breakdown. "
            "EXPAND: underspecified — missing context that must be elicited. "
            "OUTPUT: the terminal task that writes [ORCHESTRATION_END] and concludes execution."
        )
    )
    target_output: Optional[str] = Field(
        default=None,
        description=(
            "The [BracketedID] this task produces. "
            "Declare for COMPUTE only. Leave null for DECOMPOSE and EXPAND. "
            "For OUTPUT task: always declare [ORCHESTRATION_END] here."
        )
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Required only for the OUTPUT terminal task. "
            "Must explicitly list the [BracketedID]s of every node containing user-facing results. "
            "Example: 'Concludes orchestration. User results are in [Data_FinalReport] and [Data_Summary].'"
            "For all other tasks: omit."
        )
    )
 
 
class DecomposerDecision(BaseModel):
    chronological_scratchpad: str = Field(
        description=(
            "Plan before producing tasks: "
            "What is the ultimate goal? "
            "What distinct pieces of work are needed? "
            "What depends on what? "
            "Which pieces are atomic, which need breakdown, which are missing context? "
            "What are the final user-facing output nodes that must exist before orchestration ends?"
        )
    )
    extracted_tasks: List[DecomposerTaskNode] = Field(
        description=(
            "The execution plan as ordered TaskNodes. "
            "Tasks must appear after their dependencies. "
            "The final task must always be type OUTPUT, target_output [ORCHESTRATION_END], "
            "with all user-facing result nodes declared in its dependencies and description. "
            "Return [] only if the input is already a single atomic action."
        )
    )
 
 
# ============================================================================
# DSPY SIGNATURE
# ============================================================================
 
class DecomposerSignature(dspy.Signature):
    """
    You are the Decomposer Agent — a Planner for a DAG execution engine.
 
    YOUR JOB: Break any request into an executable plan of interdependent tasks.
 
    BEHAVIORAL GUIDELINES:
    - Derive task structure from intent. Explicit steps are not required in the input.
    - Each task must carry one responsibility. Split when in doubt.
    - Model dependencies honestly — if Task B needs Task A's result, declare it.
    - Do not execute. You produce instructions, never answers or computed values.
    - When a piece of work is too vague to plan, mark it EXPAND rather than guessing.
    - Any domain is valid. Plan with the same rigor for law, geometry, medicine, or code.
 
    TERMINAL TASK (MANDATORY):
    For the terminal task: instruction must state which output nodes contain the user's results and declare that it will write [ORCHESTRATION_END] to signal completion.
    Every plan must end with exactly one OUTPUT task that:
    - Has target_output: [ORCHESTRATION_END]
    - Lists every user-facing result node in its dependencies
    - Describes which nodes contain the results the user asked for
    - The terminal task must list every final user-facing output node as a dependency but not as a dependency itself.   
    This task signals the orchestration engine to halt. Without it, execution runs forever.
    
    """
 
    explicit_target: str = dspy.InputField(
        desc="The request to plan. Any domain, any structure — derive the task plan from its intent."
    )
 
    decision: DecomposerDecision = dspy.OutputField()

# 3. The Module with Assertions
"-------------------------------Compute Agent Signature-------------------------------"
class ComputeOutputNode(BaseModel):
    id: str = Field(
        description=(
            "The [BracketedID] of the data node being produced. "
            "Must match target_output_id exactly."
        )
    )
    content: Any = Field(
        default=None,
        description=(
            "The computed result. Any type: number, string, list, dict, expression. "
            "Null if execution cannot complete — never fabricate a value."
        )
    )
    description: str = Field(
        description=(
            "One sentence describing what this node holds and how it was derived."
        )
    )
 
 
class ComputeDecision(BaseModel):
    reasoning: str = Field(
        description=(
            "Full execution trace: identify the operation, bind each dependency to its value, "
            "show every intermediate step, state the result. "
            "Be explicit — this is the audit trail."
        )
    )
    output_node: Optional[ComputeOutputNode] = Field(
        default=None,
        description=(
            "The produced DataNode. "
            "Null if a required dependency is missing or execution fails — halt cleanly."
        )
    )
 
 
# ============================================================================
# DSPY SIGNATURE
# ============================================================================
 
class ComputeAgentSignature(dspy.Signature):
    """
    You are the Compute Agent — the Atomic Executor for a DAG execution engine.
 
    YOUR JOB: Execute one instruction using the provided dependency values and produce one DataNode.
 
    BEHAVIORAL GUIDELINES:
    - Treat injected dependencies as ground truth. Do not assume values not present.
    - If a required value is missing, return null — do not approximate or skip.
    - Show your work in reasoning before producing output. The trace matters.
    - One instruction produces exactly one output node.
    - Any domain is valid: compute math, format legal text, classify medical data, anything.
    """
 
    dependencies_json: str = dspy.InputField(
        desc=(
            "JSON object mapping [BracketedID] → value for all pre-fetched dependencies. "
            "Ground truth — treat every value as exact."
        )
    )
    instruction: str = dspy.InputField(
        desc="The single atomic instruction to execute."
    )
    target_output_id: str = dspy.InputField(
        desc="The [BracketedID] your output_node.id must match exactly."
    )
 
    decision: ComputeDecision = dspy.OutputField()

"-------------------------------Data Extractor Agent Signature-------------------------------"
class ExtractedDataNode(BaseModel):
    id: str = Field(
        description=(
            "A [BracketedID] label for this value. "
            "Use the source text's name if one exists. "
            "Otherwise derive a concise, descriptive label from the value's role. "
            "Reflect the logical concept, not the primitive type. "
            "Example: [Data_StartPoint] not [Data_XCoordinate] and [Data_YCoordinate]."
        )
    )
    content: Any = Field(
        description=(
            "The value as it exists in the source text. "
            "Preserve original form — do not normalize, convert, or simplify. "
            "Grouped values (coordinates, dimensions, ranges, pairs) must be extracted "
            "as a single compound value (list, dict, or tuple) under one id — not atomized. "
            "Grouping preference: semantic/logical first, syntactic second "
            "(parentheses, slashes, brackets are strong grouping signals in mathematical contexts)."
        )
    )
    description: str = Field(
        description=(
            "One sentence describing what this value represents in the context of the request."
        )
    )
 
 
class DataExtractorDecision(BaseModel):
    extracted_nodes: List[ExtractedDataNode] = Field(
        description=(
            "Data nodes extracted from the source text. "
            "Return [] if no explicit values are present — do not force extraction."
        )
    )
 
 
# ============================================================================
# DSPY SIGNATURE
# ============================================================================
 
class DataExtractorSignature(dspy.Signature):
    """
    You are the Data Extractor — a value harvester for a DAG execution engine.
 
    YOUR JOB: Identify and extract explicitly stated values from the input text as DataNodes.
 
    BEHAVIORAL GUIDELINES:
    - Extract what is given, not what can be derived. If a value requires reasoning to exist, leave it.
    - Be selective, not exhaustive. Serve the task — do not harvest every token that looks like a number.
    - When values are logically or semantically linked, treat them as one entity under one ID.
    - In mathematical or geometric contexts, let syntax guide grouping (parentheses, paired notation).
    - If nothing is explicitly stated, return empty. Restraint is correct behavior here.
    - Any domain is valid: mathematics, law, medicine, physics, literature, geometry, and beyond.
    """
 
    source_text: str = dspy.InputField(
        desc="The text to extract values from. Any domain, any structure."
    )
 
    decision: DataExtractorDecision = dspy.OutputField()
"------------------------------------ expander agent signature ------------------------------------"
class ExpanderTaskNode(BaseModel):
    id: str = Field(
        ...,
        description="The bracketed ID of the task. Format: [Task_<verb>_<noun>]."
    )
    task_type: Literal["FETCH", "EXPAND", "COMPUTE", "DECOMPOSE", "OUTPUT"] = Field(
        ...,
        description=(
            "The strict operational type. "
            "FETCH: retrieve data not present in graph from external source. No dependencies. "
            "COMPUTE: apply logic to already-present values. Single atomic operation. "
            "DECOMPOSE: structural breakdown of high-volume or multi-action work. "
            "EXPAND: delegate abstract or cognitively heavy goal to child sub-graph. "
            "OUTPUT: terminal signal. Use with target_output=[ORCHESTRATION_END] and empty dependencies to halt."
        )
    )
    reasoning: str = Field(
        ...,
        description="First-person justification for this specific node's type and structure."
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description=(
            "List of inherited bracketed variables this task waits for. "
            "Only include IDs that will exist when this task fires. "
            "Phantom IDs cause permanent deadlock."
        )
    )
    instruction: str = Field(
        ...,
        description=(
            "The precise action to execute. "
            "Contract: Action + Input [BracketedIDs] + 'store as [Output_ID]'."
        )
    )
    target_output: Optional[str] = Field(
        default=None,
        description=(
            "The pre-declared bracketed variable this node outputs. "
            "[ORCHESTRATION_END] if this node halts execution. "
            "DECOMPOSE: leave null — output resolved downstream."
        )
    )
    description: str = Field(
        ...,
        description=(
            "Objective summary of what this node does and what it produces. "
            "Final inheritor must declare the container output ID here."
        )
    )
 
 
# ============================================================================
# EXPANDER DECISION SCHEMA
# ============================================================================
 
class ExpanderDecision(BaseModel):
    extracted_tasks: List[ExpanderTaskNode] = Field(
        default_factory=list,
        description="The flat topological array of tasks."
    )
 
 
# ============================================================================
# DSPY SIGNATURE
# ============================================================================
 
class ExpanderSignature(dspy.Signature):
    """
    You are the Expander Agent — a Sub-Graph Architect for a DAG execution engine.
 
    YOUR JOB: Take an abstract or cognitively heavy goal and unroll it into
    a flat, ordered list of interdependent task nodes that collectively fulfill it.
 
    BEHAVIORAL GUIDELINES:
    - Node IDs referenced in the instruction already exist in the graph. Do not fetch them.
    - Anything not referenced must be produced by a FETCH or COMPUTE before anything depends on it.
    - Prefer FETCH for external data, COMPUTE for logic on present values.
    - Build the minimum viable path. Do not over-architect.
    - If a step is too abstract or cognitively heavy, spawn EXPAND and delegate.
    - To halt: create one OUTPUT node with empty dependencies and target_output=[ORCHESTRATION_END].
    - Every node must have a description. Final inheritor must declare its container output ID there.
 
    OUTPUT FORMAT (CRITICAL):
    Return a JSON object with exactly this shape. Do NOT return a JSON Schema.
    {
      "extracted_tasks": [
        {
          "id": "[Task_verb_noun]",
          "task_type": "FETCH | COMPUTE | DECOMPOSE | EXPAND | OUTPUT",
          "reasoning": "First-person justification for this node.",
          "dependencies": ["[BracketedID]"],
          "instruction": "Action + inputs + store as [Output_ID].",
          "target_output": "[Output_ID]",
          "description": "Objective summary of what this node produces."
        }
      ]
    }
    """
 
    explicit_target: str = dspy.InputField(
        desc=(
            "The task instruction to fulfill. "
            "Provided as JSON with instruction, description, and dependencies fields. "
            "Node IDs in dependencies already exist in the graph."
        )
    )
 
    decision: ExpanderDecision = dspy.OutputField()