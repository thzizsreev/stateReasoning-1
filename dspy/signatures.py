"""
DSPy Signatures for the Multi-Agent Pipeline.

Design contract preserved from original architecture:
- Tunnel-vision framing: global_context and explicit_target are SEPARATE fields
- Empty-array / null are valid outputs (not errors) — annotated in field descriptions
- Named prohibition labels surface as desc= strings
- Scratchpad only on Router and Execution Worker (not middle agents)
"""

import dspy
from typing import Optional


class SemanticRouterSignature(dspy.Signature):
    """
    Stage 1 (scan): Read the full query for surface signals.
    Stage 2 (flag + slice): Set boolean flags and extract verbatim sub-strings.
    DO NOT MASK INTENTS. Snippets must be verbatim substrings or empty string "".
    """

    user_query: str = dspy.InputField(
        desc="The raw user instruction to classify. DO NOT evaluate rules against transformed versions."
    )

    fact_extraction_scratchpad: str = dspy.OutputField(
        desc=(
            "Externalized scratchpad: list every token in user_query that looks like a named "
            "data binding [Bracket_ID] or an explicit numeric/string literal. "
            "This is your evidence base before setting flags."
        )
    )
    requires_data_extraction: bool = dspy.OutputField(
        desc="True if the query contains explicit named values or literals to extract as DataNodes."
    )
    requires_decomposition: bool = dspy.OutputField(
        desc="True if the query contains multi-step workflow phases or sequential task milestones."
    )
    requires_computation: bool = dspy.OutputField(
        desc="True if the query contains atomic compute steps that apply a rule to produce an output node."
    )
    data_snippet: str = dspy.OutputField(
        desc='Verbatim substring of user_query containing only the data/fact portion. "" if requires_data_extraction is False.'
    )
    decomposition_snippet: str = dspy.OutputField(
        desc='Verbatim substring of user_query containing only the decomposition/phase portion. "" if requires_decomposition is False.'
    )
    computation_snippet: str = dspy.OutputField(
        desc='Verbatim substring of user_query containing only the atomic compute steps. "" if requires_computation is False.'
    )


class DataExtractorSignature(dspy.Signature):
    """
    Cognitive steps:
    1. Scan for [Bracket_ID]=<value> patterns.
    2. Scan for plain numeric/string literals bound to a named identifier.
    3. For each found: id=[Bracket_ID], content=<value>, status="READY".
    NO EMPTY DATA SHELLS. NO SCHEMA FITTING. Return {data_nodes: []} if nothing found.
    """

    global_context: str = dspy.InputField(
        desc="Full original query. FOR REFERENCE ONLY. Do NOT extract from this."
    )
    explicit_target: str = dspy.InputField(
        desc="The scoped data snippet. Apply your extraction rules ONLY to this text."
    )

    data_nodes: list[dict] = dspy.OutputField(
        desc=(
            "List of DataNode objects: [{id: str, content: any, status: 'READY'}]. "
            "content must be the literal parsed value, never a placeholder. "
            "Return [] if no explicit values are present. NO SCHEMA FITTING."
        )
    )


class TaskDecomposerSignature(dspy.Signature):
    """
    Cognitive steps:
    1. Identify top-level phases or milestones.
    2. For each phase: if it contains sub-tasks → DECOMPOSE. If it IS atomic → COMPUTE.
    3. Extract dependencies (inputs only, NOT the IDs the task generates).
    4. target_output: null for DECOMPOSE nodes, required for COMPUTE nodes.
    5. DO NOT extract STATE_* terminal nodes as tasks.
    NO INVENTED TYPES. NO SCHEMA FITTING. Return {task_nodes: []} on non-workflow input.
    """

    global_context: str = dspy.InputField(
        desc="Full original query. FOR REFERENCE ONLY."
    )
    explicit_target: str = dspy.InputField(
        desc="The scoped decomposition snippet. Apply ONLY to this text."
    )

    task_nodes: list[dict] = dspy.OutputField(
        desc=(
            "List of TaskNode objects: "
            "[{id: str, instruction: str, dependencies: [str], task_type: 'DECOMPOSE'|'COMPUTE', "
            "target_output: str|null, status: 'PENDING'}]. "
            "target_output is null for DECOMPOSE, required for COMPUTE. "
            "dependencies: input node IDs only — never the node this task will produce. "
            "Return [] if no workflow structure is present."
        )
    )


class ComputeAgentSignature(dspy.Signature):
    """
    Cognitive steps:
    1. Scan for atomic steps: a single rule applied to inputs → single output node.
    2. Extract the verbatim instruction text (do not paraphrase).
    3. Extract input dependency IDs only (not the target_output ID).
    4. target_output: always required — infer from context if not stated explicitly.
    5. task_type is always "COMPUTE" for every node produced.
    NO INVENTED TYPES. NO SCHEMA FITTING. Return {task_nodes: []} on multi-step or pure-fact input.
    """

    global_context: str = dspy.InputField(
        desc="Full original query. FOR REFERENCE ONLY."
    )
    explicit_target: str = dspy.InputField(
        desc="The scoped computation snippet. Apply ONLY to this text."
    )

    task_nodes: list[dict] = dspy.OutputField(
        desc=(
            "List of TaskNode objects with task_type always 'COMPUTE': "
            "[{id: str, instruction: str, dependencies: [str], task_type: 'COMPUTE', "
            "target_output: str, status: 'PENDING'}]. "
            "instruction must be verbatim from the target text. "
            "target_output is always non-null; infer if not explicitly stated. "
            "Return [] if input is multi-phase or contains only data facts."
        )
    )


class ExecutionWorkerSignature(dspy.Signature):
    """
    Cognitive steps:
    1. Parse every dependency by its [ID] key — treat values as ground truth.
    2. Identify the RULE dependency and the DATA dependencies.
    3. Apply the rule to the data. Show all intermediate steps in reasoning.
    4. Produce the exact content value for the target_output node.
    ZERO HALLUCINATION. content: null on failure — no fabrication. NO EXTRA TEXT.
    """

    dependencies_json: str = dspy.InputField(
        desc=(
            "JSON object mapping [ID] → value for all pre-fetched dependencies. "
            "Treat as ground truth. Do not invent values not present here."
        )
    )
    instruction: str = dspy.InputField(
        desc="Verbatim instruction string from the TaskNode. Execute exactly this."
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step audit trail: show rule identification, data binding, "
            "and each computation step. This is your scratchpad — be explicit."
        )
    )
    content: Optional[str] = dspy.OutputField(
        desc=(
            "The computed result to store in the target DataNode. "
            "Must be a concrete value (number, dict, string). "
            "null if computation fails or inputs are insufficient. ZERO HALLUCINATION."
        )
    )

