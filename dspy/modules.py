import dspy

from signatures import SemanticRouterSignature
#from schema import RouterDecision
from signatures import RouterDecision, DecomposerSignature, DecomposerDecision
from signatures import ComputeAgentSignature, ComputeDecision, DataExtractorDecision, DataExtractorSignature
from signatures import ORCHESTRATION_END_ID, ExpanderDecision, ExpanderSignature, ExpanderTaskNode
from schema import DataNode
import json
from typing import Optional
from collections import defaultdict, deque

def verify_verbatim_snippets(inputs, outputs, trace=None):
    # FIX: 'inputs' is passed as a raw dictionary, so use bracket notation
    query = inputs["user_query"]
    
    # 'outputs' is the DSPy prediction object, so dot-notation still works here
    decision = outputs.decision
    
    # If the LLM summarized instead of copying exactly, we return False (Failed).
    if decision.data_snippet and decision.data_snippet not in query:
        return False
    if decision.decomposition_snippet and decision.decomposition_snippet not in query:
        return False
    if decision.computation_snippet and decision.computation_snippet not in query:
        return False
        
    return True

# 4. The Modern Module
class SemanticRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # We wrap the predictor in dspy.Refine!
        # If verify_verbatim_snippets returns False, Refine automatically 
        # loops back, scolds the LLM, and makes it try again (up to 3 times).
        # The corrected initialization
        base_predictor = dspy.ChainOfThought(SemanticRouterSignature)
        self.route = dspy.Refine(
            module=base_predictor, 
            N=3, # Maximum number of self-correction attempts
            reward_fn=verify_verbatim_snippets,
            threshold=1.0 # <--- ADD THIS LINE: Stop looping when score reaches 1.0 (True)
        )

    def forward(self, user_query: str) -> RouterDecision:
        # We must pass the input as an 'example' so the reward_fn can read the original query
        result = self.route(user_query=user_query)
        return result.decision
"- ------------------------------Decomposer Agent Module-------------------------------"

def _validate_decision(decision: DecomposerDecision) -> DecomposerDecision:
    """
    Enforce structural contracts post-LLM.
    - COMPUTE nodes must have target_output (warn if missing, don't drop)
    - DECOMPOSE / EXPAND / OUTPUT nodes must have target_output=None
    - A node's dependencies must never include its own target_output
    """
    cleaned = []
    for node in decision.extracted_tasks:
        updates = {}
 
        # Enforce null contract for non-COMPUTE nodes
        if node.task_type in ("DECOMPOSE", "EXPAND", "OUTPUT") and node.target_output is not None:
            updates["target_output"] = None
 
        # Warn if COMPUTE node is missing target_output (don't drop — let compiler learn)
        if node.task_type == "COMPUTE" and not node.target_output:
            print(f"  ⚠️ COMPUTE node {node.id} missing target_output — will not write to graph.")
 
        # Remove self-reference from dependencies
        if node.target_output:
            safe_deps = [d for d in node.dependencies if d != node.target_output]
            updates["dependencies"] = safe_deps
 
        cleaned.append(node.model_copy(update=updates) if updates else node)
 
    return decision.model_copy(update={"extracted_tasks": cleaned})
 
 
# ============================================================================
# DECOMPOSER MODULE
# ============================================================================
 
def _validate_decomposer_decision(decision: DecomposerDecision) -> DecomposerDecision:
    """
    Enforce structural contracts post-LLM.
 
    Per-node contracts:
    - COMPUTE: must have target_output (warn if missing)
    - DECOMPOSE / EXPAND: target_output must be null
    - OUTPUT: target_output must be [ORCHESTRATION_END], dependencies must be non-empty
 
    Graph-level contracts:
    - Exactly one OUTPUT node must exist
    - Self-referencing dependencies are stripped
    """
    cleaned = []
    output_nodes = []
 
    for node in decision.extracted_tasks:
        updates = {}
 
        if node.task_type == "COMPUTE":
            if not node.target_output:
                print(f"  ⚠️ COMPUTE node {node.id} missing target_output — will not write to graph.")
 
        elif node.task_type in ("DECOMPOSE", "EXPAND"):
            if node.target_output is not None:
                updates["target_output"] = None
 
        elif node.task_type == "OUTPUT":
            output_nodes.append(node.id)
            # Enforce [ORCHESTRATION_END] as target
            if node.target_output != ORCHESTRATION_END_ID:
                print(f"  ⚠️ OUTPUT node {node.id} has wrong target_output '{node.target_output}'. "
                      f"Correcting to {ORCHESTRATION_END_ID}.")
                updates["target_output"] = ORCHESTRATION_END_ID
            # Warn if no dependencies — premature halt risk
            if not node.dependencies:
                print(f"  ⚠️ OUTPUT node {node.id} has no dependencies — "
                      f"risks firing before results are ready.")
 
        # Strip self-referencing dependencies
        if node.target_output or updates.get("target_output"):
            target = updates.get("target_output", node.target_output)
            safe_deps = [d for d in node.dependencies if d != target]
            if safe_deps != node.dependencies:
                updates["dependencies"] = safe_deps
 
        cleaned.append(node.model_copy(update=updates) if updates else node)
 
    # Graph-level: warn if no OUTPUT node exists
    if not output_nodes:
        print(f"  ⚠️ No OUTPUT node found — orchestration will never halt. "
              f"Compiler must learn to always terminate the plan.")
 
    # Graph-level: warn if multiple OUTPUT nodes exist
    if len(output_nodes) > 1:
        print(f"  ⚠️ Multiple OUTPUT nodes found: {output_nodes}. "
              f"Only one terminal node is permitted.")
 
    return decision.model_copy(update={"extracted_tasks": cleaned})
 
 
# ============================================================================
# DECOMPOSER MODULE
# ============================================================================
 
class DecomposerAgent(dspy.Module):
    """
    The Semantic Slicer.
 
    Receives a scoped explicit_target from the Router.
    Returns a DecomposerDecision containing ordered TaskNodes.
 
    Uses ChainOfThought — the chronological_scratchpad in the signature
    serves as the externalized reasoning step before structural output.
    """
 
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(DecomposerSignature)
 
    def forward(self, explicit_target: str) -> DecomposerDecision:
        prediction = self.prog(explicit_target=explicit_target)
        decision = prediction.decision
 
        # Enforce structural contracts without touching LLM semantics
        decision = _validate_decomposer_decision(decision)
 
        return decision

"-------------------------------Compute Agent Module-------------------------------"
def _validate_compute_decision(decision: ComputeDecision, target_output_id: str) -> ComputeDecision:
    """
    Enforce output contracts post-LLM.
    - output_node.id must match target_output_id exactly
    - If mismatch: warn and null the node (don't silently write to wrong graph location)
    """
    if decision.output_node is None:
        return decision
 
    if decision.output_node.id != target_output_id:
        print(
            f"  ⚠️ output_node.id mismatch: "
            f"expected '{target_output_id}', got '{decision.output_node.id}'. "
            f"Nulling output to prevent bad graph write."
        )
        return decision.model_copy(update={"output_node": None})
 
    return decision
 
 
# ============================================================================
# COMPUTE AGENT MODULE
# ============================================================================
 
class ComputeAgent(dspy.Module):
    """
    The Atomic Executor.
 
    Receives a resolved TaskNode (instruction + target_output_id) and
    pre-fetched dependency values as a JSON string.
 
    Returns a DataNode to be upserted into the graph by the orchestrator.
    Uses ChainOfThought — reasoning scratchpad is mandatory for execution audit.
    """
 
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ComputeAgentSignature)
 
    def forward(
        self,
        instruction: str,
        target_output_id: str,
        #dependencies: dict,
    ) -> Optional[DataNode]:
        """
        Args:
            instruction:      Verbatim instruction string from the TaskNode.
            target_output_id: The [BracketedID] this execution must produce.
            dependencies:     Dict mapping [BracketedID] → resolved value.
 
        Returns:
            DataNode ready for graph upsert, or None if execution failed.
        """
        prediction = self.prog(
            #dependencies_json=json.dumps(dependencies, indent=2),
            instruction=instruction,
            target_output_id=target_output_id,
        )
 
        decision: ComputeDecision = prediction.decision
        print(f"  -> Reasoning: {decision.reasoning}")
 
        decision = _validate_compute_decision(decision, target_output_id)
 
        if decision.output_node is None:
            print(f"  ⚠️ Compute returned null for {target_output_id}. Cascade halted.")
            return None
 
        return DataNode(
            id=decision.output_node.id,
            content=decision.output_node.content,
        )

"-------------------------------Data Extractor Agent Module-------------------------------"
def _validate_decision(decision: DataExtractorDecision) -> DataExtractorDecision:
    """
    Drop structurally invalid nodes.
    - id must be non-empty
    - content must not be None (extractor never fabricates)
    """
    valid = []
    for node in decision.extracted_nodes:
        if not node.id:
            print(f"  ⚠️ Dropping node with empty id.")
            continue
        if node.content is None:
            print(f"  ⚠️ Dropping node {node.id} with null content — extractor must not fabricate.")
            continue
        valid.append(node)
 
    return decision.model_copy(update={"extracted_nodes": valid})
 
 
# ============================================================================
# DATA EXTRACTOR MODULE
# ============================================================================
 
class DataExtractorAgent(dspy.Module):
    """
    The Value Harvester.
 
    Receives source text (full query or scoped snippet).
    Returns a list of DataNodes ready for graph insertion.
 
    Uses Predict — no scratchpad needed.
    Extraction is structural recognition: find explicit values, label them, preserve them.
    """
 
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(DataExtractorSignature)
 
    def forward(self, source_text: str) -> list[DataNode]:
        prediction = self.prog(source_text=source_text)
        decision: DataExtractorDecision = prediction.decision
 
        decision = _validate_decision(decision)
 
        return [
            DataNode(id=node.id, content=node.content, description=node.description)
            for node in decision.extracted_nodes
        ]
"-------------------------------Expander Agent Module-------------------------------"
# ============================================================================
# CYCLE DETECTION
# ============================================================================
 
def _has_cycle(tasks: list[ExpanderTaskNode]) -> Optional[str]:
    """
    Topological sort via Kahn's algorithm.
    Returns a description of the cycle if found, None if graph is acyclic.
    """
    id_to_node = {t.id: t for t in tasks}
    in_degree = defaultdict(int)
    dependents = defaultdict(list)
 
    for task in tasks:
        for dep in task.dependencies:
            if dep in id_to_node:
                dependents[dep].append(task.id)
                in_degree[task.id] += 1
 
    queue = deque([t.id for t in tasks if in_degree[t.id] == 0])
    visited = 0
 
    while queue:
        node = queue.popleft()
        visited += 1
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
 
    if visited != len(tasks):
        cycle_nodes = [t.id for t in tasks if in_degree[t.id] > 0]
        return (
            f"Cyclic dependency detected among nodes: {cycle_nodes}. "
            f"These nodes are waiting on each other and will never resolve."
        )
    return None
 
 
# ============================================================================
# VALIDATION HELPER
# ============================================================================
 
def _validate_decision(decision: ExpanderDecision) -> ExpanderDecision:
    """
    Enforce structural contracts post-LLM.
 
    Per-node:
    - FETCH: must have target_output, must have empty dependencies
    - COMPUTE: must have target_output
    - EXPAND: must have target_output (container promise)
    - No task may target [ORCHESTRATION_END]
    - Self-referencing dependencies stripped
 
    Graph-level:
    - Cycle detection — if found, null extracted_tasks and populate halt_reason
    - Mutual exclusion: tasks XOR halt_reason
    """
    cleaned = []
 
    for node in decision.extracted_tasks:
        updates = {}
 
        # No task may claim [ORCHESTRATION_END] — global singleton, Decomposer only
        if node.target_output == ORCHESTRATION_END_ID:
            print(f"  ⚠️ Node {node.id} attempted to write {ORCHESTRATION_END_ID}. "
                  f"Expander does not have this power. Nulling target_output.")
            updates["target_output"] = None
 
        if node.task_type == "FETCH":
            # FETCH reaches outward — no input dependencies
            if node.dependencies:
                print(f"  ⚠️ FETCH node {node.id} has dependencies {node.dependencies}. "
                      f"FETCH nodes must have empty dependencies. Clearing.")
                updates["dependencies"] = []
            if not node.target_output and "target_output" not in updates:
                print(f"  ⚠️ FETCH node {node.id} missing target_output — "
                      f"RAG result has nowhere to land.")
 
        elif node.task_type == "COMPUTE":
            if not node.target_output and "target_output" not in updates:
                print(f"  ⚠️ COMPUTE node {node.id} missing target_output.")
 
        elif node.task_type == "EXPAND":
            if not node.target_output and "target_output" not in updates:
                print(f"  ⚠️ EXPAND node {node.id} missing container target_output — "
                      f"parent graph has nothing to wait for.")
 
        # Strip self-referencing dependencies
        effective_target = updates.get("target_output", node.target_output)
        if effective_target:
            safe_deps = [d for d in node.dependencies if d != effective_target]
            if safe_deps != node.dependencies:
                updates["dependencies"] = safe_deps
 
        cleaned.append(node.model_copy(update=updates) if updates else node)
 
    # Cycle detection on cleaned graph
    cycle_description = _has_cycle(cleaned)
    if cycle_description:
        print(f"  🔴 CYCLE DETECTED: {cycle_description}")
        print(f"  🔴 Rejecting entire sub-graph. LLM must restructure.")
        return decision.model_copy(update={
            "extracted_tasks": [],
            "halt_reason": cycle_description,
        })
 
    # Mutual exclusion enforcement
    if cleaned and decision.halt_reason:
        print(f"  ⚠️ Both extracted_tasks and halt_reason populated. "
              f"Clearing halt_reason — tasks take precedence.")
        return decision.model_copy(update={
            "extracted_tasks": cleaned,
            "halt_reason": None,
        })
 
    return decision.model_copy(update={"extracted_tasks": cleaned})
 
 
# ============================================================================
# EXPANDER MODULE
# ============================================================================
 
class ExpanderAgent(dspy.Module):
    """
    The Sub-Graph Architect.
 
    Receives an abstract goal as explicit_target.
    Returns an ExpanderDecision containing a forward-only sub-graph.
 
    Uses ChainOfThought — reasoning scratchpad forces the LLM to map
    existing vs absent nodes before producing tasks.
    """
 
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ExpanderSignature)
 
    def forward(self, explicit_target: str) -> ExpanderDecision:
        prediction = self.prog(explicit_target=explicit_target)
        decision = prediction.decision
 
        decision = _validate_decision(decision)
 
        return decision