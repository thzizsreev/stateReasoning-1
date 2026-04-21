"""
Two-Stage Crucible C1 — Epistemic Void Dataset Generator.

Stage 1: Generates the Adversarial Query (The Trap).
Stage 2: Generates the ExpanderDecision (The Halt).
"""

from __future__ import annotations
import json
import random
import os
import sys
from pathlib import Path
from openai import OpenAI
from typing import List, Literal
from pydantic import BaseModel, Field
import random
# Ensure sibling imports (e.g., `llm_utils.py`) resolve when executed from repo root.
_DATASET_DIR = Path(__file__).resolve().parent
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

# Import the project's local `dspy/signatures.py` (avoid the installed `dspy` package).
_DSPY_DIR = Path(__file__).resolve().parents[1]
if str(_DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(_DSPY_DIR))

#from signatures import ExpanderDecision, ExpanderTaskNode  # noqa: E402
from llm_utils import parse_llm_json_response, build_client
from llm_utils import OPENAI_API_KEY_HARDCODED, OPENROUTER_API_KEY_HARDCODED

# ============================================================================
# CONFIG
# ============================================================================

CRUCIBLE_NAME = "C6_EpistemicVoid_TwoStage"
TARGET_COUNT = 1
OUTPUT_PATH = "./crucible_outputs/c6_epistemic_void_twostage.json"

PROVIDER: str = "openrouter"
MODEL_OPENAI = "gpt-4o"
MODEL_OPENROUTER = "meta-llama/llama-3.1-8b-instruct"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = "https://localhost"
OPENROUTER_APP_TITLE = "stateReasoning-1"

DOMAINS = [
    "mathematics", "physics", "law", "medicine",
    "literature", "software", "economics", "biology",
    "linguistics", "history",
]

# ============================================================================
# SYSTEM PROMPTS (PLACEHOLDERS)
# ============================================================================

STAGE_1_SYSTEM_PROMPT = """
You are a Synthetic Data Generator for an enterprise DAG orchestrator.
Your goal is to simulate real-world user requests to stress-test an 8B-parameter routing model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATTRIBUTE DEFINITIONS & CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. `instruction` (Razor-Sharp Goal): Max 10 words. A highly direct, sterile command stating the final desired outcome. NO data volumes, NO procedural steps, NO noise here.
2. `description` (Verbose Context & Organic Noise): Max 100 words. Write this like a chatty, disorganized message from a coworker. It MUST contain:
   - Organic Pretext: A conversational, believable reason for the request.
   - The Target Data (Decompose Bait): Explicitly mention 25 to 50 items (e.g., "45 target sequences").
   - The Distractor (Organic Noise): Invent a highly distracting tangent about a massive, irrelevant dataset that the sender explicitly says to ignore, skip, or drop.
   - Procedural Steps (Expand Bait): List 3+ internal logical operations required on the target data.
3. `dependencies` (Literal Payload - CRITICAL): You MUST output REAL, USABLE DATA. NO summaries. NO descriptions like "Python script for X". 
   - If it is code: Provide the literal function signatures/stubs (e.g., `def run_blast(seq, db): return scores`).
   - If it is data: Provide the literal JSON snippet or CSV headers (e.g., `[{"id": "seq1", "bases": "ACTG..."}]`).
   - Keep the payload under 80 words, but it MUST be syntactically valid raw data/code.

CRITICAL RULE: ALL complexity, volume metrics, and organic noise MUST be hidden inside the `description`. The `dependencies` MUST be raw, executable structures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "query": {
    "instruction": "[Direct, short terminal goal].",
    "description": "[Conversational pretext] + [Target volume 25-50 items] + [Organic Distractor/Noise] + [Multi-step procedure].",
    "dependencies": [
      {
        "id": "[Exact_File_Name_or_Module]",
        "content": "[RAW DATA, JSON FRAGMENT, OR CODE STUBS ONLY. NO ENGLISH DESCRIPTIONS.]"
      }
    ]
  }
}

--- EXAMPLE (DOMAIN: BIOLOGY) ---
{
  "query": {
    "instruction": "Generate the species phylogeny tree summary.",
    "description": "Hey, we need the tree of life data for the conference. Can you grab the 45 protein sequences we isolated yesterday? Also, Dr. Smith uploaded 2000 raw GenBank records to the drive, but please ignore those, they are completely unverified. For our 45 sequences, I need you to run the local alignment, map the returned scores to a directed graph, and then reduce that graph into a single phylogenetic tree structure.",
    "dependencies": [
      {
        "id": "blast_core.py",
        "content": "def run_local_alignment(query_seq, target_db):\\n    '''Returns dict of alignment scores'''\\n    pass\\n\\ndef generate_score_matrix(results):\\n    pass"
      },
      {
        "id": "tree_reducer.py",
        "content": "class DirectedGraph:\\n    def __init__(self, matrix): pass\\n\\n    def reduce_to_tree(self):\\n        return TreeObject()"
      }
    ]
  }
}
"""

STAGE_2_SYSTEM_PROMPT = """
You are the Expander Agent — the Cognitive Architect for an enterprise Directed Acyclic Graph (DAG) orchestration engine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CORE RESPONSIBILITY & HARDWARE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are compiling this DAG for an execution engine powered by an **8-Billion Parameter LLM**. You must aggressively hyper-fragment the work while strictly respecting the bounds of the execution namespace.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C6 NOISE IGNORANCE (ATTENTION SELECTIVITY) - CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The User Query contains "Distractors" (data the user explicitly says to ignore/skip).
1. DO NOT CREATE A FILTER TASK.
2. DO NOT CREATE A FETCH TASK TO RETRIEVE NOISE JUST TO DROP IT.
3. Pretend the noise data simply does not exist.
4. In your FIRST task's `reasoning`, you MUST start with: "Step 0 (Noise): I am explicitly dropping [Noise Item] from the DAG."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 8B HIERARCHY OF ATTENTION (STRICT THRESHOLDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate every step against these limits in this exact order:

1. DECOMPOSE (Context): If target data > 2,000 words or > 10 items -> Spawn `DECOMPOSE` to chunk it. DO NOT decompose arrays smaller than 10 items.
2. EXPAND (Cognition): If logic > 2 distinct transformations -> Spawn `EXPAND` to delegate.
3. FETCH (Epistemic): If REQUIRED data is EXTERNAL to the namespace -> Spawn `FETCH`. NEVER fetch noise or distractor data.
4. COMPUTE (Safe Execution): Use ONLY for trivial 1-step formatting of low-volume data. 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TERMINAL DELEGATION (NO COMPULSORY COMPUTE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are STRICTLY FORBIDDEN from artificially appending a `COMPUTE` node just to "finish" a workflow if the final step involves multiple transformations or a complex merge. The final output of your graph should typically be the `target_output` of a terminal `EXPAND` node.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMESPACE PROVENANCE (ANTI-HALLUCINATION) - CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are STRICTLY FORBIDDEN from inventing dependency variables. Every variable in a `dependencies` array MUST come from:
1. Explicitly listed in the User Query's `dependencies`.
2. Created as the `target_output` of a previous task.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY a valid JSON object. No markdown blocks. Max 5 nodes.

{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_ID]",
        "task_type": "DECOMPOSE | EXPAND | FETCH | COMPUTE",
        "reasoning": "First-person justification explicitly stating Step 0 (Noise), then evaluating Hierarchy thresholds (>10 limit, >2 steps).",
        "dependencies": ["[Incoming_Var]"],
        "instruction": "Precise action... and output as [Output_Var].",
        "target_output": "[Output_Var]",
        "description": "Objective summary."
      }
    ]
  }
}

--- EXAMPLE OF PERFECT C6 GRAPH (NOISE DROPPED, LAZY FETCH) ---
{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_Decompose_Lawsuit_Exhibits]",
        "task_type": "DECOMPOSE",
        "reasoning": "Step 0 (Noise): I am explicitly dropping the 120 pages of discovery documents from the DAG entirely. I will not FETCH them. Step 1 (Hierarchy): The required target data is 45 exhibits, which exceeds the >10 item threshold. Since [Raw_Lawsuit_Exhibits] was provided in the initial dependencies, I am spawning a DECOMPOSE task to chunk them.",
        "dependencies": [
          "[Raw_Lawsuit_Exhibits]"
        ],
        "instruction": "Chunk the 45 items in [Raw_Lawsuit_Exhibits] into batches of 10, outputting as [Chunked_Exhibits].",
        "target_output": "[Chunked_Exhibits]",
        "description": "Drops discovery noise and chunks the target exhibit data."
      },
      {
        "id": "[Task_Expand_Clause_Extraction]",
        "task_type": "EXPAND",
        "reasoning": "Step 2 (Hierarchy): The prompt asks to extract five key clauses, apply non-compete analysis, and count attachments. This multi-step extraction exceeds the >2 logic steps limit. I am deploying a terminal EXPAND node to delegate this complex logic to a child architect using the provided [Contract_Analyzer].",
        "dependencies": [
          "[Chunked_Exhibits]",
          "[Contract_Analyzer]"
        ],
        "instruction": "For each chunk in [Chunked_Exhibits], use the [Contract_Analyzer] to extract the key clauses, apply non-compete analysis, and count attachments. Consolidate into a final report outputting as [Final_Contract_Analysis].",
        "target_output": "[Final_Contract_Analysis]",
        "description": "Delegates the multi-step clause extraction and analysis."
      }
    ]
  }
}
"""

class ExpanderTaskNode(BaseModel):
    id: str = Field(..., description="The bracketed ID of the task.")
    task_type: Literal["FETCH","EXPAND", "COMPUTE", "DECOMPOSE", "OUTPUT"] = Field(..., description="The strict operational type.")
    reasoning: str = Field(..., description="First-person justification for this specific node.")
    dependencies: List[str] = Field(default_factory=list, description="List of inherited bracketed variables.")
    instruction: str = Field(..., description="The precise action to execute.")
    target_output: str = Field(..., description="The pre-declared bracketed variable this node outputs. ORCHESTRATION_END if halting.")
    description: str = Field(..., description="Objective summary of what this node does.")

class ExpanderDecision(BaseModel):
    extracted_tasks: List[ExpanderTaskNode] = Field(default_factory=list, description="The flat topological array of tasks.")

# ============================================================================
# DOMAIN CONTROLLER
# ============================================================================

class DomainController:
    def __init__(self):
        self._cycle = []
        self._refill()

    def _refill(self):
        self._cycle = DOMAINS.copy()
        random.shuffle(self._cycle)

    def next(self) -> str:
        if not self._cycle:
            self._refill()
        return self._cycle.pop()

# ============================================================================
# GENERATOR PIPELINE
# ============================================================================
def get_structural_blueprint() -> dict:
    """
    Deterministically selects a structural trap for the LLM to generate.
    Balances the dataset across Decompose, Expand, and Compute.
    """
    blueprints = [
        {
            "mode": "VOLUME_MAX",
            "target": "DECOMPOSE",
            "constraint": "Request processing of 40-50 discrete items. Force a Map-Reduce segmentation logic.",
            "weight": 30
        },
        {
            "mode": "LOGIC_MAX",
            "target": "EXPAND",
            "constraint": "Request 4+ sequential analytical steps. Force deep cognitive chaining.",
            "weight": 40
        },
        {
            "mode": "EFFICIENCY_TEST",
            "target": "COMPUTE",
            "constraint": "Request a single simple transformation on under 5 items. Test for direct execution.",
            "weight": 20
        }
        
    ]
    
    # Weighted random selection
    modes = [b['mode'] for b in blueprints]
    weights = [b['weight'] for b in blueprints]
    selected_mode = random.choices(blueprints, weights=weights, k=1)[0]
    
    return selected_mode

def generate_stage_1_query(client: OpenAI, domain: str) -> dict | None:
    """Stage 1: Generate the adversarial user query with structural pressure."""
    
    # 1. Get the deterministic structural pressure
    blueprint = get_structural_blueprint()
    
    # 2. Append the pressure to the user prompt
    user_prompt = (
        f"Domain: {domain}\n"
        f"Structural Target: {blueprint['target']}\n"
        f"Strict Constraint: {blueprint['constraint']}\n\n"
        "Generate the Query JSON now. Remember: Total JSON < 60 words."
    )
    print(f"user_prompt_stage_1: {user_prompt}")
    try:
        response = client.chat.completions.create(
            model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
            temperature=0.7, 
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": STAGE_1_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        print(f"raw: {raw}")
        data = parse_llm_json_response(raw)
        print(f"data: {data}")
        # Ensure it's nested properly
        if "query" in data:
            return data["query"]
        return data 

    except Exception as e:
        print(f"  ❌ [Stage 1 - {domain}] Failed to generate query: {e}")
        return None

def generate_stage_2_decision(client: OpenAI, query_data: dict) -> dict | None:
    """Stage 2: Generate the Expander's response to the query."""
    user_prompt = f"Here is the user's query:\n{json.dumps(query_data, indent=2)}\n\nGenerate the Decision JSON now."
    print(f"user_prompt_stage_2: {user_prompt}")
    try:
        response = client.chat.completions.create(
            model=(MODEL_OPENROUTER if PROVIDER.lower() == "openrouter" else MODEL_OPENAI),
            temperature=0.2, # Lower temp for strict architectural adherence
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": STAGE_2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = parse_llm_json_response(raw)
        
        # Unpack and validate via Pydantic
        decision_raw = data.get("decision", data) # Handle if LLM wrapped it or not
        decision = ExpanderDecision(
            reasoning=decision_raw.get("reasoning", ""),
            extracted_tasks=[
                ExpanderTaskNode(**t) for t in decision_raw.get("extracted_tasks", [])
            ],
            halt_reason=decision_raw.get("halt_reason"),
        )
        return decision.model_dump()

    except Exception as e:
        print(f"  ❌ [Stage 2] Failed to generate decision: {e}")
        return None

def generate_one(client: OpenAI, domain: str | None = None) -> dict | None:
    """Orchestrates the two-stage generation process."""
    domain = domain or random.choice(DOMAINS)
    
    # Run Stage 1
    query_data = generate_stage_1_query(client, domain)
    print(f"query_data: {query_data}")
    if not query_data:
        return None
        
    # Run Stage 2
    decision_data = generate_stage_2_decision(client, query_data)
    print(f"decision_data: {decision_data}")
    if not decision_data:
        return None

    # Compile the final Golden Trace
    return {
        "domain": domain,
        "query": query_data, 
        "decision": decision_data,
    }

def run(target_count: int = TARGET_COUNT, domain: str | None = None) -> list[dict]:
    client = build_client(
        provider=PROVIDER,
        openai_api_key_hardcoded=OPENAI_API_KEY_HARDCODED,
        openrouter_api_key_hardcoded=OPENROUTER_API_KEY_HARDCODED,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_http_referer=OPENROUTER_HTTP_REFERER,
        openrouter_app_title=OPENROUTER_APP_TITLE,
    )
    # Load any existing dataset so we can append incrementally.
    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(OUTPUT_PATH)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8") or "[]")
            dataset = existing if isinstance(existing, list) else []
        except Exception:
            dataset = []
    else:
        dataset = []
    generated = 0

    domain_label = domain if domain is not None else "RANDOM"
    print(f"\n[{CRUCIBLE_NAME}] Generating {target_count} examples using Two-Stage Pipeline (domain={domain_label})...")

    while generated < target_count:
        chosen_domain = domain or random.choice(DOMAINS)
        print(f"  [{generated + 1}/{target_count}] domain={chosen_domain}")
        
        example = generate_one(client, chosen_domain)
        if example:
            dataset.append(example)
            generated += 1
            # Persist after each generated example (so we don't lose progress).
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(dataset, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print(f"  ✅ Saved {len(dataset)} examples → {OUTPUT_PATH}")
    return dataset

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    client = build_client(
        provider=PROVIDER,
        openai_api_key_hardcoded=OPENAI_API_KEY_HARDCODED,
        openrouter_api_key_hardcoded=OPENROUTER_API_KEY_HARDCODED,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_http_referer=OPENROUTER_HTTP_REFERER,
        openrouter_app_title=OPENROUTER_APP_TITLE,
    )
    
    # Test a single run
    # ex = generate_one(client, domain="physics")
    # print(json.dumps(ex, indent=4))
    # if ex:
    #     print("\n[Two-Stage Pipeline Success] Final Output:")
    #     print(json.dumps(ex, indent=4))
    run(target_count=TARGET_COUNT)