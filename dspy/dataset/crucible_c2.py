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

CRUCIBLE_NAME = "C2_EpistemicVoid_TwoStage"
TARGET_COUNT = 20
OUTPUT_PATH = "./crucible_outputs/c2_epistemic_void_twostage.json"

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
You are a Minimalist Data Generator for an 8B-parameter DAG orchestrator.
Your goal is to generate extremely short, internal-only workflow requests that force C2 Topological Forking and Merging (Parallel Execution).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE C2 PARALLEL TRAP (STRICT LIMITS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your output JSON MUST be under 60 words. The query MUST contain:

1. THE FORK (Parallel Inputs): Introduce 2 completely distinct, independent data sources or subjects (e.g., "the Q1 revenue" and "the Q1 expenses").
2. INDEPENDENT LOGIC (Parallel Execution): Demand a specific, separate action for *each* data source (e.g., "project the revenue trends" and "audit the expenses for anomalies").
3. THE MERGE (Convergence): Demand that the separate results be combined into a single final internal output.

CRITICAL: The logic applied to Data A must NOT depend on Data B. They must be capable of parallel execution.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "query": {
    "instruction": "[Fork Data A & B] + [Independent Actions for A & B] + [Merge the Results].",
    "description": "5 words max.",
    "dependencies": [
      {
        "id": "[Rule_Data]",
        "content": "10 words max."
      }
    ]
  }
}

--- EXAMPLE (DOMAIN: HR) ---
{
  "query": {
    "instruction": "Review the candidate's resume to extract their technical skills, and separately review the technical test to score their coding proficiency. Finally, combine the skills and score into a final internal hiring recommendation.",
    "description": "Parallel HR review and merge.",
    "dependencies": [
      {
        "id": "[Hiring_Rubric]",
        "content": "Skill requirements and test answer key."
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
You are compiling this DAG for an execution engine powered by an **8-Billion Parameter LLM**. 
Because your downstream execution nodes have strict KV-cache limits and weak multi-step reasoning, you must aggressively hyper-fragment the work. 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE OUTPUT GENERATION LIMIT (MAX 5 NODES) - CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An 8B model suffers from Output Token Drift if it generates more than 5 JSON nodes in a single pass. 
You are STRICTLY FORBIDDEN from generating an `extracted_tasks` array with more than 5 nodes.

If the User Query requires massive parallelization (e.g., spawning 6 parallel DECOMPOSE tasks or 4 distinct EXPAND paths):
1. Do NOT generate them all.
2. You must emit ONE "Super Router" node to club the tasks together. 
   - Use task_type: "EXPAND"
   - In `reasoning`, state: "Graph width exceeds 5-node generation limit. Delegating fan-out to a dynamic sub-router."
   - In `instruction`, command the sub-router to dynamically spawn the necessary parallel child nodes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPOLOGICAL INDEPENDENCE (C2 PARALLELISM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST construct true Directed Acyclic Graphs, not just linear lists.
1. THE FORK: If Task B does NOT require the output of Task A, they MUST NOT depend on each other. Emit them as parallel nodes.
2. THE MERGE: If Task C requires the output of both Task A and Task B, its dependencies array MUST explicitly list both: `["[Output_A]", "[Output_B]"]`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 8B HIERARCHY OF ATTENTION (STRICT THRESHOLDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate every step against these limits:
0. FETCH: If data exists outside the namespace -> Spawn `FETCH`. NO dependencies ([]). 
1. DECOMPOSE: If data > 2,000 words or > 10 items -> Spawn `DECOMPOSE` to chunk it.
2. EXPAND: If logic > 2 distinct transformations -> Spawn `EXPAND` to delegate.
3. COMPUTE: Use ONLY for 1-to-1 mappings, simple formatting, or merging two fully processed variables together.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMESPACE UNIFORMITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The `instruction` string MUST end with "...and output as [Bracketed_Variable_Name]".
2. The `target_output` MUST match exactly.
3. Downstream `dependencies` MUST match exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY a valid JSON object. No markdown blocks. Max 5 nodes.

{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_ID]",
        "task_type": "FETCH | COMPUTE | DECOMPOSE | EXPAND",
        "reasoning": "First-person justification evaluating topology, 8B thresholds, and the 5-node limit.",
        "dependencies": ["[Incoming_Var]"],
        "instruction": "Precise action... and output as [Output_Var].",
        "target_output": "[Output_Var]",
        "description": "Objective summary."
      }
    ]
  }
}

--- EXAMPLE OF A PERFECT C2 FORK/MERGE GRAPH (EXACTLY 5 NODES) ---
{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_Fetch_Candidate_Resume]",
        "task_type": "FETCH",
        "reasoning": "I need the candidate's resume, which is currently outside the namespace. I am spawning a FETCH task.",
        "dependencies": [],
        "instruction": "Retrieve the candidate's submitted resume and output as [Raw_Resume].",
        "target_output": "[Raw_Resume]",
        "description": "Fetches the resume document."
      },
      {
        "id": "[Task_Fetch_Technical_Test]",
        "task_type": "FETCH",
        "reasoning": "I also need the candidate's technical test. This is independent of the resume and requires its own FETCH task.",
        "dependencies": [],
        "instruction": "Retrieve the candidate's submitted technical test and output as [Raw_Tech_Test].",
        "target_output": "[Raw_Tech_Test]",
        "description": "Fetches the technical test document."
      },
      {
        "id": "[Task_Compute_Extract_Skills]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Topology & Hierarchy: This logic only requires [Raw_Resume] and [Hiring_Rubric]. It does NOT depend on the technical test. It is a single internal extraction step, safe for a parallel COMPUTE node.",
        "dependencies": [
          "[Raw_Resume]",
          "[Hiring_Rubric]"
        ],
        "instruction": "Extract the candidate's technical skills from [Raw_Resume] using the criteria in [Hiring_Rubric], and output as [Extracted_Skills].",
        "target_output": "[Extracted_Skills]",
        "description": "Extracts skills from the resume."
      },
      {
        "id": "[Task_Expand_Score_Test]",
        "task_type": "EXPAND",
        "reasoning": "Evaluating Topology & Hierarchy: Operates entirely in parallel to resume extraction. Scoring requires >2 steps (grading, weighting, calculating). I am spawning a parallel EXPAND node.",
        "dependencies": [
          "[Raw_Tech_Test]",
          "[Hiring_Rubric]"
        ],
        "instruction": "Grade the [Raw_Tech_Test] answers against the [Hiring_Rubric], weight the scores, calculate the final proficiency metric, and output as [Test_Score].",
        "target_output": "[Test_Score]",
        "description": "Delegates multi-step grading logic to a child architect."
      },
      {
        "id": "[Task_Compute_Merge_Recommendation]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Topology (The Merge): The parallel tracks are complete. I must now merge [Extracted_Skills] and [Test_Score]. I am deploying a final COMPUTE node. Total node count is exactly 5, respecting the generation limit.",
        "dependencies": [
          "[Extracted_Skills]",
          "[Test_Score]"
        ],
        "instruction": "Combine the [Extracted_Skills] and [Test_Score] into a final internal hiring recommendation summary, and output as [Final_Hiring_Recommendation].",
        "target_output": "[Final_Hiring_Recommendation]",
        "description": "Merges the parallel evaluations into a single final recommendation."
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