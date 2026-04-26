from __future__ import annotations
 
import json
import random
import sys
from pathlib import Path
from typing import Optional
from openai import OpenAI
import uuid
 
_COMPILE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _COMPILE_DIR.parent
for _p in [str(_COMPILE_DIR), str(_PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
 
import dspy
from dspy.teleprompt import SIMBA, MIPROv2
 
from config import configure_lm
from modules import ExpanderModule
from signatures import ExpanderTaskNode, ExpanderDecision
from judge.judge_pipeline import run_judge_pipeline
from judge.judge_schemas import PipelineInput, QueryInput, DependencyNode, TaskNode
from log_utils import JsonlRunLogger
from chart.smoke_chart import compile_with_smoke_chart, make_tracked_metric as smoke_metric
from chart.prod_chart import compile_with_live_chart, make_tracked_metric as prod_metric

from llm_utils import (
    build_client,
    parse_llm_json_response,
    parse_with_retries,
    OPENAI_API_KEY_HARDCODED,
    OPENROUTER_API_KEY_HARDCODED,
)
# ================================================  ============================
# CONFIG
# ============================================================================

PROVIDER             = "openrouter"
MODEL_CLAUDE         = "openai/gpt-oss-120b"
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
OPENROUTER_HTTP_REFERER = "https://localhost"
OPENROUTER_APP_TITLE = "stateReasoning-1"

# Failed-example logging
FAILED_SCORE_THRESHOLD = 0.4
FAILED_LOG_PATH = "./logs/failed_examples.jsonl"
 
# Dataset logging (refreshed each run)
DATASET_LOG_PATH = "./logs/dataset_generation_log.jsonl"

# SIMBA hyperparameters
# SIMBA_BSIZE     = 2
# SIMBA_NUM_STEPS = 2
DATASET_SIZE    = 15

SMOKE_TEST = False

OPTIMIZED_PATH = "./optimized/expander_optimized.json"
 
QUERY_TEMPERATURE = 0.7
 
DOMAINS = [
    "mathematics", "physics", "law", "medicine",
    "literature", "software", "economics", "biology",
    "linguistics", "history",
]


def _append_failed_example(record: dict) -> None:
    """Append one pretty JSON record. Never raises."""
    try:
        path = Path(FAILED_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
    except Exception:
        return
 
# ============================================================================
# STAGE 0 QUERY GENERATION PROMPT — placeholder
# ============================================================================
 
QUERY_GENERATION_PROMPT = """
You are an adversarial Scenario Generator for an advanced DAG (Directed Acyclic Graph) Execution Engine. 
Your objective is to generate highly diverse, domain-specific query inputs to train a downstream "Expander Agent".

The user will provide you with three variables:
1. Domain (The conceptual field, e.g., physics, law, software)
2. Structural Target (The DAG node type this query should naturally require, e.g., DECOMPOSE, EXPAND, FETCH)
3. Strict Constraint (A boundary you must mathematically or logically enforce in the query)

YOUR JOB:
Translate those variables into a realistic, high-density system query. The query must implicitly FORCE the downstream agent to utilize the Structural Target to succeed, but you must NEVER explicitly name the task types (do not say "use a decompose node").

RULES FOR THE DATA:
1. `instruction`: The exact, imperative command given to the DAG engine. It MUST explicitly reference the IDs of the provided dependencies using brackets (e.g., "Transform [Data_Input]").
2. `description`: The broader context, goal, or environment state of the execution.
3. `dependencies`: An array of pre-existing data blocks already loaded into the graph. ID formats must be strict bracketed nouns (e.g., `[Doc_Brief]`, `[Matrix_A]`). 

CRITICAL SCHEMA ENFORCEMENT RULES:
1. THE INSTRUCTION RULE: The `instruction` field MUST be under 10 words. It must be a sterile, high-level goal (e.g., "Compile the final report"). It CANNOT contain procedural steps, bracketed variable names, or mentions of noise.
2. THE DESCRIPTION RULE: The `description` field is the ONLY place allowed to contain the procedural steps, the conversational noise, and the mandated target output names. Write it like a messy message from a coworker.
3. THE NOISE RULE (FOR C6): If injecting a noise distractor, do NOT put the noise in the `dependencies` array. The noise must ONLY exist as a conversational tangent in the `description` text.
4. THE LITERAL PAYLOAD RULE: The `content` of any dependency MUST be a literal JSON array snippet (e.g., `[{"id": 1, "val": "x"}]`) or a literal code stub (e.g., `def process_data(): return x`). DO NOT use English summaries.

JSON SCHEMA REQUIREMENT (STRICT):
You must output ONLY valid JSON exactly matching this structure. No markdown formatting, no preamble.
{
  "query": {
    "instruction": "<Imperative command referencing dependency IDs>",
    "description": "<Objective summary of the environment>",
    "dependencies": [
      {
        "id": "<[Bracketed_ID]>",
        "content": "<Brief description of what this pre-existing data contains>"
      }
    ]
  }
}

CONSTRAINTS:
- Keep the total vocabulary dense and professional.
- The total JSON output should be concise (under 80 words if possible) to maintain focus.
- Ensure the dependencies provided logically match the instruction.
"""
 
# ============================================================================
# DOMAIN CONTROLLER
# ============================================================================
 
class DomainController:
    """Round-robin domain selection with shuffle per cycle."""
 
    def __init__(self):
        self._cycle: list[str] = []
        self._refill()
 
    def _refill(self):
        self._cycle = DOMAINS.copy()
        random.shuffle(self._cycle)
 
    def next(self) -> str:
        if not self._cycle:
            self._refill()
        return self._cycle.pop()
 

def pick_random_domain(domains: list[str] = DOMAINS) -> str:
    """Select a domain uniformly at random."""
    return random.choice(domains)

 
# ============================================================================
# BLUEPRINT SELECTOR
# ============================================================================
 
BLUEPRINTS = [
    {
        "mode": "STANDARD_ATOMIC_COMPUTE",
        "weight": 25,
        "target_tasks": [
            "COMPUTE: Execute a single, 1-step mathematical, logical, or formatting operation on the provided dependencies."
        ],
        "constraint_prompt": (
            "Request a very simple, 1-step mathematical, logical, or formatting operation on a small provided dataset "
            "(under 5 items). Do NOT include any conversational noise, missing data, or massive volumes. Keep it straightforward."
        ),
        "evaluation_trap": (
            "Tests Baseline Stability. The model must simply spawn a single COMPUTE node to execute the exact instruction "
            "without overcomplicating the graph, hallucinating missing data, or unnecessarily delegating to EXPAND or DECOMPOSE."
        )
    },
    {
        "mode": "C1_C2_COGNITIVE_CHAIN",
        "weight": 15,
        "target_tasks": [
            "EXPAND: Deploy a terminal node to delegate the entire multi-step analytical sequence to a child architect, strictly avoiding local computation because the logic exceeds the 2-step threshold."
        ],
        "constraint_prompt": (
            "Request 4 to 5 sequential analytical or transformative steps on a small dataset (under 5 items). "
            "Ensure the logic is highly nested and cannot be solved in a single calculation."
        ),
        "evaluation_trap": (
            "Tests C1 (Hierarchy) and C2 (Terminal Delegation). The model must realize that logic > 2 steps "
            "forbids a COMPUTE node. It must spawn an EXPAND node and delegate the instruction to a child architect "
            "without trying to solve the math itself."
        )
    },
    {
        "mode": "C6_ADVERSARIAL_NOISE",
        "weight": 15,
        "target_tasks": [
            "EXPAND: Process the valid target dataset using the required tools.",
            "IGNORE (Implicit): The system must actively drop the conversational noise without spawning a FETCH or DECOMPOSE node for it."
        ],
        "constraint_prompt": (
            "Request a multi-step analysis (3+ steps) on a valid target dataset, but inject a massive, "
            "conversational distractor. Have the 'user' mention a gigantic external dataset (e.g., '10,000 raw server logs') "
            "and explicitly command the system to completely ignore it."
        ),
        "evaluation_trap": (
            "Tests C6 (Noise Ignorance). The orchestrator must type 'Step 0 (Noise): Dropping [Noise]' in its reasoning. "
            "Crucially, it must NOT spawn a FETCH node to retrieve the noise, and it must NOT pass the noise into "
            "the EXPAND node's instruction payload."
        )
    },
    {
        "mode": "C4_MID_STATE_FIDELITY",
        "weight": 15,
        "target_tasks": [
            "EXPAND: Process the upstream data already explicitly present in the dependencies array.",
            "STRICT_BINDING (Implicit): Force the 'target_output' attribute of the node to precisely match the user's mandated variable name."
        ],
        "constraint_prompt": (
            "Simulate a pipeline mid-state handoff. Provide 1 to 2 upstream variables in brackets in the dependencies array "
            "(e.g., [Cleaned_Telemetry_Data]). Request a 3-step operation on this data, and rigidly mandate that the final "
            "output MUST be saved exactly as a specific bracketed variable name (e.g., 'Save as [Final_Audit_Matrix]')."
        ),
        "evaluation_trap": (
            "Tests C4 (Namespace Fidelity). The model must NOT hallucinate variables or spawn a FETCH node because the "
            "data is already in the dependencies. It must also perfectly bind the `target_output` of the EXPAND node to "
            "the mandated variable name, resisting the urge to invent its own naming convention."
        )
    },
    {
        "mode": "C1_FRACTAL_BRIDGE",
        "weight": 15,
        "target_tasks": [
            "DECOMPOSE: Chunk the high-volume target dataset (40-60 items) into manageable batches of 10.",
            "EXPAND: Delegate the complex, multi-step logic applied to each of those newly created chunks."
        ],
        "constraint_prompt": (
            "Request a complex 4-step transformation on a massive dataset of 40 to 60 items. "
            "Combine high data volume with deep logical complexity."
        ),
        "evaluation_trap": (
            "Tests the Fractal Bridge. The model must recognize that items > 10 requires a DECOMPOSE node first, "
            "followed by an EXPAND node to handle the logic > 2 steps for each chunk. It must correctly chain "
            "the `target_output` of the DECOMPOSE node into the `dependencies` of the EXPAND node."
        )
    },
    {
        "mode": "EPISTEMIC_VOID_FETCH",
        "weight": 10,
        "target_tasks": [
            "FETCH: Retrieve the missing target dataset from the external source because it was not provided in the dependencies.",
            "EXPAND: Apply the requested multi-step analysis on the newly fetched data."
        ],
        "constraint_prompt": (
            "Request a 3-step analysis on a highly specific dataset, but deliberately omit the dataset from the "
            "`dependencies` array (provide only the processing scripts). Do not mention where the data lives."
        ),
        "evaluation_trap": (
            "Tests Epistemic Boundaries. The model must physically check the dependencies array, realize the target "
            "data is missing, and insert a lazy FETCH node to retrieve it before spawning the EXPAND node. If it skips "
            "the FETCH, it fails."
        )
    },
    {
        "mode": "HALTING_DEADLOCK_PREVENTION",
        "weight": 5,
        "target_tasks": [
            "HALT / ORCHESTRATION_END: Terminate the graph immediately. Do not attempt to process the workflow due to the missing or unresolvable execution state."
        ],
        "constraint_prompt": (
            "Request a multi-step operation that explicitly requires pushing data to a third-party server, but "
            "intentionally state that the API keys, credentials, or target URLs are currently lost, revoked, or unknown."
        ),
        "evaluation_trap": (
            "Tests the Epistemic Firewall. The model must recognize an unresolvable execution state and trigger an "
            "abrupt halt or Orchestration End Node rather than hallucinating fake API keys to satisfy the prompt."
        )
    }
]
 
 
def get_blueprint() -> dict:
    return random.choices(BLUEPRINTS, weights=[b["weight"] for b in BLUEPRINTS], k=1)[0]
 
 
# ============================================================================
# QUERY GENERATOR — Claude via OpenRouter
# ============================================================================
 
def build_claude_client() -> OpenAI:
    return build_client(
        provider=PROVIDER,
        openai_api_key_hardcoded=OPENAI_API_KEY_HARDCODED,
        openrouter_api_key_hardcoded=OPENROUTER_API_KEY_HARDCODED,
        openrouter_base_url=OPENROUTER_BASE_URL,
        openrouter_http_referer=OPENROUTER_HTTP_REFERER,
        openrouter_app_title=OPENROUTER_APP_TITLE,
    )
 
 
def generate_query(
    claude_client: OpenAI,
    domain: str,
    blueprint: dict,
) -> Optional[dict]:
    """Generate one adversarial query via Claude."""
    user_prompt = (
        f"Domain: {domain}\n"
        f"Structural Target: {blueprint['target_tasks']}\n"
        f"Blueprint Mode: {blueprint['evaluation_trap']}\n"
        f"Strict Constraint: {blueprint['constraint_prompt']}\n\n"
        "Generate the Query JSON now. Total JSON < 60 words."
    )
 
    try:
        # Append-only detailed request/parse logging for query generation
        qlog = JsonlRunLogger(path=_PROJECT_DIR / "logs" / "query_generation_log.jsonl")

        def _call(attempt: int) -> str:
            messages = [
                {"role": "system", "content": QUERY_GENERATION_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if attempt > 1:
                messages.append({
                    "role": "system",
                    "content": "Retry: output ONLY valid JSON. No markdown, no extra text.",
                })
            response = claude_client.chat.completions.create(
                model=MODEL_CLAUDE,
                temperature=QUERY_TEMPERATURE,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return response.choices[0].message.content or ""

        data = parse_with_retries(
            call_llm=_call,
            parser_name="parse_llm_json_response",
            max_attempts=3,
            logger=qlog,
            log_stage="compile.query_generation",
            run_id=None,
            domain=domain,
            request_payload={"user_prompt": user_prompt, "blueprint_mode": blueprint.get("mode")},
        )
        return data.get("query", data)
 
    except Exception as e:
        print(f"  ❌ [QueryGen] domain={domain} failed: {e}")
        return None
 
 
# ============================================================================
# DATASET BUILDER
# ============================================================================
 
def build_trainset(
    claude_client: OpenAI,
    size: int = DATASET_SIZE,
) -> list[dspy.Example]:
    """
    Generate `size` diverse query examples using Claude.
    Each example has explicit_target (query JSON string) and domain.
    No gold labels — judge pipeline scores dynamically.
    """
    #controller = pick_random_domain()
    trainset = []
    attempts = 0
    max_attempts = size * 3
 
    print(f"\n[Dataset] Generating {size} training queries...")
 
    while len(trainset) < size and attempts < max_attempts:
        attempts += 1
        domain = pick_random_domain()
        blueprint = get_blueprint()
 
        print(f"  [{len(trainset) + 1}/{size}] domain={domain}, mode={blueprint['mode']}")
 
        query = generate_query(claude_client, domain, blueprint)
        if not query:
            continue
 
        example = dspy.Example(
            explicit_target=json.dumps(query),   # JSON string — matches ExpanderSignature input
            domain=domain,
            blueprint_mode=blueprint["mode"],
        ).with_inputs("explicit_target")
 
        trainset.append(example)
 
    print(f"[Dataset] Built {len(trainset)} examples.")
    return trainset


def write_dataset_log(trainset: list[dspy.Example], *, run_id: str) -> None:
    """
    Write the generated dataset to a separate log file (append-only).
    """
    logger = JsonlRunLogger(path=_PROJECT_DIR / DATASET_LOG_PATH)

    for i, ex in enumerate(trainset):
        try:
            explicit_target_obj = json.loads(ex.explicit_target)
        except Exception:
            explicit_target_obj = ex.explicit_target

        logger.write(
            stage="dataset_example",
            domain=getattr(ex, "domain", None),
            run_id=run_id,
            payload={
                "index": i,
                "domain": getattr(ex, "domain", None),
                "blueprint_mode": getattr(ex, "blueprint_mode", None),
                "explicit_target": explicit_target_obj,
            },
        )
 
 
# ============================================================================
# METRIC — bridge between DSPy and judge pipeline
# ============================================================================
 
def expander_metric(
    example: dspy.Example,
    prediction,
    trace=None,
) -> float:
    """
    DSPy metric function for SIMBA.
 
    1. Parse prediction.decision.extracted_tasks into judge TaskNode schema
    2. Build PipelineInput from example + prediction
    3. Call run_judge_pipeline()
    4. Return normalized_final scalar [0.0, 1.0]
    """
    try:
        # DEBUG — see exactly what Llama produced
        print(f"\n[DEBUG LLAMA RAW] prediction type: {type(prediction)}")
        print(f"[DEBUG LLAMA RAW] prediction attrs: {dir(prediction)}")
        try:
            print(f"[DEBUG LLAMA DECISION] {prediction.decision}")
        except Exception as e:
            print(f"[DEBUG LLAMA DECISION ERROR] {e}")
        try:
            print(f"[DEBUG LLAMA TASKS] {prediction.decision.extracted_tasks}")
        except Exception as e:
            print(f"[DEBUG LLAMA TASKS ERROR] {e}")
    except Exception as e:
        print(f"[DEBUG LLAMA RAW ERROR] {e}")
        
    try:
        # Parse query from example
        query_dict = json.loads(example.explicit_target)
        query_input = QueryInput(
            instruction=query_dict.get("instruction", ""),
            description=query_dict.get("description", ""),
            dependencies=[
                DependencyNode(id=d["id"], content=d.get("content", ""))
                for d in query_dict.get("dependencies", [])
            ],
        )
 
        # ExpanderModule.forward() returns an ExpanderDecision directly.
        # SIMBA passes that return value as `prediction`, so there is no
        # wrapping `.decision` attribute — just use prediction itself.
        if prediction is None:
            return 0.0

        # MIPROv2 passes ExpanderDecision directly as prediction
        # since forward() returns ExpanderDecision, not a DSPy wrapper
        from signatures import ExpanderDecision as _ExpanderDecision
        if isinstance(prediction, _ExpanderDecision):
            expander_decision = prediction
        elif hasattr(prediction, "decision"):
            expander_decision = prediction.decision
        elif hasattr(prediction, "extracted_tasks"):
            expander_decision = prediction
        else:
            print(f"  ⚠️ [Metric] Unknown prediction type: {type(prediction)}")
            return 0.0
        # Parse prediction tasks — ExpanderTaskNode → judge TaskNode
        extracted = (expander_decision.extracted_tasks or []) if expander_decision else []
        judge_tasks = [
            TaskNode(
                id=t.id,
                task_type=t.task_type,
                instruction=t.instruction,
                dependencies=t.dependencies,
                target_output=t.target_output,
                description=t.description,
                reasoning=t.reasoning,
            )
            for t in extracted
        ]
 
        pipeline_input = PipelineInput(
            domain=example.domain,
            query=query_input,
            extracted_tasks=judge_tasks,
        )
 
        result = run_judge_pipeline(pipeline_input)
        try:
            from chart.prod_chart import BUFFER
            BUFFER.record(
                step=len(BUFFER.steps),
                normalized_final=result.normalized_final,
                structural_score=result.structural_score,
                per_task_score=result.per_task_score,
                semantic_score=result.semantic_score,
                coverage_score=result.coverage_score,
            )
        except Exception:
            pass  # never let chart recording crash the metric

        print(f"  [Metric] score={result.normalized_final:.4f}")
        if result is None:
            _append_failed_example({
                "reason": "pipeline_returned_none",
                "threshold": FAILED_SCORE_THRESHOLD,
                "domain": getattr(example, "domain", None),
                "blueprint_mode": getattr(example, "blueprint_mode", None),
                "query": query_dict,
                "predicted_tasks": [t.model_dump() for t in judge_tasks],
            })
            return 0.0
 
        if result.normalized_final < FAILED_SCORE_THRESHOLD:
            _append_failed_example({
                "reason": "below_threshold",
                "threshold": FAILED_SCORE_THRESHOLD,
                "score": result.normalized_final,
                "domain": getattr(example, "domain", None),
                "blueprint_mode": getattr(example, "blueprint_mode", None),
                "query": query_dict,
                "predicted_tasks": [t.model_dump() for t in judge_tasks],
                "score_breakdown": result.model_dump(),
            })

        print(f"  [Metric] score={result.normalized_final:.4f}")
        return result.normalized_final
 
    except Exception as e:
        try:
            q = json.loads(example.explicit_target) if hasattr(example, "explicit_target") else None
        except Exception:
            q = getattr(example, "explicit_target", None)
        _append_failed_example({
            "reason": "metric_exception",
            "threshold": FAILED_SCORE_THRESHOLD,
            "domain": getattr(example, "domain", None),
            "blueprint_mode": getattr(example, "blueprint_mode", None),
            "query": q,
            "error": str(e),
        })
        print(f"  [Metric] Failed: {e}")
        return 0.0
 
 
# ============================================================================
# COMPILE
# ============================================================================
 
# def compile_expander(trainset: list[dspy.Example]) -> ExpanderModule:
#     """
#     Run SIMBA to optimize ExpanderModule prompt.
#     Returns the optimized module.
#     """
    
 
#     print(f"\n[SIMBA] Compiling ExpanderModule...")
#     print(f"  bsize={SIMBA_BSIZE}, num_steps={SIMBA_NUM_STEPS}")
#     print(f"  trainset size={len(trainset)}")
 
#     optimizer = MIPROv2(metric=expander_metric, auto="light", num_threads=1)

#     optimized = optimizer.compile(
#         ExpanderModule(),
#         trainset=trainset,
#         max_bootstrapped_demos=2,
#         max_labeled_demos=2,
#         requires_permission_to_run=False,
#         provide_traceback=True,
#     )
 
#     return optimized
def compile_expander(trainset: list[dspy.Example]) -> ExpanderModule:
    print(f"\n[MIPROv2] Compiling ExpanderModule...")
    
    if SMOKE_TEST:
        print(f"  Mode: TIER 1 - SMOKE TEST (Fast)")
        tracked_metric = smoke_metric(expander_metric)
        smoke_trainset = trainset[:2]  
        
        optimizer = MIPROv2(metric=tracked_metric, auto=None, num_candidates=2, num_threads=1)
        return compile_with_smoke_chart(optimizer, ExpanderModule(), smoke_trainset)
        
    else:
        # ==========================================
        # TIER 2: PRODUCTION OPTIMIZATION
        # ==========================================
        print(f"  Mode: TIER 2 - PRODUCTION (Deep)")
        print(f"  Trainset size: {len(trainset)} items")
        tracked_metric = prod_metric(expander_metric)
        
        # ADDED auto="medium" here in the constructor
        optimizer = MIPROv2(metric=tracked_metric, auto="medium", num_threads=1)
        
        return compile_with_live_chart(optimizer, ExpanderModule(), trainset)
 
# ============================================================================
# SAVE / LOAD
# ============================================================================
 
def save_optimized(module: ExpanderModule, path: str = OPTIMIZED_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    module.save(path)
    print(f"[Save] Optimized module → {path}")
 
 
def load_optimized(path: str = OPTIMIZED_PATH) -> ExpanderModule:
    module = ExpanderModule()
    module.load(path)
    print(f"[Load] Optimized module ← {path}")
    return module
 
 
# ============================================================================
# ENTRY POINT
# ============================================================================
def load_trainset() -> list[dspy.Example]:
        """
        Load the dataset generated by `write_dataset_log()` back into a DSPy trainset.

        The dataset log file is pretty-printed JSON records separated by blank lines.
        We read each record, extract `payload`, and reconstruct `dspy.Example` objects
        matching `build_trainset()` output.
        """
        path = _PROJECT_DIR / "logs" / "dataset_generation_log.jsonl"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        decoder = json.JSONDecoder()
        idx = 0
        records: list[dict] = []
        while idx < len(text):
            # Skip whitespace/newlines between records
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, next_idx = decoder.raw_decode(text, idx)
            records.append(obj)
            idx = next_idx

        trainset: list[dspy.Example] = []
        for rec in records:
            if not isinstance(rec, dict) or rec.get("stage") != "dataset_example":
                continue
            payload = rec.get("payload") or {}
            explicit_target = payload.get("explicit_target")
            explicit_target_str = (
                json.dumps(explicit_target, ensure_ascii=False)
                if isinstance(explicit_target, (dict, list))
                else str(explicit_target)
            )

            ex = dspy.Example(
                explicit_target=explicit_target_str,
                domain=payload.get("domain"),
                blueprint_mode=payload.get("blueprint_mode"),
            ).with_inputs("explicit_target")
            trainset.append(ex)

        return trainset

if __name__ == "__main__":
    # 1. Configure DSPy LM — Llama 3.1 8B via OpenRouter
    configure_lm("openrouter")
    
    print(f"[DEBUG] dspy.settings.lm: {dspy.settings.lm}")
    print(f"[DEBUG] LM api_key set: {bool(dspy.settings.lm.kwargs.get('api_key'))}")

    
 
    # 2. Build Claude client for query generation
    claude_client = build_claude_client()
 
    # 3. Generate trainset
    run_id = uuid.uuid4().hex
    #trainset = build_trainset(claude_client, size=DATASET_SIZE)
    trainset = load_trainset()
    #write_dataset_log(trainset, run_id=run_id)
 
    # 4. Compile with SIMBA
    optimized = compile_expander(trainset)
 
    # 5. Save
    save_optimized(optimized)
 
    print("\n✅ Compilation complete.")