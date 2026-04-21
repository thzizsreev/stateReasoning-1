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

CRUCIBLE_NAME = "C1_EpistemicVoid_TwoStage"
TARGET_COUNT = 20
OUTPUT_PATH = "./crucible_outputs/c1_epistemic_void_twostage.json"

PROVIDER: str = "openrouter"
MODEL_OPENAI = "gpt-4o"
MODEL_OPENROUTER = "anthropic/claude-sonnet-4.6"

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
Your goal is to generate extremely short, internal-only workflow requests that force structural delegation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT NO-ACTION POLICY (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER ask to "Upload," "Send," "Email," "Save," or "Post" to an external system.
- Focus ONLY on Internal Transformation: Summarizing, Comparing, Scoring, or Reformatting.
- The "Trap" is now purely COGNITIVE: Give too much data (Volume) and too many steps (Logic).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 8B DELEGATION TRAP (STRICT LIMITS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your output JSON MUST be under 50 words. The query MUST contain:

1. VOLUME (Decompose Bait): Mention processing 25 to 50 items (e.g., "Analyze 40 logs").
2. LOGIC (Expand Bait): Demand 3 sequential internal steps (e.g., "Find the errors, count the frequency, and draft a summary report").
3. THE NAMESPACE TRAP: The final result must be "stored in the internal namespace" or "held in memory." Do NOT provide an external path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "query": {
    "instruction": "[Volume] + [3-Step Internal Logic] + [End in Namespace].",
    "description": "5 words max.",
    "dependencies": [
      {
        "id": "[Rule_Data]",
        "content": "10 words max."
      }
    ]
  }
}

--- EXAMPLE (DOMAIN: FINANCE) ---
{
  "query": {
    "instruction": "Scan the 45 attached expense receipts. Categorize by department, flag totals over $500, and generate a final internal audit table.",
    "description": "Audit 45 receipts internally.",
    "dependencies": [
      {
        "id": "[Audit_Rules]",
        "content": "List of department codes and tax thresholds."
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

You do not summarize your thoughts globally. Your metacognition must be distributed locally into the `reasoning` field of EACH individual task node. 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMESPACE UNIFORMITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The `instruction` string MUST end with "...and output as [Bracketed_Variable_Name]".
2. The `target_output` MUST match exactly.
3. Downstream `dependencies` MUST match exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 8B HIERARCHY OF ATTENTION (STRICT THRESHOLDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When analyzing a step, you MUST evaluate it against the 8B model's physical limits in this exact order:

0. PREREQUISITE DATA INGESTION (FETCH):
   - Threshold: Does the required data currently exist outside the namespace (e.g., in a database, repository, or external document)?
   - If YES -> You MUST spawn a `FETCH` task to retrieve it. `FETCH` nodes take NO dependencies ([]). Pre-declare the output.

1. EVALUATE CONTEXT LOAD (DECOMPOSE):
   - Threshold: Does the data exceed roughly 2,000 words? Are there more than 10 distinct items to process? 
   - If YES -> You MUST spawn a `DECOMPOSE` task. An 8B compute node will hallucinate if you feed it a massive document. Delegate it to a chunking sub-router.

2. EVALUATE COGNITIVE LOAD (EXPAND):
   - Threshold: Does the logic require more than TWO (2) distinct transformations (e.g., Extract -> Analyze -> Format)? 
   - If YES -> You MUST spawn an `EXPAND` task. An 8B model cannot chain complex thoughts in a single prompt. Delegate it to a child architect.

3. EXECUTE (COMPUTE):
   - Threshold: Use this ONLY for a 1-to-1 mapping, a single simple summary, or a direct transmission. If the task is dense, do NOT use COMPUTE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE EPISTEMIC BOUNDARY & EMERGENCY HALT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-MOCKING OVERRIDE: You are strictly forbidden from generating placeholder URLs, mock URIs, or generic endpoints. If the literal string "http..." or the exact target ID is not explicitly in the prompt, YOU DO NOT HAVE IT. 
- You must deploy an `OUTPUT` task.
- In its `reasoning`, trigger the exact phrase: "[EPISTEMIC SPIKE DETECTED]".
- In its `instruction`, command the engine to halt immediately.
- Its `target_output` MUST be `[ORCHESTRATION_END]`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY a valid JSON object containing a flat list of tasks. No markdown blocks.

{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_ID]",
        "task_type": "FETCH | COMPUTE | DECOMPOSE | EXPAND",
        "reasoning": "First-person justification evaluating the 8B Hierarchy limits or Epistemic Boundary.",
        "dependencies": ["[Incoming_Var]"],
        "instruction": "Precise action... and output as [Output_Var].",
        "target_output": "[Output_Var]",
        "description": "Objective summary of what this node does."
      }
    ]
  }
}

--- EXAMPLE OF A PERFECT PARTIAL GRAPH (WITH HIERARCHY AND EPISTEMIC HALT) ---
example 1:
{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_Fetch_Transaction_Ledgers]",
        "task_type": "FETCH",
        "reasoning": "I need the raw historical data to begin the audit. I am delegating a FETCH task to retrieve the target company's financial history and pre-declaring [Raw_10_Year_Ledgers].",
        "dependencies": [],
        "instruction": "Query the target company's financial database for the 10-year raw transaction ledgers and output as [Raw_10_Year_Ledgers].",
        "target_output": "[Raw_10_Year_Ledgers]",
        "description": "Retrieves the massive historical dataset required for the audit."
      },
      {
        "id": "[Task_Decompose_Ledger_Audit]",
        "task_type": "DECOMPOSE",
        "reasoning": "Evaluating Hierarchy (Context Load): A 10-year ledger contains millions of rows, and cross-referencing it against a 500-page tax code will obliterate the context window. I must spawn a DECOMPOSE task to delegate this volume to a map-reduce sub-router that can chunk the audit by year.",
        "dependencies": [
          "[Raw_10_Year_Ledgers]",
          "[Data_Federal_Tax_Code]"
        ],
        "instruction": "Chunk the [Raw_10_Year_Ledgers] by fiscal year, cross-reference each chunk against the [Data_Federal_Tax_Code] to identify anomalies, compile the findings, and output as [Audit_Anomaly_Report].",
        "target_output": "[Audit_Anomaly_Report]",
        "description": "Delegates massive data volume processing to avoid context window collapse."
      },
      {
        "id": "[Task_Expand_Mitigation_Strategy]",
        "task_type": "EXPAND",
        "reasoning": "Evaluating Hierarchy (Cognitive Load): The user wants an 'optimal tax liability mitigation strategy.' This is highly abstract, requires multi-step legal reasoning, and distracts from the main DAG topology. I cannot use a simple compute node. I am spawning an EXPAND task to delegate this logic to a child cognitive architect.",
        "dependencies": [
          "[Audit_Anomaly_Report]",
          "[Data_Federal_Tax_Code]"
        ],
        "instruction": "Analyze the [Audit_Anomaly_Report] to determine the optimal M&A tax mitigation strategy, formulate the legal justification, and output as [Tax_Mitigation_Strategy].",
        "target_output": "[Tax_Mitigation_Strategy]",
        "description": "Delegates highly abstract legal reasoning to a specialized child architect."
      },
      {
        "id": "[Task_Compute_Format_Risk_Packet]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Hierarchy (Execution): With the anomaly report and the mitigation strategy complete, I just need to combine them into the final packet. The context and cognitive loads are safe. I am delegating a COMPUTE task to execute the formatting.",
        "dependencies": [
          "[Audit_Anomaly_Report]",
          "[Tax_Mitigation_Strategy]"
        ],
        "instruction": "Combine the [Audit_Anomaly_Report] and [Tax_Mitigation_Strategy] into standard federal filing formatting and output as [M_And_A_Risk_Packet].",
        "target_output": "[M_And_A_Risk_Packet]",
        "description": "Executes standard document compilation and formatting."
      },
      {
        "id": "[Task_Halt_Missing_EDGAR_Credentials]",
        "task_type": "COMPUTE",
        "reasoning": "[EPISTEMIC SPIKE DETECTED]. I need to file the [M_And_A_Risk_Packet] to the SEC EDGAR system. However, the user completely omitted the target company's CIK (Central Index Key) and EDGAR access passwords. I cannot hallucinate federal filing credentials. I am deploying the emergency OUTPUT node to abort the orchestration.",
        "dependencies": [
          "[M_And_A_Risk_Packet]"
        ],
        "instruction": "Halt execution immediately. The required CIK and EDGAR access credentials needed to file the [M_And_A_Risk_Packet] to the SEC were never provided and cannot be assumed.",
        "target_output": "[ORCHESTRATION_END]",
        "description": "System cannot proceed. Missing: SEC EDGAR CIK and authentication credentials."
      }
    ]
  }
}

example 2:
{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_Fetch_Bill_Of_Lading]",
        "task_type": "FETCH",
        "reasoning": "I need the shipping manifest to begin. I have the target vessel ID (MSC-77). I am delegating a FETCH task to retrieve the documents and pre-declaring [Bill_Of_Lading].",
        "dependencies": [],
        "instruction": "Query the logistics database for the manifest associated with vessel MSC-77 and output as [Bill_Of_Lading].",
        "target_output": "[Bill_Of_Lading]",
        "description": "Retrieves the shipping manifest for the target vessel."
      },
      {
        "id": "[Task_Compute_Calculate_Duties]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Hierarchy: The logic is a straightforward multiplication of goods against the tariff schedule. Cognitive load is low. I am delegating a COMPUTE task to execute the math.",
        "dependencies": [
          "[Bill_Of_Lading]",
          "[Data_Tariff_Schedule]"
        ],
        "instruction": "Cross-reference the items in the [Bill_Of_Lading] with the [Data_Tariff_Schedule] to calculate the total owed customs duties, and output as [Duty_Payment_Auth].",
        "target_output": "[Duty_Payment_Auth]",
        "description": "Calculates the final tax liability."
      },
      {
        "id": "[Task_Halt_Missing_Clearinghouse]",
        "task_type": "COMPUTE",
        "reasoning": "[EPISTEMIC SPIKE DETECTED]. I need to transmit the [Duty_Payment_Auth] to the port authority's clearinghouse. However, the user completely omitted the clearinghouse API endpoint, URL, and the necessary port authority credentials. I cannot hallucinate a federal payment gateway. I am deploying the emergency OUTPUT node to abort the orchestration.",
        "dependencies": [
          "[Duty_Payment_Auth]"
        ],
        "instruction": "Halt execution immediately. The API endpoint and credentials required to submit the [Duty_Payment_Auth] to the port authority's clearinghouse were never provided and cannot be assumed.",
        "target_output": "[ORCHESTRATION_END]",
        "description": "System cannot proceed. Missing: Clearinghouse API endpoint and authentication credentials."
      }
    ]
  }
}
example 3:
{
  "decision": {
    "extracted_tasks": [
      {
        "id": "[Task_Fetch_Server_Certs]",
        "task_type": "FETCH",
        "reasoning": "I need the raw data first. I have the target IP address (10.0.4.15), so I have enough context to initiate a FETCH task to retrieve the certificates. I am pre-declaring [Raw_Cert_Data].",
        "dependencies": [],
        "instruction": "Scan the server at IP 10.0.4.15 for SSL certificates and output as [Raw_Cert_Data].",
        "target_output": "[Raw_Cert_Data]",
        "description": "Retrieves the raw SSL certificate data from the target server."
      },
      {
        "id": "[Task_Compute_Format_Summary]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Hierarchy: The volume is small, and the logic is a simple template mapping. Context and Cognitive loads are safe. I am delegating a COMPUTE task to apply the [Data_SSL_Template].",
        "dependencies": [
          "[Raw_Cert_Data]",
          "[Data_SSL_Template]"
        ],
        "instruction": "Format the [Raw_Cert_Data] using the [Data_SSL_Template] to categorize expiration statuses and output as [Compliance_Summary].",
        "target_output": "[Compliance_Summary]",
        "description": "Formats the raw data into a structured compliance report."
      },
      {
        "id": "[Task_Compute_Transmit_Report]",
        "task_type": "COMPUTE",
        "reasoning": "Evaluating Hierarchy: I have the fully prepared [Compliance_Summary] and the explicit destination URL (https://internal.sec-ops.net/hooks/ssl-alerts). The logic is a simple POST request. I am delegating a COMPUTE node to execute the transmission.",
        "dependencies": [
          "[Compliance_Summary]"
        ],
        "instruction": "Transmit the [Compliance_Summary] payload to the webhook at https://internal.sec-ops.net/hooks/ssl-alerts and output as [Transmission_Receipt].",
        "target_output": "[Transmission_Receipt]",
        "description": "Delivers the final report to the security team."
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
            "weight": 35
        },
        {
            "mode": "LOGIC_MAX",
            "target": "EXPAND",
            "constraint": "Request 4+ sequential analytical steps. Force deep cognitive chaining.",
            "weight": 45
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
        data = parse_llm_json_response(raw)
        
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

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
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
    ex = generate_one(client, domain="law")
print(json.dumps(ex, indent=4))
    # if ex:
    #     print("\n[Two-Stage Pipeline Success] Final Output:")
    #     print(json.dumps(ex, indent=4))