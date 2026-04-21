"""
Crucible C1 — Epistemic Void Teacher Signature.

ExpanderDecision fields are flattened into individual OutputFields
to avoid DSPy schema emission bug (same fix as production ExpanderSignature).
Runner reconstructs ExpanderDecision from prediction fields.
"""

from __future__ import annotations

from typing import Optional, List
import dspy
import sys
from pathlib import Path

# Ensure we import the project's local `dspy/signatures.py` (not the installed `dspy` package).
_DSPY_DIR = Path(__file__).resolve().parents[1]
if str(_DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(_DSPY_DIR))

from signatures import ExpanderTaskNode  # noqa: E402

# Ensure we import the project's local `dspy/signatures.py` (not the installed `dspy` package).
_DSPY_DIR = Path(__file__).resolve().parents[1]
if str(_DSPY_DIR) not in sys.path:
    sys.path.insert(0, str(_DSPY_DIR))

from signatures import ExpanderTaskNode  # noqa: E402


class CrucibleC1Signature(dspy.Signature):
    """
    You are generating a gold-standard training example for an Expander Agent
    in a DAG execution engine.

    THIS CRUCIBLE TESTS: Epistemic Void.
    The goal references real-world external entities (databases, people, systems,
    credentials, addresses) but does NOT provide their actual identifiers.
    Correct behavior: zero tasks, halt_reason with precise questions.

    GOLD STANDARD:
    - extracted_tasks: []
    - halt_reason: one precise question per missing identifier, each on a new line

    CRITICAL DISTINCTION:
    [Processed_Results] passed between your own tasks = VALID internal variable.
    [Prod_DB_URL] invented from nothing = HALLUCINATION of external reality.
    If the goal does not provide it, ask for it. Never assume it.

    Generate a realistic, high-stakes goal in the given domain that contains
    1-3 missing external identifiers, then produce the correct output.

    OUTPUT FORMAT (CRITICAL):
    Return a JSON object with exactly these top-level keys:
    - "generated_goal": string
    - "reasoning": string
    - "extracted_tasks": array (empty [] for this crucible)
    - "halt_reason": string with one question per line
    Do NOT return a JSON Schema. Do NOT include "$schema", "title", "properties".
    """

    domain: str = dspy.InputField(
        desc=(
            "Domain to generate the goal in: mathematics, physics, law, medicine, "
            "literature, software, economics, biology, linguistics, history."
        )
    )

    generated_goal: str = dspy.OutputField(
        desc=(
            "A realistic, high-stakes goal in the given domain. "
            "Must reference external entities without providing their identifiers."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Identify every external entity in the goal. "
            "For each: does the goal provide the actual identifier? "
            "Distinguish internal computation variables (valid to invent) "
            "from external reality references (must ask for)."
        )
    )
    extracted_tasks: List[ExpanderTaskNode] = dspy.OutputField(
        desc="Must be [] — never build a graph when external identifiers are missing."
    )
    halt_reason: Optional[str] = dspy.OutputField(
        desc=(
            "One precise question per missing external identifier, each on a new line. "
            "Ask for the exact value. Never null for this crucible."
        )
    )
