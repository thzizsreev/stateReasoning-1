"""
Generation Engine — Shared Utility Library.

Dataset generation uses plain OpenAI API calls — no DSPy overhead.
DSPy is used only in compile.py for MIPROv2 optimization.

Provides:
- DomainController: round-robin domain assignment
- save() / load(): JSON serialization using ExpanderDecision schema
- merge_crucibles(): flat merge of all crucible JSON files for compiler
- run_all_crucibles(): sequential runner for all crucibles
"""

from __future__ import annotations

import json
import random
import importlib
import sys
from pathlib import Path
from typing import Optional
import dspy


DOMAINS = [
    "mathematics", "physics", "law", "medicine",
    "literature", "software", "economics", "biology",
    "linguistics", "history",
]


# ============================================================================
# DOMAIN CONTROLLER
# ============================================================================

class DomainController:
    """
    Round-robin with shuffle per cycle.
    Guarantees even distribution — no domain repeats before all others appear once.
    """

    def __init__(self, domains: list[str] = DOMAINS):
        self.domains = domains.copy()
        self._cycle = []
        self._refill()

    def _refill(self):
        self._cycle = self.domains.copy()
        random.shuffle(self._cycle)

    def next(self) -> str:
        if not self._cycle:
            self._refill()
        return self._cycle.pop()

    def reset(self):
        self._cycle = []
        self._refill()


# ============================================================================
# SERIALIZATION
# ============================================================================

def save(dataset: list[dict], path: str) -> None:
    """Save raw dict dataset to JSON. Each record already has decision as dict."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"[Save] {len(dataset)} examples → {path}")


def load(path: str) -> list[dspy.Example]:
    """Load JSON and reconstruct as dspy.Examples with ExpanderDecision objects."""
    from expander_signatures import ExpanderDecision, ExpanderTaskNode

    data = json.loads(Path(path).read_text())
    examples = []
    for record in data:
        decision_raw = record["decision"]
        decision = ExpanderDecision(
            reasoning=decision_raw["reasoning"],
            extracted_tasks=[
                ExpanderTaskNode(**t) for t in decision_raw.get("extracted_tasks", [])
            ],
            halt_reason=decision_raw.get("halt_reason"),
        )
        examples.append(
            dspy.Example(
                explicit_target=record["explicit_target"],
                domain=record["domain"],
                decision=decision,
            ).with_inputs("explicit_target")
        )
    return examples


def merge_crucibles(directory: str = "./crucible_outputs") -> list[dspy.Example]:
    """Flat merge of all crucible JSON files. Returns shuffled trainset for compiler."""
    all_examples = []
    for path in sorted(Path(directory).glob("c*_*.json")):
        examples = load(str(path))
        print(f"[Merge] {path.name}: {len(examples)} examples")
        all_examples.extend(examples)
    random.shuffle(all_examples)
    print(f"[Merge] Total: {len(all_examples)} examples")
    return all_examples


# ============================================================================
# RUN ALL CRUCIBLES
# ============================================================================

CRUCIBLE_MODULES = [
    "crucible_c1", "crucible_c2", "crucible_c3",
    "crucible_c4", "crucible_c5", "crucible_c6",
    "crucible_c7", "crucible_c8", "crucible_c9",
]


def run_all_crucibles() -> None:
    """Sequentially runs all crucible generators. Each saves its own JSON file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    print("\n[Engine] Running all crucibles sequentially...\n")

    for module_name in CRUCIBLE_MODULES:
        print(f"\n{'='*60}\n[Engine] {module_name}\n{'='*60}")
        try:
            module = importlib.import_module(module_name)
            module.run()
        except Exception as e:
            print(f"  ❌ {module_name} failed: {e}")

    print("\n[Engine] All crucibles complete.")
    merge_crucibles()


if __name__ == "__main__":
    run_all_crucibles()