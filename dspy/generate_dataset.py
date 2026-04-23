import json
import sys
from pathlib import Path

# Ensure the project directory is in the path
_RUNNER_DIR = Path(__file__).resolve().parent
if str(_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNER_DIR))

# Import the necessary functions from your compile.py
from compile import build_claude_client, build_trainset, DATASET_SIZE
from judge.judge_pipeline import run_stage_2, run_stage_3, run_stage_3s, run_structural_check
from judge.judge_schemas import PipelineInput, QueryInput, DependencyNode, TaskNode, GlobalReasoningOutput
from tester_for_pipeline import run_test_suite_stage4
# Configuration
OUTPUT_FILE = "./dataset/expander_trainset.json"
TEST_SIZE = 4  # Use a smaller number like 10 for rapid testing, or DATASET_SIZE for the full run

def save_dataset(trainset: list, filepath: str) -> None:
    """Save DSPy examples as JSON (not JSON-serialized strings)."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    serialized_data = []
    for ex in trainset:
        try:
            explicit_target = json.loads(ex.explicit_target) if isinstance(ex.explicit_target, str) else ex.explicit_target
        except Exception:
            explicit_target = ex.explicit_target
        serialized_data.append({
            "explicit_target": explicit_target,
            "domain": ex.domain,
            "blueprint_mode": ex.blueprint_mode
        })
        
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2)
        
    print(f"\n[Save] Successfully saved {len(trainset)} examples to {filepath}")

def main():
    print("=== Phase 1: Dataset Generation Isolation ===")
    
    # 1. Build Client
    print("[Init] Connecting to OpenRouter / Claude...")
    claude_client = build_claude_client()
    
    # 2. Generate Dataset
    print(f"[Init] Generating {TEST_SIZE} queries...")
    trainset = build_trainset(claude_client, size=TEST_SIZE)
    
    if not trainset:
        print("\n[Error] Trainset generation failed. Check your API keys and prompt placeholders.")
        sys.exit(1)
        
    # 3. Save to Disk
    save_dataset(trainset, OUTPUT_FILE)



if __name__ == "__main__":
    run_test_suite_stage4()