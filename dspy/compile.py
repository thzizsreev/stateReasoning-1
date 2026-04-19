import dspy
from dspy.teleprompt import MIPROv2
from modules import SemanticRouter
import dspy
from trainfile_set import trainset
from config import configure_lm

def router_metric(example, pred, trace=None):
    try:
        ex_query = example.user_query.strip()
        
        # GROUND TRUTH: example still uses .decision because of how we built dspy.Example in the trainset
        ex_b_data = example.decision.requires_data_extraction
        ex_b_decomp = example.decision.requires_decomposition
        ex_b_comp = example.decision.requires_computation
        
        # PREDICTIONS: pred IS the RouterDecision object! (Removed .decision)
        p_b_data = pred.requires_data_extraction
        p_b_decomp = pred.requires_decomposition
        p_b_comp = pred.requires_computation
        
        # SNIPPETS: (Removed .decision)
        p_data = (pred.data_snippet or "").strip()
        p_decomp = (pred.decomposition_snippet or "").strip()
        p_comp = (pred.computation_snippet or "").strip()

        # 2. Calculate Sub-Scores
        bools_match = (ex_b_data == p_b_data) and (ex_b_decomp == p_b_decomp) and (ex_b_comp == p_b_comp)
        
        v_data = (p_data in ex_query) if p_data else True
        v_decomp = (p_decomp in ex_query) if p_decomp else True
        v_comp = (p_comp in ex_query) if p_comp else True
        
        # 3. Calculate Final Score
        score = bools_match and v_data and v_decomp and v_comp
        
        # ==========================================
        # THE X-RAY: PRINT FAILURES TO TERMINAL
        # ==========================================
        if not score:
            print(f"\n[METRIC FAILED]")
            if not bools_match:
                print(f"  -> BOOLS FAILED:")
                print(f"     Expected: Data({ex_b_data}), Decomp({ex_b_decomp}), Comp({ex_b_comp})")
                print(f"     Got     : Data({p_b_data}), Decomp({p_b_decomp}), Comp({p_b_comp})")
            
            if not (v_data and v_decomp and v_comp):
                print(f"  -> VERBATIM FAILED:")
                if not v_data: print(f"     Data Snippet generated: '{p_data}'")
                if not v_decomp: print(f"     Decomp Snippet generated: '{p_decomp}'")
                if not v_comp: print(f"     Comp Snippet generated: '{p_comp}'")
                
        return score

    except Exception as e:
        print(f"\n[METRIC CRASHED] Syntax error: {e}")
        return False

# 2. The Compilation Engine
def compile_router():
    print("Initializing MIPROv2 Compiler... This will consume compute.")
    
    # Instantiate the uncompiled "student"
    student_router = SemanticRouter()
    
    # Configure the state-of-the-art Bayesian Optimizer
    # auto="light" is fast and cost-effective. Use "medium" if you expand trainset to 50+
    optimizer = MIPROv2(
        metric=router_metric,
        auto="light", 
        num_threads=4
    )
    
    # Execute the training loop
    print("Compiling... (Optimizing Instructions and Bootstrapping Demos)")
    optimized_router = optimizer.compile(
        student_router, 
        trainset=trainset, 
        max_bootstrapped_demos=2, # Limits token bloat in the final prompt
        max_labeled_demos=2
    )
    
    # Freeze the latent weights to disk
    file_path = "optimized_semantic_router.json"
    optimized_router.save(file_path)
    print(f"\n✅ Compilation Complete. State frozen to: {file_path}")

"----------------------------Decomposer Agent----------------------------"

def decomposer_math_metric(gold, pred, trace=None):
    """
    The Mathematical Loss Function for MIPROv2 compilation.
    Grades the LLM on completeness, terminal logic, and task volume.
    """
    score = 1.0
    
    # 1. Fatal Crash / Empty Payload (Binary Kill)
    try:
        decision = pred.decision if hasattr(pred, 'decision') else pred
        tasks = decision.extracted_tasks
        if not tasks: return 0.0
    except Exception:
        return 0.0

    # 2. Field Completeness Penalty
    # We deduct heavily if the LLM generated nodes with empty strings.
    for task in tasks:
        if not task.id or not task.instruction or not task.task_type:
            score -= 0.4  # Massive penalty for missing core data

    # 3. Terminal Logic Penalty
    # If the last step isn't OUTPUT, the engine won't stop.
    if tasks[-1].task_type != "OUTPUT":
        score -= 0.3

    # 4. Gold Standard Volume Alignment
    # How close was the generated array size to the true task count?
    try:
        gold_decision = gold.decision if hasattr(gold, 'decision') else gold
        gold_count = len(gold_decision.extracted_tasks)
        pred_count = len(tasks)
        
        # Deduct 0.1 for every node it missed or hallucinated over the gold count
        count_difference = abs(gold_count - pred_count)
        if count_difference > 0:
            score -= (0.1 * count_difference)
            
    except Exception:
        pass # Failsafe if gold example is malformed

    # 5. Optional Offline Hallucination Penalty
    # Even though we don't assert this at runtime, we heavily penalize it 
    # DURING TRAINING so the optimizer learns to preserve brackets natively.
    for task in tasks:
        for dep in task.dependencies:
            if dep not in task.instruction:
                score -= 0.2

    # Ensure score stays bounded between 0.0 and 1.0
    return max(0.0, score)

if __name__ == "__main__":
    # 1. IMPORT YOUR CONFIG (adjust the import path based on your folder structure)
    # from your_config_file import configure_lm 
    
    # 2. TURN ON THE ENGINE FIRST
    print("Booting up Language Model for Compilation...")
    configure_lm(backend="openrouter") 
    
    # 3. RUN THE COMPILER
    compile_router()