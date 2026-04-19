"""
DSPy Optimization Setup.

Because all agents are dspy.Module subclasses with typed Signatures,
the entire pipeline (or any individual agent) can be optimized with:
  - dspy.BootstrapFewShot  — generates few-shot examples from traces
  - dspy.MIPROv2           — full prompt optimization (stronger, slower)

Key design constraint preserved:
  Named prohibition labels (NO SCHEMA FITTING, ZERO HALLUCINATION) must
  survive optimization. They live in Signature docstrings + field desc=,
  which DSPy optimizers treat as immutable anchors.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from pipeline import QueryPipeline
from schema import FinalPipelinePayload


TRAINSET = [
    dspy.Example(
        user_query=(
            "The starting coordinates are [Origin_X]=0 and [Origin_Y]=0. "
            "Phase 1: Plot the [Base_Line] using [Rule_Horizontal]. "
            "Phase 2: Extrude the square to a 3D cube by applying [Rule_Z_Axis_Depth]."
        ),
    ).with_inputs("user_query"),
    dspy.Example(
        user_query=(
            "Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].\n"
            "Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D]."
        ),
    ).with_inputs("user_query"),
]


def pipeline_metric(example: dspy.Example, prediction: FinalPipelinePayload, trace=None) -> float:
    score = 1.0

    if prediction.task_nodes:
        for node in prediction.task_nodes:
            if node.task_type == "COMPUTE" and not node.target_output:
                score -= 0.2
            if not node.instruction:
                score -= 0.3

    if prediction.data_nodes:
        for node in prediction.data_nodes:
            if node.content is None:
                score -= 0.2

    return max(score, 0.0)


def optimize_with_bootstrap(pipeline: QueryPipeline) -> QueryPipeline:
    optimizer = BootstrapFewShot(
        metric=pipeline_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
    )
    return optimizer.compile(pipeline, trainset=TRAINSET)


def optimize_with_mipro(pipeline: QueryPipeline) -> QueryPipeline:
    optimizer = MIPROv2(
        metric=pipeline_metric,
        auto="medium",
    )
    return optimizer.compile(
        pipeline,
        trainset=TRAINSET,
        requires_permission_to_run=False,
    )


def save_optimized(pipeline: QueryPipeline, path: str = "optimized_pipeline.json") -> None:
    pipeline.save(path)
    print(f"  [DSPy] Optimized pipeline saved to {path}")


def load_optimized(path: str = "optimized_pipeline.json") -> QueryPipeline:
    pipeline = QueryPipeline()
    pipeline.load(path)
    print(f"  [DSPy] Loaded optimized pipeline from {path}")
    return pipeline


if __name__ == "__main__":
    from config import configure_lm

    configure_lm(backend="openrouter")
    pipeline = QueryPipeline()
    print("[Optimizer] Running BootstrapFewShot...")
    optimized = optimize_with_bootstrap(pipeline)
    save_optimized(optimized)
    print("[Optimizer] Done.")

