"""
MIPROv2 Production Live Chart Tracker.

Hooks into MIPROv2 compilation to display a live updating matplotlib dashboard showing:
- Overall normalized_final score per step
- Per-component breakdown: structural, per_task, semantic, coverage
- Rolling average trend line
"""

import json
import threading
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# ============================================================================
# DATA STORE — Thread-safe score buffer
# ============================================================================

@dataclass
class ScoreBuffer:
    """Thread-safe buffer for MIPROv2 step scores."""
    steps:             list[int]   = field(default_factory=list)
    normalized_finals: list[float] = field(default_factory=list)
    structural:        list[float] = field(default_factory=list)
    per_task:          list[float] = field(default_factory=list)
    semantic:          list[float] = field(default_factory=list)
    coverage:          list[float] = field(default_factory=list)
    _lock: threading.Lock          = field(default_factory=threading.Lock)

    def record(
        self, step: int, normalized_final: float, structural_score: float = 0.0,
        per_task_score: float = 0.0, semantic_score: float = 0.0, coverage_score: float = 0.0,
    ) -> None:
        with self._lock:
            self.steps.append(step)
            self.normalized_finals.append(normalized_final)
            self.structural.append(structural_score)
            self.per_task.append(per_task_score)
            self.semantic.append(semantic_score)
            self.coverage.append(coverage_score)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "steps": list(self.steps), "normalized_finals": list(self.normalized_finals),
                "structural": list(self.structural), "per_task": list(self.per_task),
                "semantic": list(self.semantic), "coverage": list(self.coverage),
            }

    def save(self, path: str = "miprov2_scores.json") -> None:
        with self._lock:
            Path(path).write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

# Global buffer
BUFFER = ScoreBuffer()

# ============================================================================
# METRIC WRAPPER — Intercepts MIPROv2 Evaluations
# ============================================================================

def make_tracked_metric(base_metric):
    """Wraps your expander_metric to record deep scores into the LIVE BUFFER."""
    step_counter = [0]

    def tracked_metric(example, prediction, trace=None):
        score = base_metric(example, prediction, trace)

        # Extract breakdown if pipeline returned it as side-effects
        structural = getattr(prediction, "_structural_score", 0.0)
        per_task   = getattr(prediction, "_per_task_score",   0.0)
        semantic   = getattr(prediction, "_semantic_score",   0.0)
        coverage   = getattr(prediction, "_coverage_score",   0.0)

        BUFFER.record(
            step=step_counter[0], normalized_final=score, structural_score=structural,
            per_task_score=per_task, semantic_score=semantic, coverage_score=coverage,
        )
        step_counter[0] += 1
        return score

    return tracked_metric

# ============================================================================
# LIVE CHART DASHBOARD
# ============================================================================

def rolling_avg(data: list[float], window: int = 5) -> list[float]:
    if len(data) < 2: return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result

def launch_live_chart(update_interval_ms: int = 2000) -> FuncAnimation:
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_main   = fig.add_subplot(gs[0, :])   # Top full-width: Main score
    ax_struct = fig.add_subplot(gs[1, 0])   # Bottom left: Structural
    ax_task   = fig.add_subplot(gs[1, 1])   # Bottom right: Per-Task

    def update(frame):
        snap = BUFFER.snapshot()
        steps = snap["steps"]
        if not steps: return

        finals, struct, per_task = snap["normalized_finals"], snap["structural"], snap["per_task"]
        trend = rolling_avg(finals, window=5)

        # ── Main Score ──
        ax_main.cla()
        ax_main.set_title("Normalized Final Score per Evaluation", fontsize=11)
        ax_main.set_ylim(-0.05, 1.05)
        ax_main.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax_main.plot(steps, finals, color="#4C9BE8", alpha=0.5, marker="o", markersize=3)
        ax_main.plot(steps, trend, color="#E84C4C", linewidth=2.0, label="Rolling avg (w=5)")
        ax_main.legend(loc="upper left")
        ax_main.grid(True, alpha=0.2)

        # ── Structural Score ──
        ax_struct.cla()
        ax_struct.set_title("Structural Penalty Accuracy", fontsize=10)
        ax_struct.set_ylim(-0.05, 1.05)
        ax_struct.plot(steps, struct, color="#4CE87A", linewidth=1.5, marker=".")
        ax_struct.grid(True, alpha=0.2)

        # ── Per-Task Score ──
        ax_task.cla()
        ax_task.set_title("Per-Task Fidelity", fontsize=10)
        ax_task.set_ylim(-1.6, 1.1)
        ax_task.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax_task.plot(steps, per_task, color="#E8C34C", linewidth=1.5, marker=".")
        ax_task.grid(True, alpha=0.2)

        best, avg = max(finals), sum(finals) / len(finals)
        fig.suptitle(f"MIPROv2 Production Run — Evals: {len(steps)} | Best: {best:.4f} | Avg: {avg:.4f}", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    anim = FuncAnimation(fig, update, interval=update_interval_ms, cache_frame_data=False)
    plt.show(block=False)
    plt.pause(0.1)
    return anim

# ============================================================================
# COMPILATION THREAD WRAPPER
# ============================================================================

def compile_with_live_chart(optimizer, module, trainset):
    """Executes MIPROv2 compilation with deep production parameters."""
    anim = launch_live_chart(update_interval_ms=2000)
    print("\n[Chart] Live Dashboard launched. Starting MIPROv2 Production Run...")

    result_box, error_box = [None], [None]

    def run_compile():
        try:
            result_box[0] = optimizer.compile(
                module,
                trainset=trainset,
                # DELETED auto="medium" from here
                provide_traceback=True,
                minibatch=False,           # KEEP FALSE: Dataset is under 100 items (32)
            )
        except Exception as e:
            error_box[0] = e

    compile_thread = threading.Thread(target=run_compile, daemon=True)
    compile_thread.start()

    while compile_thread.is_alive():
        plt.pause(2.0)

    compile_thread.join()
    
    if error_box[0]:
        raise error_box[0]

    BUFFER.save("miprov2_production_scores.json")
    print("\n[Chart] Compilation complete. Close the chart window to save the optimized model.")
    plt.show(block=True) 

    return result_box[0]