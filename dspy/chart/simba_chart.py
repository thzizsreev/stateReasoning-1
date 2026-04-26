"""
SIMBA Live Chart Tracker.

Hooks into DSPy callback system to capture per-step scores during compilation.
Displays live updating matplotlib chart showing:
- Overall normalized_final score per step
- Per-component breakdown: structural, per_task, semantic, coverage
- Rolling average trend line

Usage: import and attach before calling optimizer.compile()
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("TkAgg")  # works on local machines — change to "Qt5Agg" if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np


# ============================================================================
# DATA STORE — thread-safe score buffer
# ============================================================================

@dataclass
class ScoreBuffer:
    """Thread-safe buffer for SIMBA step scores."""
    steps:             list[int]   = field(default_factory=list)
    normalized_finals: list[float] = field(default_factory=list)
    structural:        list[float] = field(default_factory=list)
    per_task:          list[float] = field(default_factory=list)
    semantic:          list[float] = field(default_factory=list)
    coverage:          list[float] = field(default_factory=list)
    _lock: threading.Lock          = field(default_factory=threading.Lock)

    def record(
        self,
        step: int,
        normalized_final: float,
        structural_score: float = 0.0,
        per_task_score:   float = 0.0,
        semantic_score:   float = 0.0,
        coverage_score:   float = 0.0,
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
                "steps":             list(self.steps),
                "normalized_finals": list(self.normalized_finals),
                "structural":        list(self.structural),
                "per_task":          list(self.per_task),
                "semantic":          list(self.semantic),
                "coverage":          list(self.coverage),
            }

    def save(self, path: str = "simba_scores.json") -> None:
        with self._lock:
            Path(path).write_text(
                json.dumps(self.snapshot(), indent=2),
                encoding="utf-8",
            )


# Global buffer — shared between metric callback and chart
BUFFER = ScoreBuffer()


# ============================================================================
# METRIC WRAPPER — records scores then returns to SIMBA
# ============================================================================

def make_tracked_metric(base_metric):
    """
    Wraps your expander_metric to also record scores into BUFFER.
    Pass the returned function to SIMBA as the metric.
    """
    step_counter = [0]

    def tracked_metric(example, prediction, trace=None):
        score = base_metric(example, prediction, trace)

        # Try to extract component breakdown if pipeline returned a ScoreBreakdown
        # stored as a side-effect on the prediction object
        structural = getattr(prediction, "_structural_score", 0.0)
        per_task   = getattr(prediction, "_per_task_score",   0.0)
        semantic   = getattr(prediction, "_semantic_score",   0.0)
        coverage   = getattr(prediction, "_coverage_score",   0.0)

        BUFFER.record(
            step=step_counter[0],
            normalized_final=score,
            structural_score=structural,
            per_task_score=per_task,
            semantic_score=semantic,
            coverage_score=coverage,
        )
        step_counter[0] += 1
        BUFFER.save()
        return score

    return tracked_metric


# ============================================================================
# LIVE CHART
# ============================================================================

def rolling_avg(data: list[float], window: int = 5) -> list[float]:
    if len(data) < 2:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def launch_live_chart(update_interval_ms: int = 2000) -> FuncAnimation:
    """
    Launch live matplotlib chart.
    Returns the FuncAnimation object — keep a reference to prevent GC.

    Call this BEFORE starting compilation in a separate thread or before
    the blocking compile() call if using plt.pause() pattern.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("SIMBA Optimization — Live Score Tracker", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_main     = fig.add_subplot(gs[0, :])   # top full-width — main score
    ax_struct   = fig.add_subplot(gs[1, 0])   # bottom left — structural
    ax_task     = fig.add_subplot(gs[1, 1])   # bottom right — per_task

    # Colors
    C_MAIN     = "#4C9BE8"
    C_TREND    = "#E84C4C"
    C_STRUCT   = "#4CE87A"
    C_TASK     = "#E8C34C"
    C_SEMANTIC = "#C34CE8"
    C_COVERAGE = "#4CE8E8"

    def update(frame):
        snap = BUFFER.snapshot()
        steps = snap["steps"]

        if not steps:
            return

        finals    = snap["normalized_finals"]
        struct    = snap["structural"]
        per_task  = snap["per_task"]
        semantic  = snap["semantic"]
        coverage  = snap["coverage"]
        trend     = rolling_avg(finals, window=5)

        # ── Main score plot ──
        ax_main.cla()
        ax_main.set_title("Normalized Final Score per Step", fontsize=11)
        ax_main.set_xlabel("Evaluation Step")
        ax_main.set_ylabel("Score [0.0 → 1.0]")
        ax_main.set_ylim(-0.05, 1.05)
        ax_main.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax_main.plot(steps, finals, color=C_MAIN, alpha=0.5, linewidth=1.0,
                     marker="o", markersize=3, label="Score")
        ax_main.plot(steps, trend, color=C_TREND, linewidth=2.0,
                     linestyle="-", label=f"Rolling avg (w=5)")
        if finals:
            ax_main.annotate(
                f"Latest: {finals[-1]:.4f}",
                xy=(steps[-1], finals[-1]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, color=C_MAIN,
            )
        ax_main.legend(loc="upper left", fontsize=9)
        ax_main.grid(True, alpha=0.2)

        # ── Structural score ──
        ax_struct.cla()
        ax_struct.set_title("Structural Score", fontsize=10)
        ax_struct.set_ylim(-0.05, 1.05)
        ax_struct.set_xlabel("Step")
        ax_struct.plot(steps, struct, color=C_STRUCT, linewidth=1.5,
                       marker=".", markersize=4)
        ax_struct.grid(True, alpha=0.2)

        # ── Per-task score (can go negative) ──
        ax_task.cla()
        ax_task.set_title("Per-Task Score", fontsize=10)
        ax_task.set_ylim(-1.6, 1.1)
        ax_task.set_xlabel("Step")
        ax_task.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax_task.plot(steps, per_task, color=C_TASK, linewidth=1.5,
                     marker=".", markersize=4)
        ax_task.grid(True, alpha=0.2)

        # Stats in title
        if finals:
            best  = max(finals)
            avg   = sum(finals) / len(finals)
            fig.suptitle(
                f"SIMBA Optimization — Steps: {len(steps)} | "
                f"Best: {best:.4f} | Avg: {avg:.4f} | Latest: {finals[-1]:.4f}",
                fontsize=12, fontweight="bold"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    anim = FuncAnimation(fig, update, interval=update_interval_ms, cache_frame_data=False)
    plt.show(block=False)
    plt.pause(0.1)
    return anim


# ============================================================================
# COMPILATION THREAD WRAPPER
# ============================================================================

def compile_with_live_chart(
    optimizer,
    module,
    trainset,
    base_metric,
    update_interval_ms: int = 2000,
):
    """
    Runs SIMBA compilation while displaying a live chart.
    """
    tracked = make_tracked_metric(base_metric)
    
    # --- ADD THIS LINE ---
    # Override the optimizer's original metric with our tracked version
    optimizer.metric = tracked 
    # ---------------------

    # Launch chart before compilation blocks
    anim = launch_live_chart(update_interval_ms=update_interval_ms)

    print("[Chart] Live chart launched. Starting SIMBA compilation...")

    # Run compilation in a separate thread so chart stays responsive
    result_box = [None]
    error_box  = [None]

    def run_compile():
        try:
            result_box[0] = optimizer.compile(
                module,
                trainset=trainset,
            )
        except Exception as e:
            error_box[0] = e

    compile_thread = threading.Thread(target=run_compile, daemon=True)
    compile_thread.start()

    # Keep chart alive while compilation runs
    while compile_thread.is_alive():
        plt.pause(update_interval_ms / 1000.0)

    compile_thread.join()

    if error_box[0]:
        raise error_box[0]

    print("[Chart] Compilation complete. Chart will remain open.")
    print("[Chart] Close the chart window to continue.")

    BUFFER.save("simba_scores_final.json")
    plt.show(block=True)   # block until user closes window

    return result_box[0]