import threading
import json
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# Global thread-safe buffer
class ScoreBuffer:
    def __init__(self):
        self.steps = []
        self.finals = []
        self._lock = threading.Lock()

    def record(self, step: int, score: float):
        with self._lock:
            self.steps.append(step)
            self.finals.append(score)

    def snapshot(self):
        with self._lock:
            return list(self.steps), list(self.finals)

BUFFER = ScoreBuffer()

def make_tracked_metric(base_metric):
    """Wraps your metric to pipe scores to the Matplotlib buffer."""
    step_counter = [0]
    def tracked_metric(example, prediction, trace=None):
        score = base_metric(example, prediction, trace)
        BUFFER.record(step_counter[0], score)
        step_counter[0] += 1
        return score
    return tracked_metric

def launch_smoke_chart(update_interval_ms: int = 1000) -> FuncAnimation:
    """Launches a fast-updating chart for the smoke test."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("MIPROv2 Smoke Test — Live Tracker", fontweight="bold")
    
    C_MAIN = "#4C9BE8"

    def update(frame):
        steps, finals = BUFFER.snapshot()
        if not steps:
            return
        
        ax.cla()
        ax.set_title(f"Evaluations: {len(steps)} | Latest: {finals[-1]:.4f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Normalized Final Score")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.plot(steps, finals, color=C_MAIN, marker="o", markersize=5, linewidth=1.5)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

    anim = FuncAnimation(fig, update, interval=update_interval_ms, cache_frame_data=False)
    plt.show(block=False)
    plt.pause(0.1)
    return anim

def compile_with_smoke_chart(optimizer, module, trainset):
    """Executes MIPROv2 in a thread with production parameters."""
    
    # FIX: Make sure this calls launch_smoke_chart
    anim = launch_smoke_chart(update_interval_ms=2000) 
    print("\n[Chart] Live smoke chart launched. Starting MIPROv2 Production Compile...")

    result_box = [None]
    error_box = [None]

    def run_compile():
        try:
            result_box[0] = optimizer.compile(
                module,
                trainset=trainset,
                auto="medium",             
                provide_traceback=True,
                minibatch=False,           
            )
        except Exception as e:
            error_box[0] = e  

    compile_thread = threading.Thread(target=run_compile, daemon=True)
    compile_thread.start()

    while compile_thread.is_alive():
        plt.pause(1.0)

    compile_thread.join()

    if error_box[0]:
        raise error_box[0]

    print("[Chart] Smoke test complete. Close chart window to terminate.")
    plt.show(block=True) 

    return result_box[0]