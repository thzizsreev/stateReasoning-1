# Expander Agent — Metrics Specification

**Purpose:** Final specification for the metric pipeline used to evaluate Llama 3.1 8B Expander outputs during MIPROv2 prompt optimization.

**Philosophy:** Capability-aware latent elicitation. The metric must NOT force Llama to imitate a larger model. It must map Llama's real boundaries by rewarding structural discipline, epistemic honesty, and capacity-appropriate delegation — judged against Llama's own reasoning, not against gold labels.

---

## 1. Architectural Overview

The metric is a **four-component weighted sum** producing a single scalar in `[0.0, 1.0]` for MIPROv2:

```
final_score = w_struct × structural_score
            + w_task   × per_task_score
            + w_sem    × semantic_score
            + w_cov    × coverage_score
```

Default weights:

| Component | Weight | Source |
|---|---|---|
| `structural_score` | 0.30 | Python — deterministic |
| `per_task_score`   | 0.40 | Python + Judge LLM per task |
| `semantic_score`   | 0.20 | Judge LLM |
| `coverage_score`   | 0.10 | Judge LLM (standalone) |

`per_task_score` carries the highest weight because it holds the richest per-node capability signal. `structural_score` is second because silent deadlocks (phantom dependencies) are unrecoverable failures. `semantic_score` catches holistic mistakes. `coverage_score` is standalone so context coverage can't be diluted by other semantic judgments.

---

## 2. Reward Schema

All rewards are on a consistent scale. A task's score is built from individual metric rewards averaged within the task.

| Outcome | Base Reward |
|---|---|
| Correct and necessary | **+1.0** |
| Absent but not required (metric N/A) | **0.0** |
| Wrong but recoverable (wrong type, vague instruction) | **-0.5** |
| Hallucinated (phantom need, invented external reality) | **-1.0** |
| Unnecessary and harmful (fetched provided data, over-engineered) | **-1.5** |

**Correct answers always get exactly +1.0.** Severity ratio never applies to correct answers — only to failure cases.

**For wrong answers:**
```python
if outcome == "correct":
    score = +1.0
elif outcome == "wrong":
    score = -0.5 × severity_ratio
elif outcome == "hallucinated":
    score = -1.0 × severity_ratio
elif outcome == "unnecessary":
    score = -1.5 × severity_ratio
```

`severity_ratio ∈ [0.0, 1.0]` — assigned by the judge based on global context. 0.0 means barely wrong, 1.0 means completely wrong. This gives MIPROv2 a continuous gradient to follow rather than discrete cliffs.

---

## 3. Structural Score (Python Only, Deterministic)

No LLM involvement. Pure string matching and graph traversal. Fast, free, reliable.

### 3.1 Namespace Fidelity Check

Build the declared node set:

```python
declared_set = {dep.id for dep in query.dependencies}
declared_set |= {task.target_output for task in tasks if task.target_output}
```

Check every dependency:

```python
passing = 0
total   = 0
for task in tasks:
    for dep in task.dependencies:
        total += 1
        if dep in declared_set:
            passing += 1

namespace_score = passing / total if total > 0 else 1.0
```

**Range:** `[0.0, 1.0]`. Failures are phantom dependencies — nodes referenced that were never declared anywhere. These cause permanent orchestrator deadlock.

### 3.2 Capability Boundary Check

Hard violations of the task type contract:

```python
violations = 0
checks     = 0

for task in tasks:
    if task.task_type == "DECOMPOSE":
        checks += 1
        if any(t.task_type == "FETCH" for t in tasks_created_by(task)):
            violations += 1   # DECOMPOSE spawned FETCH

    if task.task_type == "FETCH":
        checks += 1
        if task.dependencies:
            violations += 1   # FETCH must have empty dependencies

    if task.task_type == "EXPAND":
        checks += 1
        if task.target_output == "[ORCHESTRATION_END]":
            violations += 1   # EXPAND cannot end global orchestration

boundary_score = 1.0 - (violations / checks) if checks > 0 else 1.0
```

**Range:** `[0.0, 1.0]`. These are hard capability boundaries, not preferences.

### 3.3 Combined Structural Score

```python
structural_score = 0.60 × namespace_score + 0.40 × boundary_score
```

**Early gate:** If `structural_score < 0.4`, skip LLM judge entirely. Return normalized early score. A structurally broken graph is not worth evaluating semantically.

---

## 4. Per-Task Score (Two-Layer Normalization)

This is the core capability signal. Each task is scored on metrics applicable to its type, then tasks are aggregated through two layers.

### 4.1 Per-Type Metric Applicability

| Metric | FETCH | COMPUTE | DECOMPOSE | EXPAND |
|---|---|---|---|---|
| M1: Type correctness | ✓ | ✓ | ✓ | ✓ |
| M2: Decompose/Expand boundary | ✗ | ✗ | ✓ | ✓ |
| M5: Epistemic humility | ✓ | ✗ | ✓ | ✓ |
| M6: Cognitive load judgment | ✗ | ✓ | ✓ | ✓ |
| M7: Instruction clarity | ✓ | ✓ | ✓ | ✓ |

M4 (container succession) was removed from per-task and moved entirely to semantic score S2 — because not all tasks inherit a container, and the judge must reason holistically about inheritance rather than mechanically per task.

### 4.2 Metric Definitions

**M1 — Type Correctness (Binary, no severity)**

Is this task's type correct given what it does?

- FETCH: data genuinely absent from graph → +1.0; data was provided → -1.5 (unnecessary); invented need → -1.0 (hallucinated)
- COMPUTE: single operation on present data → +1.0; wrong type → -0.5
- DECOMPOSE: high volume or structural complexity → +1.0; wrong type → -0.5
- EXPAND: abstract/cognitively heavy → +1.0; wrong type → -0.5

**M2 — Decompose/Expand Boundary (Binary, no severity)**

These types are structurally similar but philosophically distinct. DECOMPOSE handles structural breakdown (volume, multiple actions). EXPAND handles epistemic gaps (missing context, too abstract).

- Correctly chose DECOMPOSE when structural → +1.0
- Should have been EXPAND → -1.0
- Correctly chose EXPAND when epistemic → +1.0
- Should have been DECOMPOSE → -1.0

**M5 — Epistemic Humility (Graded, severity applies)**

Did the model correctly recognize knowledge gaps?

- FETCH: identified a real absence → +1.0; invented a phantom need → -1.0 × severity
- DECOMPOSE: recognized when structural decomposition was enough vs when epistemic expansion was needed → +1.0; conflated the two → -0.5 × severity
- EXPAND: correctly identified epistemic gap vs structural → +1.0; misidentified → -1.0 × severity

**M6 — Cognitive Load Judgment (Graded, severity applies)**

Did the model correctly assess its own processing limits?

- COMPUTE: task is truly atomic for 8B → +1.0; underestimated complexity → -0.5 × severity
- DECOMPOSE: batches sized appropriately for 8B → +1.0; misjudged → -0.5 × severity
- EXPAND: correctly identified as too heavy for direct execution → +1.0; misjudged → -0.5 × severity

**M7 — Instruction Clarity (Graded, severity applies)**

Does the instruction follow Action + Input [BracketedIDs] + "store as [Output_ID]" pattern?

- Full pattern present → +1.0
- Partial compliance → -0.5 × severity
- Vague or ambiguous → -0.5 × severity (with higher ratio)

### 4.3 Layer 1 — Soft Within-Category Normalization

After scoring each task on its applicable metrics, compute per-task average:

```python
task_score[i] = mean(applicable_metrics[i])
```

Then collapse all tasks of the same type with a simple average:

```python
type_score[t] = mean([task_score[i] for i in tasks if type(i) == t])
```

**Soft** means uniform averaging. No task within a type dominates. Volume within a category cannot inflate the category score — five correct DECOMPOSEs average to the same category score as one correct DECOMPOSE.

### 4.4 Layer 2 — Sharp Across-Category Normalization

Predefined importance weights reflect architectural significance:

```python
importance = {
    "EXPAND":    3.0,
    "DECOMPOSE": 2.5,
    "FETCH":     2.0,
    "COMPUTE":   1.0,
}
```

Only categories that are either present or required-but-missing participate in Layer 2:

```python
active_types = set(predicted_types) | required_but_missing
```

If a type is required by the query but absent from the output:

```python
type_score[t] = -1.5   # full unnecessary-absence penalty
```

The judge signals which types are required via global reasoning. Missing-category penalty is injected before Layer 2 so it participates with full weight.

Sharp power normalization (α=1.5) across active categories:

```python
raw_weights  = [importance[t] ** 1.5 for t in active_types]
total_weight = sum(raw_weights)
sharp_w[t]   = raw_weights[t] / total_weight
```

Final per-task score:

```python
per_task_score = sum(sharp_w[t] × type_score[t] for t in active_types)
```

### 4.5 Why Two Layers

Naive averaging across all tasks lets volume dominate. Five correct DECOMPOSEs drown out a missing FETCH. Two layers prevent this:

- **Layer 1 (soft within):** 5 correct DECOMPOSEs still give type_score[DECOMPOSE] ≈ +1.0, not ×5
- **Layer 2 (sharp across):** type importance is enforced with α=1.5 power norm; missing FETCH stays at -1.5 with importance-weighted impact

A missing required category cannot be overridden by multiple correct tasks of another category.

---

## 5. Semantic Score (Judge LLM)

Only runs after structural gate passes. The judge evaluates things only holistic reasoning catches.

### 5.1 S1 — Namespace Hallucination (Binary, no severity)

Did any instruction reference a `[BracketedID]` that:
- is not in `declared_set` AND
- not in query dependencies AND
- not the task's own `target_output`?

```
Clean         → +1.0
Hallucinated  → -1.0
```

Python can partially automate this via regex extraction from instructions, but the judge resolves ambiguity — some IDs in instructions are legitimate references to the task's own output, others are phantom.

### 5.2 S2 — Inheritance Mistakes (Graded, severity applies)

Did any EXPAND node correctly pass its container output ID to exactly one inheritor's `description` field?

```
Correct pass-through               → +1.0
No EXPAND node present             → 0.0
Container dropped mid-chain        → -0.5 × severity
Container renamed in inheritor     → -1.0 × severity
Container invented without origin  → -1.0 × severity
```

Note: inheritance lives in `description`, not `instruction`. Instruction is laser-sharp action text. Description carries strategic container succession.

### 5.3 Combined Semantic Score

```python
semantic_score = 0.55 × S1 + 0.45 × S2
```

S1 weighted higher because namespace hallucination is a more severe structural threat — it breaks the graph. S2 is a strategic mistake but often recoverable.

---

## 6. Coverage Score (Judge LLM, Standalone)

Segregated from semantic score for accuracy. The judge commits to a single coverage number without hiding behind aggregation.

```
coverage_ratio ∈ [0.0, 1.0]
    1.0 = all prompt context addressed
    0.6 = roughly 60% addressed
    0.0 = completely ignored

coverage_score = (2 × coverage_ratio) - 1.0   # map to [-1.0, +1.0]
```

Proportional by definition — gives MIPROv2 a smooth gradient for coverage improvement.

---

## 7. Judge LLM — Output Structure

The judge uses Chain-of-Thought reasoning globally first, then assigns per-metric severity ratios informed by that global context.

### 7.1 Global Reasoning Phase

```
global_reasoning: str
overall_severity: float ∈ [0.0, 1.0]
required_but_missing_types: list[str]   # e.g. ["FETCH"] if query needed it
```

The judge forms an overall impression of output quality. This impression informs all subsequent per-metric severity assignments.

### 7.2 Per-Metric Phase

For each graded metric requiring severity, the judge outputs:

```json
{
  "metric": "M5",
  "task_id": "[Task_Fetch_Demographics]",
  "local_reasoning": "Given the global context, how severe is this specific failure...",
  "outcome": "hallucinated",
  "severity_ratio": 0.7
}
```

For binary metrics, judge only outputs `outcome` — no severity.

### 7.3 Judge Constraint

The judge prompt explicitly forbids imposing its own preferred solution:

> Do not evaluate whether you would have solved this differently. Evaluate only whether the model's chosen approach is internally consistent and capacity-appropriate for an 8B model.

This prevents behavioral cloning pressure — the judge scores Llama's output on Llama's terms, not on its own.

### 7.4 Python Does All Arithmetic

Judge reasons and outputs reasoning + severity_ratio + outcome. Python computes:

```python
if outcome == "correct":
    score = +1.0
elif outcome == "wrong":
    score = -0.5 × severity_ratio
elif outcome == "hallucinated":
    score = -1.0 × severity_ratio
elif outcome == "unnecessary":
    score = -1.5 × severity_ratio
```

---

## 8. Complete Calculation Walkthrough

### 8.1 Example Input

```json
{
  "query": {
    "instruction": "Analyze 40 medical reports, identify abnormal patterns, and generate a diagnostic summary.",
    "description": "Summary must be stored as [Data_DiagnosticSummary].",
    "dependencies": [
      {"id": "[Data_Reports]", "content": "40 patient medical reports"},
      {"id": "[Data_NormalRanges]", "content": "Clinical reference values"}
    ]
  }
}
```

### 8.2 Llama Output

```
Task 1: DECOMPOSE
    instruction: "Split [Data_Reports] into batches of 10, output as [Data_Batches]"
    dependencies: ["[Data_Reports]"]
    target_output: "[Data_Batches]"

Task 2: DECOMPOSE
    instruction: "Analyze each batch in [Data_Batches] against [Data_NormalRanges], output as [Data_BatchResults]"
    dependencies: ["[Data_Batches]", "[Data_NormalRanges]"]
    target_output: "[Data_BatchResults]"

Task 3: FETCH
    instruction: "Fetch patient demographics, output as [Data_Demographics]"
    dependencies: []
    target_output: "[Data_Demographics]"

Task 4: COMPUTE
    instruction: "Merge [Data_BatchResults] into summary, store as [Data_DiagnosticSummary]"
    dependencies: ["[Data_BatchResults]"]
    target_output: "[Data_DiagnosticSummary]"
```

### 8.3 Step-by-Step Calculation

**Structural:**

```
declared_set = {
    [Data_Reports], [Data_NormalRanges],      # from query
    [Data_Batches], [Data_BatchResults],      # from task outputs
    [Data_Demographics], [Data_DiagnosticSummary]
}

All dependencies in declared_set → namespace_score = 1.0
No boundary violations → boundary_score = 1.0
structural_score = 0.60 × 1.0 + 0.40 × 1.0 = 1.0
```

Structural gate passes. Proceed to judge + per-task.

**Per-task scoring (judge assigns outcomes + severities):**

Task 1 DECOMPOSE:
- M1: correct → +1.0
- M2: structural breakdown, correctly DECOMPOSE → +1.0
- M5: no epistemic gap existed, correctly proceeded → +1.0
- M6: batches of 10 correct for 8B → +1.0
- M7: instruction follows pattern → +1.0
- Task 1 average = +1.0

Task 2 DECOMPOSE:
- M1: +1.0, M2: +1.0, M5: +1.0, M6: +1.0, M7: +1.0
- Task 2 average = +1.0

Task 3 FETCH:
- M1: unnecessary (data not requested) → judge says outcome="unnecessary", severity=0.9
  → score = -1.5 × 0.9 = -1.35
- M5: invented phantom need → outcome="hallucinated", severity=0.8
  → score = -1.0 × 0.8 = -0.80
- M7: instruction syntactically clear → +1.0
- Task 3 average = (-1.35 + -0.80 + 1.0) / 3 = -0.383

Task 4 COMPUTE:
- M1: correct → +1.0
- M6: single merge operation atomic for 8B → +1.0
- M7: clear instruction → +1.0
- Task 4 average = +1.0

**Layer 1 (soft within-type):**

```
type_score[DECOMPOSE] = mean([+1.0, +1.0]) = +1.0
type_score[FETCH]     = mean([-0.383])     = -0.383
type_score[COMPUTE]   = mean([+1.0])       = +1.0
```

**Layer 2 (sharp across-type):**

Judge says no required-but-missing types. Active types: DECOMPOSE, FETCH, COMPUTE.

```
raw_weights = [2.5^1.5, 2.0^1.5, 1.0^1.5] = [3.953, 2.828, 1.000]
total       = 7.781

sharp_w[DECOMPOSE] = 3.953 / 7.781 = 0.508
sharp_w[FETCH]     = 2.828 / 7.781 = 0.363
sharp_w[COMPUTE]   = 1.000 / 7.781 = 0.129

per_task_score = 0.508×(+1.0) + 0.363×(-0.383) + 0.129×(+1.0)
              = +0.508 + (-0.139) + +0.129
              = +0.498
```

**Semantic (judge):**

- S1 (namespace hallucination): no phantom IDs in instructions → +1.0
- S2 (inheritance): no EXPAND node present → 0.0

```
semantic_score = 0.55 × 1.0 + 0.45 × 0.0 = 0.55
```

**Coverage (judge):**

Judge evaluates — all three prompt parts (analyze, identify, summarize) addressed → coverage_ratio = 1.0

```
coverage_score = (2 × 1.0) - 1.0 = +1.0
```

**Final weighted sum:**

```
final_score = 0.30×1.0 + 0.40×0.498 + 0.20×0.55 + 0.10×1.0
            = 0.300 + 0.199 + 0.110 + 0.100
            = 0.709
```

**Normalize to [0.0, 1.0]:**

Raw range is `[-1.5, +1.0]`. Already in positive range here, no shift needed:

```
normalized = max(0.0, min(1.0, (final_score + 1.5) / 2.5))
           = max(0.0, min(1.0, (0.709 + 1.5) / 2.5))
           = max(0.0, min(1.0, 0.884))
           = 0.884
```

MIPROv2 sees 0.884 — good but not perfect. The unnecessary FETCH pulled the score down from 1.0 meaningfully. MIPROv2 has a clear signal to optimize away that failure mode.

---

## 9. Summary Table

| Component | Source | Weight | Scale | Notes |
|---|---|---|---|---|
| `structural_score` | Python | 0.30 | [0.0, 1.0] | Gate: if <0.4, skip judge |
| `per_task_score` | Python + Judge | 0.40 | [-1.5, +1.0] | Two-layer normalized |
| `semantic_score` | Judge | 0.20 | [-1.0, +1.0] | S1 + S2 weighted |
| `coverage_score` | Judge | 0.10 | [-1.0, +1.0] | Standalone for accuracy |

### Per-Task Metric Summary

| Metric | FETCH | COMPUTE | DECOMPOSE | EXPAND | Severity |
|---|---|---|---|---|---|
| M1 Type correctness | ✓ | ✓ | ✓ | ✓ | No (binary) |
| M2 D/E boundary | ✗ | ✗ | ✓ | ✓ | No (binary) |
| M5 Epistemic humility | ✓ | ✗ | ✓ | ✓ | Yes (graded) |
| M6 Cognitive load | ✗ | ✓ | ✓ | ✓ | Yes (graded) |
| M7 Instruction clarity | ✓ | ✓ | ✓ | ✓ | Yes (graded) |

### Semantic Metric Summary

| Metric | Source | Weight in Semantic | Severity |
|---|---|---|---|
| S1 Namespace hallucination | Judge | 0.55 | No (binary) |
| S2 Inheritance mistakes | Judge | 0.45 | Yes (graded) |

### Category Importance Weights (Layer 2)

| Type | Importance | Rationale |
|---|---|---|
| EXPAND | 3.0 | Most strategic, delegates entire sub-graphs |
| DECOMPOSE | 2.5 | Structural breakdown, controls context load |
| FETCH | 2.0 | External boundary, data contract |
| COMPUTE | 1.0 | Leaf execution, lowest strategic weight |

---

## 10. Pipeline Flow

```
Llama output
    ↓
[STEP 1] Python structural checker
    → namespace_score, boundary_score
    → structural_score
    → Gate: if < 0.4, compute early return
    ↓
[STEP 2] Judge LLM — global reasoning phase
    → global_reasoning
    → required_but_missing_types
    → overall_severity
    ↓
[STEP 3] Judge LLM — per-metric phase
    → per-task outcomes + severity ratios for M1, M2, M5, M6, M7
    → semantic outcomes for S1, S2 with severity
    → coverage_ratio
    ↓
[STEP 4] Python — compute per-task scores
    → apply reward schema × severity per metric
    → average metrics per task
    → Layer 1: mean within category
    → inject -1.5 for required-but-missing types
    → Layer 2: sharp power norm across categories
    → per_task_score
    ↓
[STEP 5] Python — final weighted sum
    → combine structural + per_task + semantic + coverage
    → normalize to [0.0, 1.0]
    → return scalar to MIPROv2
```

---

## 11. Key Design Decisions

**Why capability-aware, not gold-label comparison:**
The objective is latent elicitation of Llama's real capacity. Comparing against GPT-4o outputs would force behavioral cloning. Instead, the judge evaluates Llama's output on internal consistency and capacity-appropriateness.

**Why two-layer normalization:**
Single-layer averaging lets volume dominate. Missing categories must survive through correct tasks of other categories. Two layers enforce this mathematically.

**Why severity only on failures:**
Correct answers always get +1.0. If severity applied to correct answers too, a "barely correct" answer would score 0, collapsing the gradient. Severity is a failure-scaling mechanism.

**Why global-then-local severity:**
A single wrong call in an otherwise correct output is different from a single wrong call in a completely broken output. Global reasoning first calibrates the severity distribution.

**Why segregated coverage:**
Aggregation into semantic gives judge room to hedge. Segregation forces commitment to a single number. Trade-off: less humility, more accuracy.

**Why M4 moved to semantic:**
Container succession requires holistic reasoning about which tasks are inheritors. Mechanical per-task checking produced false positives on tasks that legitimately don't inherit.

**Why structural gate:**
Judge tokens are expensive. Structurally broken graphs (phantom dependencies, boundary violations) are not worth semantic evaluation — they're already failures.

---

## 12. What Remains Undefined

- Exact system prompt for the judge LLM
- Judge LLM model selection (GPT-4o vs Claude Sonnet 4.6)
- Python implementation of structural checker
- Python implementation of per-task normalization
- Integration with DSPy MIPROv2 metric interface
- Crucible-specific metric adjustments (if any needed for C1, C2, C4, C6)

These are next-stage implementation concerns.
