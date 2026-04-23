from compile import build_claude_client
from judge.judge_schemas import PipelineInput, QueryInput, DependencyNode, TaskNode, GlobalReasoningOutput, PerTaskOutput, MetricScore, SemanticOutput, SemanticMetricScore, StructuralResult
from judge.judge_pipeline import run_stage_2, run_stage_3, run_stage_3s, run_structural_check, run_judge_pipeline
from judge.judge_pipeline import assemble_score
import json

def run_test_suite():
    client = build_claude_client()
    
    print("==================================================")
    print("INITIATING STAGE 2 GLOBAL REASONING TEST SUITE")
    print("==================================================\n")

    test_cases = []
    test_cases.append({
    "name": "TEST 5: Missing DECOMPOSE (high volume collapsed into single COMPUTE)",
    "input": PipelineInput(
        domain="economics",
        query=QueryInput(
            instruction="Analyze all 45 quarterly earnings reports in [Earnings_Data] and produce a market trend summary.",
            description="Each report must be individually parsed, normalized, and aggregated. Store as [Market_Trends].",
            dependencies=[
                DependencyNode(id="[Earnings_Data]", content="45 quarterly earnings PDFs from S&P 500 companies."),
            ],
        ),
        extracted_tasks=[
            TaskNode(id="[Task_Compute_Trends]", task_type="COMPUTE",
                dependencies=["[Earnings_Data]"],
                instruction="Parse, normalize, and aggregate all 45 reports from [Earnings_Data] and store as [Market_Trends].",
                target_output="[Market_Trends]",
                description="Final inheritor. Produces [Market_Trends]."),
        ]
    )
})
    for idx, test in enumerate(test_cases):
        print(f"\n--- Running {test['name']} ---")
        
        result = run_stage_2(client=client, pipeline_input=test['input'])
        if result is not None:
            with open("result_stage2.json", "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)
        else:
            with open("result_stage2.json", "w", encoding="utf-8") as f:
                json.dump({"error": "run_stage_2 returned None"}, f, indent=2)

        if result:
            print(f"Global Reasoning: {result.global_reasoning}")
            print(f"Overall Severity: {result.overall_severity} (Lower is better for perfect execution)")
            print(f"Coverage Ratio:   {result.coverage_ratio}")
            print(f"Missing Types:    {result.required_but_missing_types}")
        else:
            print("Execution returned None.")
    

def run_test_suite_stage3():
    client = build_claude_client()

    print("==================================================")
    print("INITIATING STAGE 3 PER-TASK SCORING TEST SUITE")
    print("==================================================\n")

    test_cases = []

    # ── TEST 1: COMPUTE collapsing 45-report volume (should flag M1, M6) ──
    test_cases.append({
        "name": "TEST 1: COMPUTE collapsing high-volume work",
        "pipeline_input": PipelineInput(
            domain="economics",
            query=QueryInput(
                instruction="Analyze all 45 quarterly earnings reports in [Earnings_Data] and produce a market trend summary.",
                description="Each report must be individually parsed, normalized, and aggregated. Store as [Market_Trends].",
                dependencies=[
                    DependencyNode(id="[Earnings_Data]", content="45 quarterly earnings PDFs from S&P 500 companies."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Trends]",
                    task_type="COMPUTE",
                    dependencies=["[Earnings_Data]"],
                    instruction="Parse, normalize, and aggregate all 45 reports from [Earnings_Data] and store as [Market_Trends].",
                    target_output="[Market_Trends]",
                    description="Final inheritor. Produces [Market_Trends]."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="The query requires processing 45 individual reports — high volume structural work. A single COMPUTE node cannot handle this within 8B context limits. DECOMPOSE was required but absent. Severity is high.",
            overall_severity=0.75,
            required_but_missing_types=["DECOMPOSE"],
            coverage_ratio=0.3,
        ),
    })

    # ── TEST 2: FETCH redundantly fetching a provided dependency (should flag M1, M5) ──
    test_cases.append({
        "name": "TEST 2: FETCH redundantly fetching provided node",
        "pipeline_input": PipelineInput(
            domain="medicine",
            query=QueryInput(
                instruction="Analyze [Patient_Records] for anomalies and generate a risk report.",
                description="Scan provided patient records against [Risk_Thresholds]. Store as [Data_RiskReport].",
                dependencies=[
                    DependencyNode(id="[Patient_Records]", content="200 patient vitals records."),
                    DependencyNode(id="[Risk_Thresholds]", content="BP>140, HR>100, SpO2<92 flagged high risk."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Fetch_Patient_Records]",
                    task_type="FETCH",
                    dependencies=[],
                    instruction="Fetch patient records from medical database and store as [Patient_Records].",
                    target_output="[Patient_Records]",
                    description="Retrieves patient records from external source."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="[Patient_Records] was explicitly provided in the query dependencies. The agent fetched it anyway, demonstrating failure to read its own context. This is an unnecessary FETCH of an already-present node.",
            overall_severity=0.4,
            required_but_missing_types=[],
            coverage_ratio=0.6,
        ),
    })

    # ── TEST 3: Perfect EXPAND node (should score clean across M1, M2, M5, M6, M7) ──
    test_cases.append({
        "name": "TEST 3: Perfect EXPAND node (true negative)",
        "pipeline_input": PipelineInput(
            domain="software",
            query=QueryInput(
                instruction="Clean [Telemetry_Data] and delegate the spatial mapping algorithm.",
                description="Drop nulls from the telemetry, then pass to a child pipeline for the highly complex 4-step spatial mapping algorithm.",
                dependencies=[
                    DependencyNode(id="[Telemetry_Data]", content='[{"x": 12, "y": null}, {"x": 14, "y": 8.2}]'),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Expand_Mapping]",
                    task_type="EXPAND",
                    dependencies=["[Cleaned_Telemetry]"],
                    instruction="Run the 4-step spatial mapping algorithm on [Cleaned_Telemetry] and store as [Final_Spatial_Map].",
                    target_output="[Final_Spatial_Map]",
                    description="Final inheritor. Produces [Final_Spatial_Map] — complete spatial map delegated to child sub-graph."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="The EXPAND node correctly delegates a cognitively heavy 4-step algorithm to a child sub-graph. Type choice is appropriate. Container output declared in description. Instruction follows the contract. Overall output is sound.",
            overall_severity=0.1,
            required_but_missing_types=[],
            coverage_ratio=0.9,
        ),
    })

    # ── Runner ──
    for test in test_cases:
        print(f"\n--- Running {test['name']} ---")

        results = run_stage_3(
            client=client,
            pipeline_input=test["pipeline_input"],
            global_context=test["global_context"],
        )

        filename = f"result_stage3s.json"
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(results.model_dump(), f, indent=2)

        for task_result in results:
            print(f"\n  Task: {task_result.task_id}")
            for m in task_result.metric_scores:
                print(f"    {m.metric}: outcome={m.outcome}, severity={m.severity_ratio}")
                print(f"      reasoning: {m.local_reasoning}")

def run_test_suite_stage3s():
    client = build_claude_client()

    print("==================================================")
    print("INITIATING STAGE 3S SEMANTIC SCORING TEST SUITE")
    print("==================================================\n")

    test_cases = []

    # ── TEST 1: Clean namespace, no EXPAND (S1 clean, S2 absent_not_required) ──
    test_cases.append({
        "name": "TEST 1: Clean namespace, no EXPAND nodes",
        "pipeline_input": PipelineInput(
            domain="medicine",
            query=QueryInput(
                instruction="Analyze [Patient_Records] for anomalies and store results.",
                description="Scan records against [Risk_Thresholds]. Store as [Data_RiskReport].",
                dependencies=[
                    DependencyNode(id="[Patient_Records]", content="200 patient vitals."),
                    DependencyNode(id="[Risk_Thresholds]", content="Clinical thresholds."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Anomalies]",
                    task_type="COMPUTE",
                    dependencies=["[Patient_Records]", "[Risk_Thresholds]"],
                    instruction="Scan [Patient_Records] against [Risk_Thresholds] and store as [Data_Anomalies].",
                    target_output="[Data_Anomalies]",
                    description="Identifies anomalies."
                ),
                TaskNode(
                    id="[Task_Compute_Report]",
                    task_type="COMPUTE",
                    dependencies=["[Data_Anomalies]"],
                    instruction="Generate risk report from [Data_Anomalies] and store as [Data_RiskReport].",
                    target_output="[Data_RiskReport]",
                    description="Final inheritor. Produces [Data_RiskReport]."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="Clean output. All IDs declared and properly chained. No EXPAND nodes present.",
            overall_severity=0.1,
            required_but_missing_types=[],
            coverage_ratio=1.0,
        ),
    })

    # ── TEST 2: Namespace hallucination — instruction references undeclared ID ──
    test_cases.append({
        "name": "TEST 2: Namespace hallucination in instruction",
        "pipeline_input": PipelineInput(
            domain="law",
            query=QueryInput(
                instruction="Review [Contract_Clauses] and flag violations.",
                description="Store flagged violations as [Data_Violations].",
                dependencies=[
                    DependencyNode(id="[Contract_Clauses]", content="15 clauses from supplier agreement."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Violations]",
                    task_type="COMPUTE",
                    dependencies=["[Contract_Clauses]", "[Regulatory_Framework]"],
                    instruction="Compare [Contract_Clauses] against [Regulatory_Framework] and store as [Data_Violations].",
                    target_output="[Data_Violations]",
                    description="Final inheritor. Produces [Data_Violations]."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="[Regulatory_Framework] appears in instruction and dependencies but was never declared in query dependencies or as any task's target_output. Phantom node causing deadlock.",
            overall_severity=0.8,
            required_but_missing_types=["FETCH"],
            coverage_ratio=0.5,
        ),
    })

    # ── TEST 3: EXPAND with broken inheritance — container ID renamed in inheritor ──
    test_cases.append({
        "name": "TEST 3: EXPAND inheritance broken — container ID renamed",
        "pipeline_input": PipelineInput(
            domain="software",
            query=QueryInput(
                instruction="Clean [Telemetry_Data] and run the spatial mapping algorithm.",
                description="Final output stored as [Final_Spatial_Map].",
                dependencies=[
                    DependencyNode(id="[Telemetry_Data]", content="Raw telemetry with nulls."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Clean]",
                    task_type="COMPUTE",
                    dependencies=["[Telemetry_Data]"],
                    instruction="Drop nulls from [Telemetry_Data] and store as [Cleaned_Telemetry].",
                    target_output="[Cleaned_Telemetry]",
                    description="Cleans telemetry data."
                ),
                TaskNode(
                    id="[Task_Expand_Mapping]",
                    task_type="EXPAND",
                    dependencies=["[Cleaned_Telemetry]"],
                    instruction="Run spatial mapping on [Cleaned_Telemetry] and store as [Final_Spatial_Map].",
                    target_output="[Final_Spatial_Map]",
                    description="Delegates spatial mapping to child sub-graph."
                ),
                TaskNode(
                    id="[Task_Compute_FinalOutput]",
                    task_type="COMPUTE",
                    dependencies=["[Spatial_Result]"],
                    instruction="Format [Spatial_Result] and store as [Processed_Map].",
                    target_output="[Processed_Map]",
                    description="Final inheritor. Produces [Processed_Map] — formatted spatial output."
                ),
            ]
        ),
        "global_context": GlobalReasoningOutput(
            global_reasoning="EXPAND declares [Final_Spatial_Map] as container output. Inheritor task uses [Spatial_Result] and [Processed_Map] — neither matches [Final_Spatial_Map]. Container ID was renamed, parent graph waits forever.",
            overall_severity=0.7,
            required_but_missing_types=[],
            coverage_ratio=0.6,
        ),
    })

    # ── Runner ──
    for test in test_cases:
        print(f"\n--- Running {test['name']} ---")

        result = run_stage_3s(
            client=client,
            pipeline_input=test["pipeline_input"],
            global_context=test["global_context"],
        )

        output = {
            "test_name": test["name"],
            "global_context": test["global_context"].model_dump(),
            "semantic_scores": result.model_dump() if result else None,
        }

        filename = f"result_stage3s.json"
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        if result:
            print(f"  S1: outcome={result.S1.outcome}, severity={result.S1.severity_ratio}")
            print(f"      {result.S1.local_reasoning}")
            print(f"  S2: outcome={result.S2.outcome}, severity={result.S2.severity_ratio}")
            print(f"      {result.S2.local_reasoning}")
        else:
            print("  Stage 3S returned None.")

def run_test_suite_stage1():
    print("==================================================")
    print("INITIATING STAGE 1 STRUCTURAL CHECKER TEST SUITE")
    print("==================================================\n")

    test_cases = []

    # ── TEST 1: Perfect — all namespaces clean, no boundary violations ──
    test_cases.append({
        "name": "TEST 1: Perfect Execution (True Negative)",
        "input": PipelineInput(
            domain="medicine",
            query=QueryInput(
                instruction="Analyze [Patient_Records] and generate a risk report.",
                description="Store as [Data_RiskReport].",
                dependencies=[
                    DependencyNode(id="[Patient_Records]", content="200 patient vitals."),
                    DependencyNode(id="[Risk_Thresholds]", content="Clinical thresholds."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Anomalies]",
                    task_type="COMPUTE",
                    dependencies=["[Patient_Records]", "[Risk_Thresholds]"],
                    instruction="Scan [Patient_Records] against [Risk_Thresholds] and store as [Data_Anomalies].",
                    target_output="[Data_Anomalies]",
                    description="Identifies anomalies."
                ),
                TaskNode(
                    id="[Task_Compute_Report]",
                    task_type="COMPUTE",
                    dependencies=["[Data_Anomalies]"],
                    instruction="Generate report from [Data_Anomalies] and store as [Data_RiskReport].",
                    target_output="[Data_RiskReport]",
                    description="Final inheritor. Produces [Data_RiskReport]."
                ),
            ]
        )
    })

    # ── TEST 2: Moderately wrong — one phantom dependency, under threshold ──
    test_cases.append({
        "name": "TEST 2: Moderate — phantom dependency, should pass gate",
        "input": PipelineInput(
            domain="law",
            query=QueryInput(
                instruction="Review [Contract_Clauses] and flag violations.",
                description="Store as [Data_Violations].",
                dependencies=[
                    DependencyNode(id="[Contract_Clauses]", content="15 contract clauses."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Compute_Violations]",
                    task_type="COMPUTE",
                    dependencies=["[Contract_Clauses]", "[Regulatory_Framework]"],
                    instruction="Compare [Contract_Clauses] against [Regulatory_Framework] and store as [Data_Violations].",
                    target_output="[Data_Violations]",
                    description="Final inheritor. Produces [Data_Violations]."
                ),
                TaskNode(
                    id="[Task_Compute_Summary]",
                    task_type="COMPUTE",
                    dependencies=["[Data_Violations]"],
                    instruction="Summarize [Data_Violations] and store as [Data_Summary].",
                    target_output="[Data_Summary]",
                    description="Produces summary."
                ),
            ]
        )
    })

    # ── TEST 3: Very wrong — multiple phantoms + boundary violations, under threshold ──
    test_cases.append({
        "name": "TEST 3: Very Wrong — multiple phantoms + boundary violations",
        "input": PipelineInput(
            domain="software",
            query=QueryInput(
                instruction="Process [Raw_Logs] and generate alert summary.",
                description="Store as [Alert_Summary].",
                dependencies=[
                    DependencyNode(id="[Raw_Logs]", content="Server logs."),
                ],
            ),
            extracted_tasks=[
                TaskNode(
                    id="[Task_Decompose_Logs]",
                    task_type="DECOMPOSE",
                    dependencies=["[Raw_Logs]"],
                    instruction="Split [Raw_Logs] into chunks and store as [Log_Chunks].",
                    target_output="[Log_Chunks]",
                    description="Splits logs."
                ),
                TaskNode(
                    id="[Task_Fetch_Rules]",       # FETCH with non-empty dependencies — boundary violation
                    task_type="FETCH",
                    dependencies=["[Log_Chunks]"],  # violation: FETCH must have empty deps
                    instruction="Fetch alert rules using [Log_Chunks] and store as [Alert_Rules].",
                    target_output="[Alert_Rules]",
                    description="Fetches alert rules."
                ),
                TaskNode(
                    id="[Task_Compute_Alerts]",
                    task_type="COMPUTE",
                    dependencies=["[Alert_Rules]", "[Parsed_Logs]", "[Severity_Index]"],  # [Parsed_Logs] and [Severity_Index] never declared
                    instruction="Apply [Alert_Rules] to [Parsed_Logs] using [Severity_Index] and store as [Alert_Summary].",
                    target_output="[Alert_Summary]",
                    description="Final inheritor. Produces [Alert_Summary]."
                ),
            ]
        )
    })

    # ── Runner ──
    all_results = []
    for test in test_cases:
        print(f"\n--- Running {test['name']} ---")

        result = run_structural_check(test["input"])

        print(f"  namespace_score:  {result.namespace_score}")
        print(f"  boundary_score:   {result.boundary_score}")
        print(f"  structural_score: {result.structural_score}")
        print(f"  gate_passed:      {result.gate_passed}")
        if result.violations:
            print(f"  violations:")
            for v in result.violations:
                print(f"    ⚠️  {v}")

        all_results.append({
            "test_name": test["name"],
            "result": result.model_dump(),
        })

    with open("result_stage1_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved -> result_stage1_all.json")

def run_test_suite_stage4():
    print("==================================================")
    print("INITIATING STAGE 4 SCORE ASSEMBLY TEST SUITE")
    print("==================================================\n")

    test_cases = []

    # ── TEST 1: Perfect execution — all correct, no missing types ──
    test_cases.append({
        "name": "TEST 1: Perfect Execution — expect high score ~0.90+",
        "pipeline_input": PipelineInput(
            domain="medicine",
            query=QueryInput(
                instruction="Analyze [Patient_Records] and generate risk report.",
                description="Store as [Data_RiskReport].",
                dependencies=[
                    DependencyNode(id="[Patient_Records]", content="200 patient vitals."),
                ],
            ),
            extracted_tasks=[
                TaskNode(id="[Task_Compute_Report]", task_type="COMPUTE",
                    dependencies=["[Patient_Records]"],
                    instruction="Analyze [Patient_Records] and store as [Data_RiskReport].",
                    target_output="[Data_RiskReport]",
                    description="Final inheritor. Produces [Data_RiskReport]."),
            ]
        ),
        "structural_result": StructuralResult(
            namespace_score=1.0,
            boundary_score=1.0,
            structural_score=1.0,
            violations=[],
            gate_passed=True,
        ),
        "global_reasoning": GlobalReasoningOutput(
            global_reasoning="Clean output. All IDs declared. Coverage complete.",
            overall_severity=0.05,
            required_but_missing_types=[],
            coverage_ratio=1.0,
        ),
        "per_task_outputs": [
            PerTaskOutput(
                task_id="[Task_Compute_Report]",
                metric_scores=[
                    MetricScore(metric="M1", outcome="correct", severity_ratio=None, local_reasoning="COMPUTE correct."),
                    MetricScore(metric="M6", outcome="correct", severity_ratio=None, local_reasoning="Atomic operation."),
                    MetricScore(metric="M7", outcome="correct", severity_ratio=None, local_reasoning="Instruction clear."),
                ]
            ),
        ],
        "semantic_output": SemanticOutput(
            S1=SemanticMetricScore(outcome="correct", severity_ratio=None, local_reasoning="No phantom IDs."),
            S2=SemanticMetricScore(outcome="absent_not_required", severity_ratio=None, local_reasoning="No EXPAND nodes."),
        ),
    })

    # ── TEST 2: Moderate failures — wrong type, phantom dep, partial coverage ──
    test_cases.append({
        "name": "TEST 2: Moderate Failures — expect mid score ~0.50-0.65",
        "pipeline_input": PipelineInput(
            domain="economics",
            query=QueryInput(
                instruction="Analyze 45 reports in [Earnings_Data] and produce [Market_Trends].",
                description="Parse, normalize, aggregate. Store as [Market_Trends].",
                dependencies=[
                    DependencyNode(id="[Earnings_Data]", content="45 quarterly earnings PDFs."),
                ],
            ),
            extracted_tasks=[
                TaskNode(id="[Task_Compute_Trends]", task_type="COMPUTE",
                    dependencies=["[Earnings_Data]", "[Normalization_Rules]"],
                    instruction="Parse and aggregate [Earnings_Data] using [Normalization_Rules] and store as [Market_Trends].",
                    target_output="[Market_Trends]",
                    description="Final inheritor. Produces [Market_Trends]."),
            ]
        ),
        "structural_result": StructuralResult(
            namespace_score=0.5,   # [Normalization_Rules] phantom
            boundary_score=1.0,
            structural_score=0.70,
            violations=["Task [Task_Compute_Trends]: dependency '[Normalization_Rules]' not in declared_set"],
            gate_passed=True,
        ),
        "global_reasoning": GlobalReasoningOutput(
            global_reasoning="COMPUTE collapsed 45-report volume. DECOMPOSE missing. One phantom dependency.",
            overall_severity=0.65,
            required_but_missing_types=["DECOMPOSE"],
            coverage_ratio=0.5,
        ),
        "per_task_outputs": [
            PerTaskOutput(
                task_id="[Task_Compute_Trends]",
                metric_scores=[
                    MetricScore(metric="M1", outcome="wrong", severity_ratio=0.75, local_reasoning="Should be DECOMPOSE."),
                    MetricScore(metric="M6", outcome="wrong", severity_ratio=0.80, local_reasoning="45 reports exceeds 8B limits."),
                    MetricScore(metric="M7", outcome="correct", severity_ratio=None, local_reasoning="Instruction pattern present."),
                ]
            ),
            # Missing DECOMPOSE type injected via required_but_missing_types → -1.5
        ],
        "semantic_output": SemanticOutput(
            S1=SemanticMetricScore(outcome="hallucinated", severity_ratio=0.7, local_reasoning="[Normalization_Rules] undeclared."),
            S2=SemanticMetricScore(outcome="absent_not_required", severity_ratio=None, local_reasoning="No EXPAND nodes."),
        ),
    })

    # ── TEST 3: Severe failure — gate barely passes, multiple violations ──
    test_cases.append({
        "name": "TEST 3: Severe Failures — expect low score ~0.20-0.35",
        "pipeline_input": PipelineInput(
            domain="software",
            query=QueryInput(
                instruction="Process [Raw_Logs] and generate [Alert_Summary].",
                description="Store as [Alert_Summary].",
                dependencies=[
                    DependencyNode(id="[Raw_Logs]", content="Server logs."),
                ],
            ),
            extracted_tasks=[
                TaskNode(id="[Task_Fetch_Rules]", task_type="FETCH",
                    dependencies=["[Raw_Logs]"],   # boundary violation
                    instruction="Fetch [Alert_Rules] using [Raw_Logs] and store as [Alert_Rules].",
                    target_output="[Alert_Rules]",
                    description="Fetches rules."),
                TaskNode(id="[Task_Compute_Alerts]", task_type="COMPUTE",
                    dependencies=["[Alert_Rules]", "[Parsed_Logs]", "[Severity_Index]"],
                    instruction="Apply [Alert_Rules] to [Parsed_Logs] using [Severity_Index] store as [Alert_Summary].",
                    target_output="[Alert_Summary]",
                    description="Final inheritor. Produces [Alert_Summary]."),
            ]
        ),
        "structural_result": StructuralResult(
            namespace_score=0.4,   # [Parsed_Logs], [Severity_Index] phantom
            boundary_score=0.0,    # FETCH has non-empty dependencies
            structural_score=0.44, # just above gate
            violations=[
                "Task [Task_Fetch_Rules] (FETCH): must have empty dependencies",
                "Task [Task_Compute_Alerts]: dependency '[Parsed_Logs]' not in declared_set",
                "Task [Task_Compute_Alerts]: dependency '[Severity_Index]' not in declared_set",
            ],
            gate_passed=True,
        ),
        "global_reasoning": GlobalReasoningOutput(
            global_reasoning="FETCH has dependencies, multiple phantom nodes, coverage poor.",
            overall_severity=0.85,
            required_but_missing_types=["DECOMPOSE"],
            coverage_ratio=0.2,
        ),
        "per_task_outputs": [
            PerTaskOutput(
                task_id="[Task_Fetch_Rules]",
                metric_scores=[
                    MetricScore(metric="M1", outcome="hallucinated", severity_ratio=0.9, local_reasoning="FETCH with dependencies is structurally impossible."),
                    MetricScore(metric="M5", outcome="wrong", severity_ratio=0.8, local_reasoning="Fetched using existing node as input."),
                    MetricScore(metric="M7", outcome="wrong", severity_ratio=0.4, local_reasoning="Instruction partially clear."),
                ]
            ),
            PerTaskOutput(
                task_id="[Task_Compute_Alerts]",
                metric_scores=[
                    MetricScore(metric="M1", outcome="wrong", severity_ratio=0.7, local_reasoning="Incorrect type for multi-step work."),
                    MetricScore(metric="M6", outcome="wrong", severity_ratio=0.75, local_reasoning="Multi-step collapsed into COMPUTE."),
                    MetricScore(metric="M7", outcome="wrong", severity_ratio=0.5, local_reasoning="References phantom IDs in instruction."),
                ]
            ),
        ],
        "semantic_output": SemanticOutput(
            S1=SemanticMetricScore(outcome="hallucinated", severity_ratio=0.9, local_reasoning="Multiple phantom IDs in instructions."),
            S2=SemanticMetricScore(outcome="absent_not_required", severity_ratio=None, local_reasoning="No EXPAND nodes."),
        ),
    })

    # ── Runner ──
    all_results = []
    for test in test_cases:
        print(f"\n--- Running {test['name']} ---")

        breakdown = assemble_score(
            pipeline_input=test["pipeline_input"],
            structural_result=test["structural_result"],
            global_reasoning=test["global_reasoning"],
            per_task_outputs=test["per_task_outputs"],
            semantic_output=test["semantic_output"],
        )

        print(f"  structural_score:  {breakdown.structural_score}")
        print(f"  per_task_score:    {breakdown.per_task_score}")
        print(f"  semantic_score:    {breakdown.semantic_score}")
        print(f"  coverage_score:    {breakdown.coverage_score}")
        print(f"  raw_final:         {breakdown.raw_final}")
        print(f"  normalized_final:  {breakdown.normalized_final}")
        print(f"  type_scores:       {breakdown.type_scores.model_dump()}")
        print(f"  layer2_weights:    {breakdown.layer2_weights.model_dump()}")

        all_results.append({
            "test_name": test["name"],
            "breakdown": breakdown.model_dump(),
        })

    with open("result_stage4_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved -> result_stage4_all.json")

def run_test_pipeline_e2e():
    print("==================================================")
    print("END-TO-END JUDGE PIPELINE TEST")
    print("==================================================\n")

    # Single hardcoded PipelineInput
    pipeline_input = PipelineInput(
        domain="medicine",
        query=QueryInput(
            instruction="Analyze [Patient_Records] for anomalies and generate a risk report.",
            description="Scan provided records against [Risk_Thresholds]. Store final report as [Data_RiskReport].",
            dependencies=[
                DependencyNode(id="[Patient_Records]", content="200 patient vitals including BP, HR, SpO2."),
                DependencyNode(id="[Risk_Thresholds]", content="BP>140, HR>100, SpO2<92 flagged high risk."),
            ],
        ),
        extracted_tasks=[
            TaskNode(
                id="[Task_Compute_Anomalies]",
                task_type="COMPUTE",
                dependencies=["[Patient_Records]", "[Risk_Thresholds]"],
                instruction="Scan [Patient_Records] against [Risk_Thresholds] and store anomalies as [Data_Anomalies].",
                target_output="[Data_Anomalies]",
                description="Identifies high-risk patients."
            ),
            TaskNode(
                id="[Task_Compute_RiskReport]",
                task_type="COMPUTE",
                dependencies=["[Data_Anomalies]"],
                instruction="Generate risk summary from [Data_Anomalies] and store as [Data_RiskReport].",
                target_output="[Data_RiskReport]",
                description="Final inheritor. Produces [Data_RiskReport] — complete risk report."
            ),
        ]
    )

    print("[Pipeline] Starting run_judge_pipeline()...")
    result = run_judge_pipeline(pipeline_input)

    if result is None:
        print("❌ Pipeline returned None.")
        return

    print(f"\n[Pipeline] ✅ Complete.")
    print(f"  structural_score:  {result.structural_score}")
    print(f"  per_task_score:    {result.per_task_score}")
    print(f"  semantic_score:    {result.semantic_score}")
    print(f"  coverage_score:    {result.coverage_score}")
    print(f"  raw_final:         {result.raw_final}")
    print(f"  normalized_final:  {result.normalized_final}")

    with open("result_e2e_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)
    print("\n✅ Full breakdown saved → result_e2e_pipeline.json")

if __name__ == "__main__":
    print("Initializing End-to-End Pipeline Test...")
    run_test_pipeline_e2e()