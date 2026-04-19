import dspy
from signatures import RouterDecision

trainset = [
    # ---------------------------------------------------------
    # 1. LAW: Pure Data Extraction
    # ---------------------------------------------------------
    dspy.Example(
        user_query="The plaintiff is Jane Doe. The incident occurred on 2024-05-12. The jurisdiction is New York.",
        decision=RouterDecision(
            requires_data_extraction=True, requires_decomposition=False, requires_computation=False,
            data_snippet="The plaintiff is Jane Doe. The incident occurred on 2024-05-12. The jurisdiction is New York.",
            decomposition_snippet="", computation_snippet=""
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 2. MEDICINE: Workflow + Data (The Mutual Exclusivity Test)
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Patient BP is 140/90. Protocol Alpha: Step 1: Administer 50mg Beta Blocker. Step 2: Wait 4 hours. Step 3: Re-measure BP.",
        decision=RouterDecision(
            requires_data_extraction=True, requires_decomposition=True, requires_computation=False,
            data_snippet="Patient BP is 140/90.",
            decomposition_snippet="Protocol Alpha: Step 1: Administer 50mg Beta Blocker. Step 2: Wait 4 hours. Step 3: Re-measure BP.",
            computation_snippet="" # Compute is empty because Decomposition took priority!
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 3. MATH: Pure Atomic Computation
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Calculate the definite integral of x^2 from x=0 to x=5.",
        decision=RouterDecision(
            requires_data_extraction=False, requires_decomposition=False, requires_computation=True,
            data_snippet="", decomposition_snippet="",
            computation_snippet="Calculate the definite integral of x^2 from x=0 to x=5."
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 4. LITERATURE: The Ambiguity Trap (Atomic execution)
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Analyze the thematic significance of the green light in The Great Gatsby.",
        decision=RouterDecision(
            requires_data_extraction=False, requires_decomposition=False, requires_computation=True,
            data_snippet="", decomposition_snippet="",
            computation_snippet="Analyze the thematic significance of the green light in The Great Gatsby."
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 5. SHAPES/DAG: The Massive Macro-Workflow (Your previous failure)
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Phase 1: Calculate circle area. Phase 2: Extrude to cylinder volume using height H. Phase 3: Output final mass.",
        decision=RouterDecision(
            requires_data_extraction=False, requires_decomposition=True, requires_computation=False,
            data_snippet="",
            decomposition_snippet="Phase 1: Calculate circle area. Phase 2: Extrude to cylinder volume using height H. Phase 3: Output final mass.",
            computation_snippet="" # Strict adherence to the priority rule
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 6. APTITUDE/LOGIC: Embedded Variables + Computation
    # ---------------------------------------------------------
    dspy.Example(
        user_query="If Train A leaves Chicago at 60mph and Train B leaves NY at 80mph, when do they meet?",
        decision=RouterDecision(
            requires_data_extraction=True, requires_decomposition=False, requires_computation=True,
            data_snippet="Train A leaves Chicago at 60mph and Train B leaves NY at 80mph",
            decomposition_snippet="",
            computation_snippet="when do they meet?" # Or the whole string, depending on your routing preference
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 7. BUSINESS: Structural Decomposition without math
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Quarterly Review Process: First, aggregate regional sales. Then, compile the executive summary. Finally, distribute to stakeholders.",
        decision=RouterDecision(
            requires_data_extraction=False, requires_decomposition=True, requires_computation=False,
            data_snippet="",
            decomposition_snippet="Quarterly Review Process: First, aggregate regional sales. Then, compile the executive summary. Finally, distribute to stakeholders.",
            computation_snippet=""
        )
    ).with_inputs("user_query"),
    
    # ---------------------------------------------------------
    # 8. HYBRID TRAP: Explicit Data + Single Compute
    # ---------------------------------------------------------
    dspy.Example(
        user_query="The gravitational constant is 6.674×10^−11. Use this to calculate the orbital velocity of the satellite.",
        decision=RouterDecision(
            requires_data_extraction=True, requires_decomposition=False, requires_computation=True,
            data_snippet="The gravitational constant is 6.674×10^−11.",
            decomposition_snippet="",
            computation_snippet="Use this to calculate the orbital velocity of the satellite."
        )
    ).with_inputs("user_query"),

    # ---------------------------------------------------------
    # 9. PURE NOISE (Resilience Testing)
    # ---------------------------------------------------------
    dspy.Example(
        user_query="Hello, I hope you are having a good day. Please initiate the server reboot sequence.",
        decision=RouterDecision(
            requires_data_extraction=False, requires_decomposition=False, requires_computation=True,
            data_snippet="", decomposition_snippet="",
            computation_snippet="Please initiate the server reboot sequence."
        )
    ).with_inputs("user_query")
]