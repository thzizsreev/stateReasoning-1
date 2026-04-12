from typing import Dict, Any

PROMPTS: Dict[str, Any] = {}

# ==========================================
# SYSTEM PERSONA
# ==========================================
PROMPTS["SYS_001_PERSONA"] = """You are executing a strict symbolic derivation program.
This is NOT a reasoning task. This is NOT a geometry task.

CRITICAL RULE:
You must strictly adhere to the bracketed Node IDs (e.g., [Data_A]). 
When explicitly told to save an output to a specific Node ID, you must use that exact string."""

# ==========================================
# PRIMITIVES 
# ==========================================
PROMPTS["DEF_PRIMITIVES"] = """PRIMITIVE OBJECTS:
Point(name, row, col)
Line(name, orientation, start_point, end_point)
Filled(row, col)"""

# ==========================================
# ISOLATED RULES (Skills)
# ==========================================
PROMPTS["RULE_R1"] = f"""Rule R1 (HorizontalExtension):
this is a rule that extends a horizontal line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Point P, an Integer Length L.
OUTPUT: A new Point Q = (P.row, P.col + L).}}"""

PROMPTS["RULE_R2"] = """Rule R2 (VerticalExtension):
this is a rule that extends a vertical line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Point P, an Integer Length L.
OUTPUT: A new Point Q = (P.row + L, P.col).}}"""

PROMPTS["RULE_R3"] = """Rule R3 (CornerCompletion):
this is a rule that completes a corner of a square by connecting two lines. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: Point B = (r1, c1), Point D = (r2, c2).
OUTPUT: A new Point C = (r2, c1).}}"""

PROMPTS["RULE_R4_HORIZONTAL"] = """Rule R4 (Strict Successor-Based Line Filling) - Horizontal:
this is a rule that fills a horizontal line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Horizontal Line L with start_point(r, c_start) and end_point(r, c_end).
PROCESS:
1. Derive Filled(r, c_start)
2. If c_start < c_end: Let step = +1 Else: Let step = -1
3. Let current_c = c_start
4. While current_c ≠ c_end:
      a. current_c = current_c + step
      b. Derive Filled(r, current_c)
5. Stop ONLY after Filled(r, c_end) is derived.
OUTPUT: A JSON List of all derived Filled() objects.}}"""

PROMPTS["RULE_R4_VERTICAL"] = """Rule R4 (Strict Successor-Based Line Filling) - Vertical:
this is a rule that fills a vertical line by a given length. which will be used my llm's upon calling. the 
entire instruction is as follows:
{{
INPUT: A Vertical Line L with start_point(r_start, c) and end_point(r_end, c).
PROCESS:
1. Derive Filled(r_start, c)
2. If r_start < r_end: Let step = +1 Else: Let step = -1
3. Let current_r = r_start
4. While current_r ≠ r_end:
      a. current_r = current_r + step
      b. Derive Filled(current_r, c)
5. Stop ONLY after Filled(r_end, c) is derived.
OUTPUT: A JSON List of all derived Filled() objects.}}"""

# ==========================================
# EXECUTION PHASES
# ==========================================
PROMPTS["TASK_PHASE_1"] = """PHASE 1 — POINT DERIVATION (MANDATORY)

Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE]."""


PROMPTS["TASK_PHASE_2"] = """PHASE 2 — LINE DERIVATION (MANDATORY)

Task 2.1: Create a Horizontal Line from [Data_A] to [Data_B]. Save output to node ID: [Data_Line_AB].
Task 2.2: Create a Vertical Line from [Data_B] to [Data_C]. Save output to node ID: [Data_Line_BC].
Task 2.3: Create a Horizontal Line from [Data_C] to [Data_D]. Save output to node ID: [Data_Line_CD].
Task 2.4: Create a Vertical Line from [Data_D] to [Data_A]. Save output to node ID: [Data_Line_DA].

Once all 4 lines are successfully derived, output a final status node named: [STATE_PHASE_2_DONE]."""


PROMPTS["TASK_PHASE_3"] = """PHASE 3 — LINE FILLING (STRICT)

Task 3.1: Apply [RULE_R4_HORIZONTAL] to [Data_Line_AB]. Save the List of Filled objects to node ID: [Data_Fill_AB].
Task 3.2: Apply [RULE_R4_VERTICAL] to [Data_Line_BC]. Save the List of Filled objects to node ID: [Data_Fill_BC].
Task 3.3: Apply [RULE_R4_HORIZONTAL] to [Data_Line_CD]. Save the List of Filled objects to node ID: [Data_Fill_CD].
Task 3.4: Apply [RULE_R4_VERTICAL] to [Data_Line_DA]. Save the List of Filled objects to node ID: [Data_Fill_DA].

Once all 4 fill lists are successfully derived, output a final status node named: [STATE_PHASE_3_DONE]."""