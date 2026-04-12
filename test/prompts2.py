PROMPTS = {
    "DEF_PRIMITIVES": "PRIMITIVE OBJECTS: Point(name, row, col).",
    "RULE_R1": "Rule R1 (HorizontalExtension): INPUT: Point P, Integer L -> OUTPUT: Point Q = (P.row, P.col + L)",
    "RULE_R2": "Rule R2 (VerticalExtension): INPUT: Point P, Integer L -> OUTPUT: Point Q = (P.row + L, P.col)",
    "RULE_R3": "Rule R3 (CornerCompletion): INPUT: Point B, Point D -> OUTPUT: Point C = (D.row, B.col)",
    "PHASE_1": "Derive [Data_B] from [Data_A] using [RULE_R1] and [Data_L]. Derive [Data_D] from [Data_A] using [RULE_R2] and [Data_L]. Derive [Data_C] from [Data_B] and [Data_D] using [RULE_R3]. Once [Data_C] is created, output a target node named [PHASE_1_DONE].",
}

