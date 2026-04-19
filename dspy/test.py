from modules import SemanticRouter, DecomposerAgent, ComputeAgent, DataExtractorAgent
import dspy
from config import configure_lm

# In your main engine runtime

query = f"""
Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE].

Task 2.1: Create a Horizontal Line from [Data_A] to [Data_B]. Save output to node ID: [Data_Line_AB].
Task 2.2: Create a Vertical Line from [Data_B] to [Data_C]. Save output to node ID: [Data_Line_BC].
Task 2.3: Create a Horizontal Line from [Data_C] to [Data_D]. Save output to node ID: [Data_Line_CD].
Task 2.4: Create a Vertical Line from [Data_D] to [Data_A]. Save output to node ID: [Data_Line_DA].

Once all 4 lines are successfully derived, output a final status node named: [STATE_PHASE_2_DONE].

Task 3.1: Apply [RULE_R4_HORIZONTAL] to [Data_Line_AB]. Save the List of Filled objects to node ID: [Data_Fill_AB].
Task 3.2: Apply [RULE_R4_VERTICAL] to [Data_Line_BC]. Save the List of Filled objects to node ID: [Data_Fill_BC].
Task 3.3: Apply [RULE_R4_HORIZONTAL] to [Data_Line_CD]. Save the List of Filled objects to node ID: [Data_Fill_CD].
Task 3.4: Apply [RULE_R4_VERTICAL] to [Data_Line_DA]. Save the List of Filled objects to node ID: [Data_Fill_DA].

Once all 4 fill lists are successfully derived, output a final status node named: [STATE_PHASE_3_DONE].

"""
query_1= """ 
the user requests a square with a side length of 10. with point A at (43,79)
"""
if __name__ == "__main__":
    # 1. WAKE UP THE ENGINE
    print("Booting up Language Model...")
    # configure_lm(backend="openrouter") # You MUST have an active LM to run inference!
    configure_lm(backend="openrouter")
    # 2. INSTANTIATE THE BLANK ARCHITECTURE
    print("Initializing Semantic Router architecture...")
    # router = SemanticRouter() 
    decomposer = DecomposerAgent()
    # compute = ComputeAgent() 
    #extractor = DataExtractorAgent()

    # print(f"\nTesting Query: {query}\n")
    # output = compute(instruction=query, target_output_id="[Data_B]")

    print("testing decomposer agent...")
    output = decomposer(explicit_target=query)
    print(output.model_dump_json(indent=4))
    for k, v in output.model_dump().items():
        print(f"{k}: {v}")

    print("testing data extractor agent...")
    
    # output = extractor(source_text=query)
    # #print(output.model_dump_json(indent=4))
    # for node in output:
    #     for k, v in node.model_dump().items():
    #         print(f"{k}: {v}")

