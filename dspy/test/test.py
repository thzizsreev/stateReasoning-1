from modules import SemanticRouter, DecomposerAgent, ComputeAgent, DataExtractorAgent
import dspy
import litellm
import os
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
The Scenario: A marketing director types a single, highly abstract goal into the system. They provide no steps, no variables, and no instructions on how to do it.

User Query: > "Generate a personalized AI welcome video for our new enterprise client, Acme Corp, and email it to their lead.
"""
if __name__ == "__main__":
    #os.environ["OPENROUTER_API_KEY"] = "your_key_here"

    response = litellm.completion(
        model="openrouter/meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "Say hello."}],
        api_base="https://openrouter.ai/api/v1",
        api_key="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",
    )
    print(response)
    # 1. WAKE UP THE ENGINE
    # print("Booting up Language Model...")
    # # configure_lm(backend="openrouter") # You MUST have an active LM to run inference!
    # configure_lm(backend="openrouter")
    # # 2. INSTANTIATE THE BLANK ARCHITECTURE
    # print("Initializing Semantic Router architecture...")
    # # router = SemanticRouter() 
    # # decomposer = DecomposerAgent()
    # # compute = ComputeAgent() 
    # #extractor = DataExtractorAgent()
    # expander = ExpanderAgent()
    # print(f"\nTesting Query: {query_1}\n")
    # output = expander(explicit_target=query_1)
    # print(output.model_dump_json(indent=4))
    # for task in output.extracted_tasks:
    #     for k, v in task.model_dump().items():
    #         print(f"{k}: {v}")

    # print(f"\nTesting Query: {query}\n")
    # output = compute(instruction=query, target_output_id="[Data_B]")

    # print("testing decomposer agent...")
    # output = decomposer(explicit_target=query)
    # print(output.model_dump_json(indent=4))
    # for k, v in output.model_dump().items():
    #     print(f"{k}: {v}")

    # print("testing data extractor agent...")
    
    # output = extractor(source_text=query)
    # #print(output.model_dump_json(indent=4))
    # for node in output:
    #     for k, v in node.model_dump().items():
    #         print(f"{k}: {v}")

