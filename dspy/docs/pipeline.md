Stage 1: The Mold (Pydantic Schema)

The Goal: Force the LLM to externalize its reasoning before it commits to a classification. We use the "Upside-Down Schema" to cure autoregressive hallucination.



Step 1: Define the scratchpad at the very top. The LLM must analyze the available data versus the required data first.

Step 2: Define the Payload and the Escape Hatch strings.

Step 3: Define the Classification Booleans at the very bottom.

The Code (Epistemic Base):



Python



from pydantic import BaseModel, Fieldclass AgentDecision(BaseModel):

    # 1. THE WORK

    analysis_scratchpad: str = Field(description="Step-by-step analysis of whether all required variables are present.")

    extracted_payload: str = Field(description="The verbatim extraction or computation. Empty string '' if data is missing.")

    missing_variables: list[str] = Field(description="List of required variables that are missing. Empty list [] if solvable.")



    # 2. THE CLASSIFICATION (The LLM looks at what it just wrote above to answer this)

    has_sufficient_data: bool = Field(description="MUST be True if extracted_payload has text. MUST be False if missing_variables has items.")

Stage 2: The Contract (DSPy Signature)

The Goal: Provide the ontology and the strict negative constraints, but leave the instructions lean enough that the Bayesian optimizer has room to rewrite them.



Step 1: Define the specific "Environment Context" for this agent (e.g., what a Phase is, what a Rule is).

Step 2: Explicitly state the mutually exclusive priority rules.

Step 3: Bind the Pydantic Mold to the OutputField.

The Code:



Python



import dspyclass AgentSignature(dspy.Signature):

    """

    You are the [Agent Name] for an AGI DAG Engine.



    ONTOLOGY:

    [Define the 2-3 key terms here]



    CRITICAL PRIORITY RULE:

    If you lack the data to solve the query, you MUST populate 'missing_variables' and leave the payload empty. Do not guess.

    """

    user_query: str = dspy.InputField()

    decision: AgentDecision = dspy.OutputField()

Stage 3: The Plumbing (DSPy Module)

The Goal: Connect the Signature to the LLM without stripping away the metadata that the downstream Metric needs.



Step 1: Use dspy.ChainOfThought or dspy.Predict natively.

Step 2: Do not unpack the .decision object in the forward pass (preventing the Zero-Score Death Loop). Let the metric handle the unpacking.

The Code:



Python



class CognitiveAgent(dspy.Module):

    def __init__(self):

        super().__init__()

        # TypedPredictor enforces the Pydantic JSON schema

        self.prog = dspy.TypedPredictor(AgentSignature)



    def forward(self, user_query: str):

        # Return the raw DSPy prediction object, NOT just the Pydantic model

        return self.prog(user_query=user_query)

Stage 4: The X-Ray Inspector (The Metric)

The Goal: Build a brutally strict judge that intercepts Python crashes, scores the LLM purely on deterministic logic, and prints detailed failure logs to the terminal.



Step 1: Wrap the entire metric in a try/except block to catch ValidationErrors.

Step 2: Check boolean alignment (Did has_sufficient_data accurately reflect the state of the payload?).

Step 3: Physically verify string rules (e.g., payload in user_query).

Step 4: Print exact failures to stdout before returning False.

Stage 5: The Crucible (Master Test Suite)

The Goal: Build the 100-300 example dataset using the Sabotage Method to ensure a uniform distribution of solvable and unsolvable tasks.



Step 1 (The Normal Cases): Use a frontier model to generate 50 perfectly solvable queries and their correct JSON outputs.

Step 2 (The Boundary Cases): Generate 50 heavily obfuscated queries (bad grammar, massive run-on paragraphs, embedded noise) that are still technically solvable.

Step 3 (The Sabotage / Epistemic Cases): Take those 100 queries and run a script to delete critical node IDs or variables. Feed them to the Teacher Model to generate the "Halt State" JSON where has_sufficient_data = False.

Step 4: Combine into a single trainset.py file.

Stage 6: The Compiler (MIPROv2)

The Goal: Unleash the Bayesian Optimizer on your Master Test Suite to find the perfect instruction set and few-shot anchor examples.



Step 1: Configure your local LM to a high-reasoning model via OpenRouter.

Step 2: Initialize MIPROv2 with your X-Ray Metric.

Step 3: Run the compiler.

Step 4: Inspect the output JSON. This compiled prompt graph is now your ultimate "Teacher" capable of generating the thousands of traces you will eventually need for QDoRA.