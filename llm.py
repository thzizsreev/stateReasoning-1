#import ollama
import json
import re
from typing import Any, Dict, Union
from openai import OpenAI
import os


def parse_llm_json_response(text: str) -> Union[Dict[str, Any], list]:
    """
    Extract the first valid JSON object or array from an LLM response.
    Handles embedded newlines inside JSON string values via regex pre-processing.
    """
    if not text:
        raise ValueError("Empty LLM response; expected JSON.")

    def sanitize(raw: str) -> str:
        # Replace literal newlines inside JSON string values only
        # Matches content between quotes, replaces \n with \\n
        return re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r') + '"',
            raw
        )

    def try_parse(blob: str):
        blob = blob.strip()
        if not blob:
            return None
        # Try raw first
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try sanitized
        try:
            parsed = json.loads(sanitize(blob))
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            return None

    # 1) Fenced blocks
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        result = try_parse(block)
        if result is not None:
            return result

    # 2) Robust scan: JSONDecoder.raw_decode across the whole text.
    # This avoids false positives from bracketed IDs like "[Data_A]" in scratchpads.
    decoder = json.JSONDecoder()
    candidates = []
    for start_idx in [m.start() for m in re.finditer(r"[\[\{]", text)]:
        snippet = text[start_idx:].lstrip()
        if not snippet:
            continue

        # Try decode on raw snippet first
        try:
            parsed, _ = decoder.raw_decode(snippet)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try decode on sanitized snippet (fixes literal newlines inside strings)
        try:
            parsed, _ = decoder.raw_decode(sanitize(snippet))
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract valid JSON. First 800 chars:\n{text[:800]}")

class OllamaClient:
    """A wrapper class to handle LLM API interactions and force structured JSON."""
    def __init__(self, model_name: str = "llama3.1:latest"):
        self.model_name = model_name

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Sends the prompt to Ollama and returns a parsed JSON dictionary."""
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={"temperature": 0.0}, # 0.0 is critical for logical consistency
            format="json"
        )
        return json.loads(response['message']['content'])

class OpenRouterClient:
    """A wrapper class to handle OpenRouter API interactions and force structured JSON."""
    
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_referer: str | None = None,
        app_title: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-405b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Pull and strip the key to prevent header injection errors
        #self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.api_key =""
        print(f"Model name: {self.model_name}")
        print(f"API key: {self.api_key}")
        if not self.api_key:
            raise RuntimeError(
                "🚨 OPENROUTER_API_KEY is missing or empty. Set it in .env or as an environment variable."
            )

        self.http_referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost")
        self.app_title = app_title or os.getenv("OPENROUTER_APP_TITLE", "Neuro-Symbolic DAG Engine")

        # Match the working LightRAG/test.py OpenRouter client structure
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_title,
            },
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Sends the prompt to OpenRouter and returns a parsed JSON dictionary."""
        try:
            # Mirror OllamaClient’s contract: messages -> JSON dict
            response = self.client.chat.completions.create(
                model=self.model_name,
                #response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            raw_content = response.choices[0].message.content or ""
            print(f"Raw content: {raw_content} \n\n")
            return parse_llm_json_response(raw_content)
            
        except Exception as e:
            print(f"OpenRouter Execution Failed: {e}")
            raise e

def test_openrouter():
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY is not set in .env")
        return

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"API key  : {api_key[:12]}...{api_key[-4:]}")
    print()

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LightRAG"),
        },
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
        {"role": "user", "content": "Say hello and confirm you are working. whT Is 2+2?"},
    ]

    print("Sending request...")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        timeout=30.0,
    )

    choice = resp.choices[0].message
    print(f"Response : {choice.content}")
    print(f"Model    : {resp.model}")
    print(f"Tokens   : prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")


if __name__ == "__main__":
    test = OpenRouterClient()
    resp = """
    ```json
{
  "termination_flag_detected": false,
  "task_nodes": [
    {
      "id": "Point_Derivation",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Origin_Point_A]", "[Side_Length_L]", "[RULE_R1]", "[RULE_R2]", "[RULE_R3]"],
      "instruction": "Apply [RULE_R1] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point B. Apply [RULE_R2] to [Origin_Point_A] using length [Side_Length_L]. Save the output to a point D. Apply [RULE_R3] using the outputs generated from Task 1.1 and Task 1.2. Save the output to a point C.",
      "description": "[INTENT]: Derive three new points (B, C, D) using the given rules and origin point.

[DEFERRED LOGIC]:
- Apply [RULE_R1] to [Origin_Point_A] with length [Side_Length_L] to get point B.
- Apply [RULE_R2] to [Origin_Point_A] with length [Side_Length_L] to get point D.
- Apply [RULE_R3] to points B and D to get point C.

[SHELL ARCHITECTURE]:
- Shell: [Point_B] | Type: named shell | Content: point B coordinates | Consumer: PHASE 2 — LINE DERIVATION.
- Shell: [Point_C] | Type: named shell | Content: point C coordinates | Consumer: PHASE 2 — LINE DERIVATION.
- Shell: [Point_D] | Type: named shell | Content: point D coordinates | Consumer: PHASE 2 — LINE DERIVATION.

[PRUNED DEPENDENCIES]:
- [Origin_Point_A]: origin point coordinates — required as primary input for point derivation.
- [Side_Length_L]: side length value — required as primary input for point derivation.
- [RULE_R1]: rule for horizontal extension — required for point B derivation.
- [RULE_R2]: rule for vertical extension — required for point D derivation.
- [RULE_R3]: rule for corner completion — required for point C derivation."
    },
    {
      "id": "Line_Derivation",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Point_B]", "[Point_C]", "[Point_D]"],
      "instruction": "Create a Horizontal Line from [Origin_Point_A] to the point B. Save output to a new Line AB. Create a Vertical Line from the point B to the point C. Save output to a new Line BC. Create a Horizontal Line from the point D to the point C. Save output to a new Line DC. Create a Vertical Line from the point D to the [Origin_Point_A]. Save output to a new Line DA.",
      "description": "[INTENT]: Derive four new lines (AB, BC, DC, DA) using the points derived in PHASE 1.

[DEFERRED LOGIC]:
- Create a horizontal line from [Origin_Point_A] to point B to get line AB.
- Create a vertical line from point B to point C to get line BC.
- Create a horizontal line from point D to point C to get line DC.
- Create a vertical line from point D to [Origin_Point_A] to get line DA.

[SHELL ARCHITECTURE]:
- Shell: [Line_AB] | Type: named shell | Content: line AB coordinates | Consumer: PHASE 3 — LINE FILLING.
- Shell: [Line_BC] | Type: named shell | Content: line BC coordinates | Consumer: PHASE 3 — LINE FILLING.
- Shell: [Line_DC] | Type: named shell | Content: line DC coordinates | Consumer: PHASE 3 — LINE FILLING.
- Shell: [Line_DA] | Type: named shell | Content: line DA coordinates | Consumer: PHASE 3 — LINE FILLING.

[PRUNED DEPENDENCIES]:
- [Point_B]: point B coordinates — required as primary input for line AB derivation.
- [Point_C]: point C coordinates — required as primary input for lines BC and DC derivation.
- [Point_D]: point D coordinates — required as primary input for lines DC and DA derivation."
    },
    {
      "id": "Line_Fill_Horizontal_AB",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Line_AB]", "[RULE_R4_HORIZONTAL]"],
      "instruction": "Apply [RULE_R4_HORIZONTAL] to the first horizontal line generated in PHASE 2. Save the List of Filled objects to a new node ID.",
      "description": "[INTENT]: Fill the horizontal line AB using [RULE_R4_HORIZONTAL].

[DEFERRED LOGIC]:
- Apply [RULE_R4_HORIZONTAL] to line AB to get a list of filled objects.

[SHELL ARCHITECTURE]:
- Shell: [Line_Fill_Collection] | Type: collection shell | Content: list of filled objects for line AB | Consumer: Terminal node if SYS_FLAG present, else open-ended.
- DOWNSTREAM INSTRUCTION: Downstream agent MUST append each filled object into [Line_Fill_Collection]. Do not create new shells for individual filled objects.

[PRUNED DEPENDENCIES]:
- [Line_AB]: line AB coordinates — required as primary input for line filling.
- [RULE_R4_HORIZONTAL]: rule for horizontal line filling — required for filling line AB."
    },
    {
      "id": "Line_Fill_Vertical_BC",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Line_BC]", "[RULE_R4_VERTICAL]"],
      "instruction": "Apply [RULE_R4_VERTICAL] to the first vertical line generated in PHASE 2. Save the List of Filled objects to a new node ID.",
      "description": "[INTENT]: Fill the vertical line BC using [RULE_R4_VERTICAL].

[DEFERRED LOGIC]:
- Apply [RULE_R4_VERTICAL] to line BC to get a list of filled objects.

[SHELL ARCHITECTURE]:
- Shell: [Line_Fill_Collection] | Type: collection shell | Content: list of filled objects for line BC | Consumer: Terminal node if SYS_FLAG present, else open-ended.
- DOWNSTREAM INSTRUCTION: Downstream agent MUST append each filled object into [Line_Fill_Collection]. Do not create new shells for individual filled objects.

[PRUNED DEPENDENCIES]:
- [Line_BC]: line BC coordinates — required as primary input for line filling.
- [RULE_R4_VERTICAL]: rule for vertical line filling — required for filling line BC."
    },
    {
      "id": "Line_Fill_Horizontal_DC",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Line_DC]", "[RULE_R4_HORIZONTAL]"],
      "instruction": "Apply [RULE_R4_HORIZONTAL] to the second horizontal line generated in PHASE 2. Save the List of Filled objects to a new node ID.",
      "description": "[INTENT]: Fill the horizontal line DC using [RULE_R4_HORIZONTAL].

[DEFERRED LOGIC]:
- Apply [RULE_R4_HORIZONTAL] to line DC to get a list of filled objects.

[SHELL ARCHITECTURE]:
- Shell: [Line_Fill_Collection] | Type: collection shell | Content: list of filled objects for line DC | Consumer: Terminal node if SYS_FLAG present, else open-ended.
- DOWNSTREAM INSTRUCTION: Downstream agent MUST append each filled object into [Line_Fill_Collection]. Do not create new shells for individual filled objects.

[PRUNED DEPENDENCIES]:
- [Line_DC]: line DC coordinates — required as primary input for line filling.
- [RULE_R4_HORIZONTAL]: rule for horizontal line filling — required for filling line DC."
    },
    {
      "id": "Line_Fill_Vertical_DA",
      "task_type": "DECOMPOSE",
      "dependencies": ["[Line_DA]", "[RULE_R4_VERTICAL]"],
      "instruction": "Apply [RULE_R4_VERTICAL] to the second vertical line generated in PHASE 2. Save the List of Filled objects to a new node ID.",
      "description": "[INTENT]: Fill the vertical line DA using [RULE_R4_VERTICAL].

[DEFERRED LOGIC]:
- Apply [RULE_R4_VERTICAL] to line DA to get a list of filled objects.

[SHELL ARCHITECTURE]:
- Shell: [Line_Fill_Collection] | Type: collection shell | Content: list of filled objects for line DA | Consumer: Terminal node if SYS_FLAG present, else open-ended.
- DOWNSTREAM INSTRUCTION: Downstream agent MUST append each filled object into [Line_Fill_Collection]. Do not create new shells for individual filled objects.

[PRUNED DEPENDENCIES]:
- [Line_DA]: line DA coordinates — required as primary input for line filling.
- [RULE_R4_VERTICAL]: rule for vertical line filling — required for filling line DA."
    }
  ],
  "terminal_node": null
}
```
"""
    json_resp = parse_llm_json_response(resp)
    print(json_resp)
    #test_openrouter()