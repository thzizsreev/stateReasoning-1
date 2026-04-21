import ollama
import json
import re
from typing import Dict, Any
from openai import OpenAI
import os

OPENROUTER_API_KEY_HARDCODED = ""


def parse_llm_json_response(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from an LLM response.

    Uses regex to locate candidate JSON blocks (including ```json fenced blocks),
    then validates candidates via json.loads().
    """
    if not text:
        raise ValueError("Empty LLM response; expected JSON.")

    # 1) Prefer fenced JSON blocks.
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fenced:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2) Try to find JSON object candidates in raw text.
    # Greedy match gives us the largest {...} span; we then validate by parsing.
    candidates = re.findall(r"\{[\s\S]*\}", text)
    for cand in candidates:
        cand = cand.strip()
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3) Fallback: brace-balanced scan starting from first '{' (handles extra trailing text).
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        parsed = json.loads(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break

    preview = text[:800]
    raise ValueError(f"Could not extract a valid JSON object. First 800 chars:\n{preview}")


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
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Pull and strip the key to prevent header injection errors
        self.api_key = (api_key or OPENROUTER_API_KEY_HARDCODED or os.environ.get("OPENROUTER_API_KEY", "")).strip()
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
    #test_openrouter()