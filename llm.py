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
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Pull and strip the key to prevent header injection errors
        #self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.api_key ="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"
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
    #test_openrouter()