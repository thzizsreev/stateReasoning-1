"""
DSPy LM Configuration.

Replaces: OllamaClient / OpenRouterClient injection throughout agents.py.

DSPy uses a global LM configured once at startup. All modules use it automatically.
You can swap the backend (Ollama ↔ OpenRouter ↔ OpenAI) with a single line change here.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import dspy

from pipeline import QueryPipeline, ComputeNodeExecutor, process_query, execute_compute_node

_DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_dotenv(path: Path = _DEFAULT_DOTENV_PATH) -> None:
    """
    Minimal .env loader (no extra dependency).

    - Supports KEY=VALUE (optionally quoted).
    - Ignores blank lines and comments starting with '#'.
    - Does NOT override variables already present in the environment.
    """
    try:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            existing = os.environ.get(key)
            if existing is None or str(existing).strip() == "":
                os.environ[key] = value
    except Exception:
        return


def configure_lm(backend: str = "openrouter", model: Optional[str] = None):
    """
    Configure the global DSPy LM.

    Backends:
      - "openrouter": dspy.LM with base_url override (matches OpenRouterClient)
      - "ollama":     dspy.LM with Ollama base_url
      - "openai":     dspy.LM direct (expects OPENAI_API_KEY in env)
    """
    if backend == "openrouter":
        _load_dotenv()
        model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. Put it in stateReasoning-1/.env "
                "or set it in your environment before running."
            )

        # LiteLLM's OpenAI adapter (used by `openai/*` models) reads OPENAI_API_KEY.
        # When using OpenRouter via api_base, we map the OpenRouter key into OPENAI_API_KEY.
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        lm = dspy.LM(
            model=f"openai/{model}",
            api_base=api_base,
            api_key="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",
            temperature=0.0,
            max_tokens=2048,
        )
    elif backend == "ollama":
        model = model or os.getenv("OLLAMA_MODEL", "mistral")
        lm = dspy.LM(
            model=f"ollama/{model}",
            api_base="http://localhost:11434",
            temperature=0.0,
            max_tokens=2048,
        )
    elif backend == "openai":
        _load_dotenv()
        model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        lm = dspy.LM(
            model=model,
            temperature=0.0,
            max_tokens=2048,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    dspy.configure(lm=lm)
    print(f"  [DSPy] LM configured: {backend} / {model}")
    return lm


if __name__ == "__main__":
    configure_lm(backend="openrouter")
    pipeline = QueryPipeline()

    query2 = """Task 1.1: Apply [RULE_R1] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_B].
Task 1.2: Apply [RULE_R2] to [Data_A] using length [Data_L]. Save the output to node ID: [Data_D].
Task 1.3: Apply [RULE_R3] using [Data_B] and [Data_D]. Save the output to node ID: [Data_C].

Once [Data_B], [Data_C], and [Data_D] are successfully derived, output a final status node named: [STATE_PHASE_1_DONE]."""

    result = asyncio.run(process_query(query2, pipeline))
    print("\n=== FINAL PAYLOAD ===")
    print(result.model_dump_json(indent=2))
