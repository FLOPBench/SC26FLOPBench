import json
import os
import sys
import time
from pathlib import Path
import json
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.llm_models import OpenRouterLLMSettings, build_openrouter_llm
from langchain_core.messages import HumanMessage

QUESTION = "What is the capital of France?"


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _flatten_message_content(content.get("text") or content.get("content"))
    if isinstance(content, list):
        return "".join(_flatten_message_content(part) for part in content if part)
    return ""


def _extract_response_text(message: Any) -> str:
    return _flatten_message_content(getattr(message, "content", ""))


def _get_response_id(message: Any) -> str | None:
    metadata = getattr(message, "response_metadata", None) or {}
    response_id = metadata.get("id")
    if isinstance(response_id, list):
        return response_id[-1] if response_id else None
    return response_id


def test_openrouter_api_query() -> None:
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        raise RuntimeError(
            "Missing OPENAI_API_KEY/OPENROUTER_API_KEY; set one before running the OpenRouter check."
        )

    settings = OpenRouterLLMSettings(model_name="openai/gpt-5.1-codex-mini")
    llm = build_openrouter_llm(settings)

    prompt = [HumanMessage(content=QUESTION)]
    print("Running OpenRouter LLM sanity check...", flush=True)
    start = time.perf_counter()
    result = llm.generate([prompt])
    duration = time.perf_counter() - start

    generation = result.generations[0][0]
    answer_text = _extract_response_text(generation.message)
    print(f"OpenRouter query took {duration:.3f} seconds")
    print(f"LLM output text: {answer_text}")

    token_usage = (result.llm_output or {}).get("token_usage") or {}
    prompt_tokens = token_usage.get("prompt_tokens")
    completion_tokens = token_usage.get("completion_tokens")
    total_tokens = token_usage.get("total_tokens")
    print(
        "Tokens (prompt/completion/total):",
        prompt_tokens or "unknown",
        completion_tokens or "unknown",
        total_tokens or "unknown",
    )

    response_id = _get_response_id(generation.message)
    print("Query identifier hash:", response_id or "unknown")

    if result.run:
        print("LangChain run_id:", result.run[0].run_id)

    _print_metadata(result, generation.message)

    assert "paris" in answer_text.lower(), "Model did not mention Paris in its answer."


def _print_metadata(result: Any, message: Any) -> None:
    def _pretty(obj: Any) -> str:
        if obj in (None, {}, []):
            return "none"
        return json.dumps(obj, indent=2, default=str)

    print("LLM output metadata:\n", _pretty(result.llm_output))
    response_metadata = getattr(message, "response_metadata", None)
    print("Message response metadata:\n", _pretty(response_metadata))
    message_metadata = getattr(message, "metadata", None)
    print("Message metadata:\n", _pretty(message_metadata))
    additional_kwargs = getattr(message, "additional_kwargs", None)
    print("Message additional kwargs:\n", _pretty(additional_kwargs))
    if result.run:
        print(
            "Detailed run entries:\n",
            json.dumps([run.model_dump() for run in result.run], indent=2, default=str),
        )
