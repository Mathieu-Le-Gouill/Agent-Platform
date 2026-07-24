from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "OpenAILLM": "agent_platform.integrations.llm.openai.openai",
    "AnthropicLLM": "agent_platform.integrations.llm.anthropic.anthropic",
    "MistralLLM": "agent_platform.integrations.llm.mistral.mistral",
    "OllamaLLM": "agent_platform.integrations.llm.ollama.ollama",
    "HuggingFaceLLM": "agent_platform.integrations.llm.huggingface.huggingface",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
