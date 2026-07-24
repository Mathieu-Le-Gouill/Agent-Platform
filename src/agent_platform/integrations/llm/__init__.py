from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "OpenAILLM": "agent_platform.integrations.llm.openai.openai",
    "AnthropicLLM": "agent_platform.integrations.llm.anthropic.anthropic",
    "MistralLLM": "agent_platform.integrations.llm.mistral.mistral",
    "OllamaLLM": "agent_platform.integrations.llm.ollama.ollama",
    "HuggingFaceLLM": "agent_platform.integrations.llm.huggingface.huggingface",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
