from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "OpenAILLM": "agent_platform.integrations.llm.openai.provider",
    "AnthropicLLM": "agent_platform.integrations.llm.anthropic.provider",
    "MistralLLM": "agent_platform.integrations.llm.mistral.provider",
    "OllamaLLM": "agent_platform.integrations.llm.ollama.provider",
    "HuggingFaceLLM": "agent_platform.integrations.llm.huggingface.provider",
    "GoogleLLM": "agent_platform.integrations.llm.google.provider",
}

# Short slugs for `Settings.default_llm_model`'s "<provider>:<model>" strings
# (see `core.config.parse_model_string`), resolved by `config.container.build_provider`.
PROVIDER_ALIASES: dict[str, str] = {
    "openai": "OpenAILLM",
    "anthropic": "AnthropicLLM",
    "mistral": "MistralLLM",
    "ollama": "OllamaLLM",
    "huggingface": "HuggingFaceLLM",
    "google": "GoogleLLM",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
