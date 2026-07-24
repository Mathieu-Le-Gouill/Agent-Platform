from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "OpenAIEmbeddingProvider": "agent_platform.integrations.embeddings.openai.openai",
    "MistralEmbeddingProvider": "agent_platform.integrations.embeddings.mistral.mistral",
    "OllamaEmbeddingProvider": "agent_platform.integrations.embeddings.ollama.ollama",
    "HuggingFaceEmbeddingProvider": "agent_platform.integrations.embeddings.huggingface.huggingface",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
