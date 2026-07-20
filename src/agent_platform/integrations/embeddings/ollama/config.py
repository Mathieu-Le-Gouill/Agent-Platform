from __future__ import annotations
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OllamaEmbeddingConfig(EmbeddingConfig):
    # `OllamaEmbeddings.model` — the Ollama model name passed to `/api/embed`.
    model: str = "nomic-embed-text"
    # `OllamaEmbeddings.top_p`, part of the `options` dict on `/api/embed`.
    top_p: float | None = None
    # `OllamaEmbeddings.top_k`, part of the `options` dict on `/api/embed`.
    top_k: int | None = None
    # `OllamaEmbeddings.temperature`, part of the `options` dict on `/api/embed`.
    temperature: float | None = None
    # `OllamaEmbeddings.keep_alive` — seconds the model stays loaded in
    # memory after the request (Ollama server default: 5 minutes). Unlike
    # the Ollama LLM client, `OllamaEmbeddings.keep_alive` is `int | None`
    # only — no duration-string form (verified against installed
    # `langchain_ollama/embeddings.py`).
    keep_alive: int | None = None
    # NOTE: base `batch_size` is never forwarded — installed `langchain-ollama`
    # `OllamaEmbeddings` has no request-batching-count field; it sends the
    # full input list to `/api/embed` in one call.


# sources: https://github.com/ollama/ollama/blob/main/docs/api.md
