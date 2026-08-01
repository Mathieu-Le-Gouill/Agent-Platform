from __future__ import annotations

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OllamaEmbeddingConfig(EmbeddingConfig):
    # The Ollama model name passed to `/api/embed`.
    model: str = "nomic-embed-text"
    # Part of the `options` dict on `/api/embed`.
    top_p: float | None = None
    # Part of the `options` dict on `/api/embed`.
    top_k: int | None = None
    # Part of the `options` dict on `/api/embed`.
    temperature: float | None = None
    # Seconds (or a duration string like "5m") the model stays loaded in
    # memory after the request (Ollama server default: 5 minutes).
    keep_alive: int | str | None = None
    # NOTE: base `batch_size` is never forwarded — the native `ollama` SDK's
    # `embed()` has no request-batching-count field; it sends the full input
    # list to `/api/embed` in one call.


# sources: https://github.com/ollama/ollama/blob/main/docs/api.md
