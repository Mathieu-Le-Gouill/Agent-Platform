from __future__ import annotations

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class MistralEmbeddingConfig(EmbeddingConfig):
    # Mistral embedding model id passed to the `/v1/embeddings` endpoint.
    model: str = "mistral-embed"
    # `max_retries` (inherited from `RequestOptions`) is unused: the native
    # `mistralai` SDK's retry config is a time-based backoff with no
    # attempt-count knob; the platform's own `@with_retry()` decorator on
    # `aembed_document`/`aembed_query` already provides equivalent retry
    # behavior.
    # Base URL for the Mistral API, forwarded as `server_url`.
    endpoint: str = "https://api.mistral.ai/v1/"
    # Unused: the provider issues one `embeddings.create` call per
    # `embed_document`/`embed_query` invocation (batching happens one level
    # up, in `components/embedder.py`), so there is no per-provider
    # concurrency to throttle.
    wait_time: int | None = None
    # Unused, same reason as `wait_time` above.
    max_concurrent_requests: int | None = None


# sources: https://docs.mistral.ai/api/endpoint/embeddings
