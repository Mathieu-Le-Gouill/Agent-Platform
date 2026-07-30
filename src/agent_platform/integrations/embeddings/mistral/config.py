from __future__ import annotations

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class MistralEmbeddingConfig(EmbeddingConfig):
    # Mistral embedding model id passed to the `/v1/embeddings` endpoint.
    model: str = "mistral-embed"
    # `max_retries` (inherited from `RequestOptions`) is unused: the native
    # `mistralai` SDK's retry config is a time-based backoff with no
    # attempt-count knob (same gap as the LLM domain's Mistral provider);
    # the platform's own `@with_retry()` decorator on
    # `aembed_document`/`aembed_query` already provides equivalent retry
    # behavior.
    # Base URL for the Mistral API, forwarded as `server_url`.
    endpoint: str = "https://api.mistral.ai/v1/"
    # Unused post-migration: was a LangChain-wrapper-level throttle/concurrency
    # knob for its internal multi-batch dispatch. The provider now issues one
    # `embeddings.create` call per `embed_document`/`embed_query` invocation
    # (batching already happens one level up, in `components/embedder.py`),
    # so there is no per-provider concurrency to throttle.
    wait_time: int | None = None
    # Unused post-migration, same reason as `wait_time` above.
    max_concurrent_requests: int | None = None
    # NOTE: base `dimensions` IS now forwarded (as `output_dimension`) — the
    # native SDK exposes Matryoshka truncation that `langchain-mistralai`
    # never surfaced. This is the correctness fix this migration exists for.


# sources: https://docs.mistral.ai/api/endpoint/embeddings
