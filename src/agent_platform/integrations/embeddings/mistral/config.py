from __future__ import annotations
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class MistralEmbeddingConfig(EmbeddingConfig):
    # `MistralAIEmbeddings.model` — the Mistral embedding model id passed to
    # the `/v1/embeddings` endpoint.
    model: str = "mistral-embed"
    # Retry count for `MistralAIEmbeddings.max_retries`.
    max_retries: int | None = None
    # `MistralAIEmbeddings.endpoint` — base URL for the Mistral API.
    endpoint: str = "https://api.mistral.ai/v1/"
    # `MistralAIEmbeddings.wait_time` — seconds to wait before retrying on a 429 response.
    wait_time: int | None = None
    # `MistralAIEmbeddings.max_concurrent_requests` — max in-flight requests.
    max_concurrent_requests: int | None = None
    # NOTE: base `dimensions` is intentionally never forwarded — installed
    # `langchain-mistralai` 1.1.6 `MistralAIEmbeddings` has `extra="forbid"`
    # and no `dimensions` field (verified against installed site-packages).
    # The Mistral REST API's Matryoshka `output_dimension` truncation isn't
    # exposed by this client version — a library gap, not fixable via config.
    # NOTE: base `batch_size` is also never forwarded — `MistralAIEmbeddings`
    # batches internally by token count (`_get_batches`, 16k-token cap), with
    # no count-based batch-size knob to map `batch_size` onto.


# sources: https://docs.mistral.ai/api/endpoint/embeddings
