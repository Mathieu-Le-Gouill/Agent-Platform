from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OpenAIEmbeddingConfig(EmbeddingConfig):
    # OpenAI embedding model id passed to `embeddings.create(model=...)`.
    model: str = "text-embedding-ada-002"
    # Max retry attempts on failed requests, forwarded to the client constructor.
    max_retries: int | None = None
    # Extra kwargs merged into the `embeddings.create` call.
    model_kwargs: dict | None = None
    # Native top-level `encoding_format` param on `embeddings.create`.
    encoding_format: Literal["float", "base64"] | None = None
    # Unused post-migration: was LangChain's client-side auto-tokenize-and-split
    # behavior for inputs exceeding the model's context length. The native
    # `openai` SDK's `embeddings.create` has no equivalent; requests longer
    # than the model's context simply get a 400 from the API.
    check_embedding_ctx_length: bool = True


"""
sources: https://developers.openai.com/api/reference/resources/embeddings (exact request schema)
"""
