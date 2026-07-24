from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OpenAIEmbeddingConfig(EmbeddingConfig):
    # `OpenAIEmbeddings.model` — the OpenAI embedding model id passed to
    # `embeddings.create(model=...)`.
    model: str = "text-embedding-ada-002"
    # `OpenAIEmbeddings.max_retries` — max retry attempts on failed requests
    # to the OpenAI client.
    max_retries: int | None = None
    # Extra kwargs merged into the OpenAI `embeddings.create` call; only
    # forwarded to the client when explicitly set (`OpenAIEmbeddings.model_kwargs`
    # is non-Optional with `default_factory=dict`, so `None` cannot be passed
    # through directly). See installed `langchain_openai/embeddings/base.py`.
    model_kwargs: dict | None = None
    # No dedicated constructor field on `OpenAIEmbeddings`; routed through
    # `model_kwargs` (the client's own escape-hatch channel for `create()` params).
    encoding_format: Literal["float", "base64"] | None = None
    # `OpenAIEmbeddings.check_embedding_ctx_length` — whether to tokenize and
    # auto-split inputs longer than the model's context length before sending.
    check_embedding_ctx_length: bool = True


"""
sources: https://platform.openai.com/docs/api-reference/embeddings/create
         https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html
"""
