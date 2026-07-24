from __future__ import annotations

from enum import StrEnum

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class HuggingFaceEmbeddingMode(StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"


class HuggingFaceEmbeddingConfig(EmbeddingConfig):
    # Local mode: `SentenceTransformer` model id/path passed to
    # `HuggingFaceEmbeddings.model_name`. Hosted mode: model id passed to
    # `InferenceClient`/`HuggingFaceEndpointEmbeddings`.
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Selects whether embeddings run locally via `sentence-transformers`
    # or remotely via the Hugging Face Inference API/endpoint.
    mode: HuggingFaceEmbeddingMode = HuggingFaceEmbeddingMode.LOCAL
    # Local mode: `HuggingFaceEmbeddings.model_kwargs` (Sentence Transformer
    # constructor kwargs, e.g. `device`). Hosted mode: merged into the
    # `InferenceClient.feature_extraction(**model_kwargs)` call — the only
    # pass-through channel `HuggingFaceEndpointEmbeddings` offers.
    model_kwargs: dict | None = None
    # Local mode only: `HuggingFaceEmbeddings.encode_kwargs`, forwarded to
    # `SentenceTransformer.encode(...)`.
    encode_kwargs: dict | None = None
    # `HuggingFaceEndpointEmbeddings.provider` — hosted-inference provider
    # name (e.g. "sambanova").
    provider: str | None = None
    # Hosted mode only, routed via `model_kwargs={"truncate": ...}` into
    # `InferenceClient.feature_extraction(truncate=...)`.
    truncate: bool | None = None
    # Hosted mode only, routed via `model_kwargs={"normalize": ...}` into
    # `InferenceClient.feature_extraction(normalize=...)`. Only available on
    # Text-Embeddings-Inference-backed servers.
    normalize: bool | None = None


# sources: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
