from __future__ import annotations

from enum import StrEnum

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class HuggingFaceEmbeddingMode(StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"


class HuggingFaceEmbeddingConfig(EmbeddingConfig):
    # Local mode: `sentence_transformers.SentenceTransformer` model id/path.
    # Hosted mode: model id passed to `InferenceClient`.
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Selects whether embeddings run locally via `sentence-transformers`
    # or remotely via the Hugging Face Inference API.
    mode: HuggingFaceEmbeddingMode = HuggingFaceEmbeddingMode.LOCAL
    # Local mode: extra `SentenceTransformer(...)` constructor kwargs (e.g.
    # `device`). Hosted mode: merged into the
    # `InferenceClient.feature_extraction(...)` call as extra kwargs.
    model_kwargs: dict | None = None
    # Local mode only: forwarded to `SentenceTransformer.encode(...)`.
    encode_kwargs: dict | None = None
    # `InferenceClient` hosted-inference provider name (e.g. "sambanova").
    provider: str | None = None
    # Hosted mode: native `feature_extraction(truncate=...)` top-level param.
    truncate: bool | None = None
    # Hosted mode: native `feature_extraction(normalize=...)` top-level param.
    normalize: bool | None = None


# sources: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
#          https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
