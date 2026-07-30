from agent_platform.core.config import ModelConfig, RequestOptions


class RerankerConfig(RequestOptions, ModelConfig):
    # Provider-neutral placeholder; every provider subclass overrides this
    # with its own current model id. Kept as a currently-valid Cohere id
    # (docs.cohere.com/reference/rerank) so base-class instantiation stays usable.
    model: str = "rerank-v4.0-fast"
    # Number of top results to return; forwarded to the provider's own
    # top_n/top_k param where supported, then re-sliced client-side.
    top_k: int | None = None
    # Whether to populate TextChunk.confidence with the provider's
    # relevance score. See integrations/reranking/scoring.py.
    return_scores: bool = False
    # Batch size for chunked reranking (components/reranker.py); not every
    # provider has a native equivalent (e.g. flashrank has none).
    batch_size: int = 32
    # When True, min-max normalize returned scores into [0, 1]; when False,
    # scores are wrapped unbounded via Score.logit to avoid validation
    # errors on out-of-range raw provider scores.
    normalize_scores: bool = True
