from agent_platform.core.schemas.config import ProviderConfig


class RerankerConfig(ProviderConfig):
    model: str = "rerank-english-v3.0"
    top_k: int | None = None
    return_scores: bool = False
    batch_size: int = 32
    normalize_scores: bool = True
