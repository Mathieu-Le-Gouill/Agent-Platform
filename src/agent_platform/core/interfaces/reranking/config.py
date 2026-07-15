from pydantic import BaseModel, ConfigDict


class RerankerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "rerank-english-v3.0"
    top_k: int | None = None
    return_scores: bool = False
    batch_size: int = 32
    normalize_scores: bool = True
