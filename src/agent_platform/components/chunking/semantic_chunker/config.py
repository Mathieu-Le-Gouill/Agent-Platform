from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent_platform.core.schemas.enums import SimilarityMetric


class SemanticChunkerConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    breakpoint_percentile_threshold: float = 95.0
    min_sentences_per_chunk: int = 1
    sentence_split_regex: str = r"(?<=[.?!])\s+"
    metric: SimilarityMetric = SimilarityMetric.COSINE
