from pydantic import BaseModel, ConfigDict, Field

from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.schemas.enums import SimilarityMetric
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
)


class EmbeddingClassifierConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    chunker_config: RecursiveChunkerConfig = Field(
        default_factory=RecursiveChunkerConfig,
    )
    embedding_config: EmbeddingConfig = Field(
        default_factory=HuggingFaceEmbeddingConfig,
    )

    top_k: int = 5
    similarity_threshold: float | None = None
    multi_label: bool = False
    unknown_label: str = "unknown"
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    llm_rerank: bool = False
    llm_classifier_config: LLMClassifierConfig | None = None
