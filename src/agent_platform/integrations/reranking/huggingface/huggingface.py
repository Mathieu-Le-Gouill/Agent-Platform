from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from agent_platform.integrations.reranking.langchain_base import LangChainReranker
from agent_platform.integrations.reranking.huggingface.config import (
    HuggingFaceRerankerConfig,
)


class HuggingFaceRerankerProvider(
    LangChainReranker[HuggingFaceRerankerConfig]
):
    def _default_config(self) -> HuggingFaceRerankerConfig:
        return HuggingFaceRerankerConfig()

    def _client(self, config: HuggingFaceRerankerConfig) -> CrossEncoderReranker:
        encoder = HuggingFaceCrossEncoder(model_name=config.model)
        return CrossEncoderReranker(model=encoder)
