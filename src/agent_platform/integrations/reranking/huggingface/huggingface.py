from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pydantic import ConfigDict

from agent_platform.integrations.reranking.langchain_base import LangChainReranker
from agent_platform.integrations.reranking.huggingface.config import (
    HuggingFaceRerankerConfig,
)


class _ScoredCrossEncoderReranker(BaseDocumentCompressor):
    """Reimplements langchain's CrossEncoderReranker with two fixes:

    - `top_n` is optional (the upstream class hardcodes `int = 3`), so we can
      leave truncation entirely to the caller's own `top_k` slicing.
    - the relevance score is written to `doc.metadata["relevance_score"]`,
      which the upstream `CrossEncoderReranker.compress_documents` never
      does (it sorts and returns bare documents).

    Also sidesteps `langchain_community.document_compressors.CrossEncoderReranker`,
    which no longer exists in the installed langchain-community version (it
    moved to `langchain_classic`); depending only on `BaseDocumentCompressor`
    keeps this provider decoupled from that internal reshuffle.
    """

    model: Any
    top_n: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Any = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        if self.top_n is not None:
            ranked = ranked[: self.top_n]

        result = []
        for doc, score in ranked:
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = float(score)
            result.append(doc_copy)
        return result


class HuggingFaceRerankerProvider(LangChainReranker[HuggingFaceRerankerConfig]):
    def _default_config(self) -> HuggingFaceRerankerConfig:
        return HuggingFaceRerankerConfig()

    def _client(self, config: HuggingFaceRerankerConfig) -> BaseDocumentCompressor:
        encoder = HuggingFaceCrossEncoder(
            model_name=config.model,
            model_kwargs={"device": config.device},
        )
        return _ScoredCrossEncoderReranker(model=encoder, top_n=config.top_k)
