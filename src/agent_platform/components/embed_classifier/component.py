from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.components.chunker.component import Chunker
from agent_platform.components.embed_classifier.config import (
    EmbeddingClassifierConfig,
)
from agent_platform.components.embedder.component import Embedder
from agent_platform.components.llm_classifier.component import LLMClassifier
from agent_platform.components.similarity_scorer.component import (
    SimilarityConfig,
    SimilarityInput,
    SimilarityScorer,
)
from agent_platform.core.interfaces.classification.response import (
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument

EmbedConfigT = TypeVar("EmbedConfigT", bound=EmbeddingConfig)

_EmbedClassifierInput = tuple[list[TextDocument], list[str], EmbeddingClassifierConfig]


class EmbeddingClassifier(
    Component[_EmbedClassifierInput, ClassificationResponse],
    Generic[EmbedConfigT],
):
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder[EmbedConfigT],
        similarity_scorer: SimilarityScorer,
        llm_classifier: LLMClassifier | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._similarity_scorer = similarity_scorer
        self._llm_classifier = llm_classifier

    async def arun(self, input: _EmbedClassifierInput) -> ClassificationResponse:
        items, candidate_labels, config = input

        chunks_per_doc: dict[int, list[TextChunk]] = {}
        for i, doc in enumerate(items):
            chunks = await self._chunker.arun([doc])
            chunks_per_doc[i] = chunks

        all_chunks = [c for chunks in chunks_per_doc.values() for c in chunks]

        embed_response = await self._embedder.arun(all_chunks)
        chunk_vectors = [
            (chunk, list(emb.vector))
            for chunk, emb in zip(all_chunks, embed_response.embeddings)
        ]

        label_chunks = [TextChunk(text=label) for label in candidate_labels]
        label_embed_response = await self._embedder.arun(label_chunks)
        label_vectors = {
            label: list(emb.vector)
            for label, emb in zip(candidate_labels, label_embed_response.embeddings)
        }

        results: list[ClassificationResult] = []
        for doc_idx in range(len(items)):
            doc_chunks = chunks_per_doc[doc_idx]
            doc_chunk_vecs = [(c, v) for c, v in chunk_vectors if c in doc_chunks]

            sim_input = SimilarityInput(doc_chunk_vecs, label_vectors)
            sim_config = SimilarityConfig(
                top_k=config.top_k,
                threshold=config.similarity_threshold,
                multi_label=config.multi_label,
                unknown_label=config.unknown_label,
                metric=config.similarity_metric,
            )
            self._similarity_scorer._config = sim_config
            result = await self._similarity_scorer.arun(sim_input)

            if config.llm_rerank and self._llm_classifier is not None:
                llm_cfg = config.llm_classifier_config
                if llm_cfg is not None:
                    rerank_input = ([items[doc_idx]], candidate_labels, llm_cfg)
                    rerank_response = await self._llm_classifier.arun(rerank_input)
                    if rerank_response.results:
                        result = rerank_response.results[0]

            results.append(result)

        return ClassificationResponse(results=results)
