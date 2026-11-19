from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from agent_platform.components.base import Component
from agent_platform.components.chunking.chunker.component import Chunker, ChunkerInput
from agent_platform.components.classification.embed_classifier.config import (
    EmbeddingClassifierConfig,
)
from agent_platform.components.classification.similarity_scorer.component import (
    SimilarityConfig,
    SimilarityInput,
    SimilarityScorer,
)
from agent_platform.components.retrieval.embedder.component import (
    Embedder,
    EmbedderInput,
)
from agent_platform.core.interfaces.classification.response import (
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument

EmbedConfigT = TypeVar("EmbedConfigT", bound=EmbeddingConfig)


class EmbedClassifierInput(NamedTuple):
    items: list[TextDocument]
    candidate_labels: list[str]
    config: EmbeddingClassifierConfig


class EmbeddingClassifier(
    Component[EmbedClassifierInput, ClassificationResponse],
    Generic[EmbedConfigT],
):
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder[EmbedConfigT],
        similarity_scorer: SimilarityScorer,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._similarity_scorer = similarity_scorer

    async def arun(self, input: EmbedClassifierInput) -> ClassificationResponse:
        items, candidate_labels, config = input

        chunks_per_doc: dict[int, list[TextChunk]] = {}
        for i, doc in enumerate(items):
            chunks = await self._chunker.arun(ChunkerInput([doc], None))
            chunks_per_doc[i] = chunks

        # embed every doc's chunks in one batched call, then re-split by doc below
        all_chunks = [c for chunks in chunks_per_doc.values() for c in chunks]

        embed_response = await self._embedder.arun(EmbedderInput(all_chunks, None))
        chunk_vectors = [
            (chunk, list(emb.vector))
            for chunk, emb in zip(all_chunks, embed_response.embeddings)
        ]

        label_chunks = [TextChunk(text=label) for label in candidate_labels]
        label_embed_response = await self._embedder.arun(
            EmbedderInput(label_chunks, None)
        )
        label_vectors = {
            label: list(emb.vector)
            for label, emb in zip(candidate_labels, label_embed_response.embeddings)
        }

        results: list[ClassificationResult] = []
        for doc_idx in range(len(items)):
            doc_chunks = chunks_per_doc[doc_idx]
            doc_chunk_vecs = [(c, v) for c, v in chunk_vectors if c in doc_chunks]

            sim_config = SimilarityConfig(
                top_k=config.top_k,
                threshold=config.similarity_threshold,
                multi_label=config.multi_label,
                unknown_label=config.unknown_label,
                metric=config.similarity_metric,
            )
            sim_input = SimilarityInput(doc_chunk_vecs, label_vectors, sim_config)
            result = await self._similarity_scorer.arun(sim_input)
            results.append(result)

        return ClassificationResponse(results=results)
