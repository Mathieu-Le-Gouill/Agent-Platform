from __future__ import annotations

import re
from typing import NamedTuple
from uuid import uuid4

from agent_platform.components.base import Component
from agent_platform.components.embedder.component import Embedder, EmbedderInput
from agent_platform.components.semantic_chunker.config import SemanticChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.similarity import compute_similarity


class SemanticChunkerInput(NamedTuple):
    documents: list[TextDocument]
    config: SemanticChunkerConfig | None


class SemanticChunker(Component[SemanticChunkerInput, list[TextChunk]]):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def arun(self, input: SemanticChunkerInput) -> list[TextChunk]:
        documents, config = input
        config = config or SemanticChunkerConfig()
        chunks: list[TextChunk] = []
        for doc in documents:
            chunks.extend(await self._chunk_document(doc, config))
        return chunks

    async def _chunk_document(
        self, doc: TextDocument, config: SemanticChunkerConfig
    ) -> list[TextChunk]:
        sentences = [
            s.strip()
            for s in re.split(config.sentence_split_regex, doc.text)
            if s.strip()
        ]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(doc, sentences[0], 0)]

        embed_response = await self._embedder.arun(
            EmbedderInput([TextChunk(text=s) for s in sentences], None)
        )
        vectors = [list(e.vector) for e in embed_response.embeddings]

        # distance (1 - similarity) between consecutive sentences: a spike means a topic shift
        distances = [
            1.0 - compute_similarity(vectors[i], vectors[i + 1], config.metric)
            for i in range(len(vectors) - 1)
        ]
        threshold = _percentile(distances, config.breakpoint_percentile_threshold)
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        groups: list[list[str]] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            # ignore a breakpoint until the group meets the minimum size, merging it forward
            if i in breakpoints and len(current) >= config.min_sentences_per_chunk:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        return [
            self._make_chunk(doc, " ".join(group), idx)
            for idx, group in enumerate(groups)
        ]

    def _make_chunk(self, doc: TextDocument, text: str, index: int) -> TextChunk:
        return TextChunk(
            id=uuid4(),
            document_id=doc.id,
            text=text,
            index=index,
            format=doc.format,
            metadata={"source": doc.source, "chunking_strategy": "semantic"},
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # linear interpolation between the two nearest ranks (numpy's default method)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)
