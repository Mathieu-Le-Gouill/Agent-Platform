from __future__ import annotations

import re
from uuid import uuid4

from agent_platform.components.base import Component
from agent_platform.components.embedder import Embedder
from agent_platform.components.semantic_chunker.config import SemanticChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.similarity import compute_similarity


class SemanticChunker(Component[list[TextDocument], list[TextChunk]]):
    def __init__(
        self,
        embedder: Embedder,
        config: SemanticChunkerConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or SemanticChunkerConfig()

    async def arun(self, input: list[TextDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for doc in input:
            chunks.extend(await self._chunk_document(doc))
        return chunks

    async def _chunk_document(self, doc: TextDocument) -> list[TextChunk]:
        sentences = [
            s.strip()
            for s in re.split(self._config.sentence_split_regex, doc.text)
            if s.strip()
        ]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(doc, sentences[0], 0)]

        embed_response = await self._embedder.arun(
            [TextChunk(text=s) for s in sentences]
        )
        vectors = [list(e.vector) for e in embed_response.embeddings]

        distances = [
            1.0 - compute_similarity(vectors[i], vectors[i + 1], self._config.metric)
            for i in range(len(vectors) - 1)
        ]
        threshold = _percentile(distances, self._config.breakpoint_percentile_threshold)
        breakpoints = {i for i, d in enumerate(distances) if d > threshold}

        groups: list[list[str]] = []
        current: list[str] = []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if (
                i in breakpoints
                and len(current) >= self._config.min_sentences_per_chunk
            ):
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
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)
