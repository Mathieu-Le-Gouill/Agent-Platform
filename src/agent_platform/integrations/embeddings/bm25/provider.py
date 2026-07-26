from __future__ import annotations

import re
import zlib
from collections import Counter

from agent_platform.core.interfaces.embeddings.sparse import BaseSparseEmbeddingProvider
from agent_platform.core.schemas.vector import SparseVector
from agent_platform.integrations.embeddings.bm25.config import BM25Config

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class BM25SparseEmbedder(BaseSparseEmbeddingProvider[BM25Config]):
    """Raw term-frequency sparse vectors via feature hashing.

    No corpus-fit step: each token is hashed into a fixed-size vocabulary
    space, so the same token always lands on the same index whether this text
    is being indexed or queried. IDF weighting is left to the vector store
    (e.g. Qdrant's `Modifier.IDF`), since it needs corpus-wide statistics this
    stateless, per-text embedder deliberately doesn't try to replicate.
    """

    def _default_config(self) -> BM25Config:
        return BM25Config()

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _hash_token(self, token: str, vocabulary_size: int) -> int:
        return zlib.crc32(token.encode("utf-8")) % vocabulary_size

    async def embed_sparse(
        self,
        text: str,
        config: BM25Config | None = None,
    ) -> SparseVector:
        config = config or self._default_config()
        tokens = self._tokenize(text)
        counts = Counter(
            self._hash_token(token, config.vocabulary_size) for token in tokens
        )
        if not counts:
            return SparseVector(indices=[], values=[])

        indices = sorted(counts)
        values = [float(counts[index]) for index in indices]
        return SparseVector(indices=indices, values=values)
