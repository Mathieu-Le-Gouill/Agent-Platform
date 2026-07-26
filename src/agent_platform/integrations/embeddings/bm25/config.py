from __future__ import annotations

from agent_platform.core.config import ProviderConfig


class BM25Config(ProviderConfig):
    # Feature-hashing space: each token is hashed into [0, vocabulary_size),
    # so no corpus-level vocabulary fitting is needed before embedding a text.
    vocabulary_size: int = 2**18
