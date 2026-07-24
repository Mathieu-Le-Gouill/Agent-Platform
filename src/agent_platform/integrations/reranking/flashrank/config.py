from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class FlashRankConfig(RerankerConfig):
    # FlashRank model name, e.g. ms-marco-MiniLM-L-12-v2 (github.com/PrithivirajDamodaran/FlashRank).
    model: str = "ms-marco-MiniLM-L-12-v2"
    # Directory where FlashRank downloads/caches model files; defaults to the library's own cache dir when None.
    cache_dir: str | None = None
    # The max token length passed to the tokenizer/model.
    max_length: int | None = None

    # NOTE: inherited batch_size has no FlashRank equivalent (Ranker.rerank
    # scores the whole passage list in one call) and is not forwarded.


# sources: https://github.com/PrithivirajDamodaran/FlashRank
