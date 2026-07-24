from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class JinaRerankerConfig(RerankerConfig):
    # Jina reranker model id, e.g. jina-reranker-v2-base-multilingual (jina.ai/reranker).
    model: str = "jina-reranker-v2-base-multilingual"

    # NOTE: langchain_community.document_compressors.jina_rerank.JinaRerank
    # (installed version) has no truncation/return_documents fields at all
    # -- only session/top_n/model/jina_api_key/user_agent. Library gap: not
    # exposed as config fields since the wrapper has nowhere to route them.


# sources: https://jina.ai/reranker/
