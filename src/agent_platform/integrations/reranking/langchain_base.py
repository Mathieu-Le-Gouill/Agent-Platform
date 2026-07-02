from abc import abstractmethod
from typing import Sequence

from langchain_core.documents import (
    BaseDocumentCompressor,
    Document as LC_Document,
)

from agent_platform.integrations.reranking.base import BaseReranker
from agent_platform.integrations.reranking.config import RerankerConfig

from agent_platform.models.chunk import TextChunk

from agent_platform.bridges.langchain.chunk import (
    to_langchain as chunk_to_lc,
    from_langchain_many as chunks_from_lc,
)


class LangChainReranker(BaseReranker[TextChunk]):

    @abstractmethod
    def _client(self) -> BaseDocumentCompressor:
        ...


    async def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: RerankerConfig = RerankerConfig(),
    ) -> Sequence[TextChunk]:
        
        client = self._client()
        
        documents: list[LC_Document] = [chunk_to_lc(item) for item in items]
        
        result = await client.acompress_documents(documents, query)
        
        if config is not None and config.top_k is not None:
            result = result[: config.top_k]
            
        return chunks_from_lc(result)