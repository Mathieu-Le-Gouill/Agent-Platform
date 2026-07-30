from typing import Any

from agent_platform.components.embedder.component import EmbedConfigT, Embedder
from agent_platform.components.generator.component import (
    GenerationConfigT,
    Generator,
)
from agent_platform.components.reranker.component import Reranker, RerankerConfigT
from agent_platform.components.vector_search.component import VectorSearch
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.message import Prompt


async def query(
    query_text: str,
    embedder: Embedder[EmbedConfigT],
    vector_search: VectorSearch,
    reranker: Reranker[TextChunk, RerankerConfigT],
    generator: Generator[GenerationConfigT],
    k: int = 5,
    filter: dict[str, Any] | None = None,
    embedding_config: EmbedConfigT | None = None,
    search_config: VectorStoreConfig | None = None,
    reranker_config: RerankerConfigT | None = None,
    generation_config: GenerationConfigT | None = None,
) -> LLMResponse:
    query_chunk = TextChunk(text=query_text, index=0)
    embedding = await embedder.arun(([query_chunk], embedding_config))
    vector = embedding.embeddings[0].to_list()

    results = await vector_search.arun((vector, k, filter, search_config))
    chunks = [chunk for chunk, _score in results]

    reranked = await reranker.arun((query_text, chunks, reranker_config))
    context = "\n\n".join(chunk.text for chunk in reranked)

    prompt = Prompt.build(
        system=f"Answer the question using only this context:\n{context}",
        user=query_text,
    )
    return await generator.arun((prompt, generation_config))
