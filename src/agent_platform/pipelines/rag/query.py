from typing import Any

from agent_platform.components.embedder.component import Embedder
from agent_platform.components.generator.component import Generator
from agent_platform.components.reranker.component import Reranker
from agent_platform.components.vector_search.component import VectorSearch
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.message import Prompt


async def query(
    query_text: str,
    embedder: Embedder,
    vector_search: VectorSearch,
    reranker: Reranker,
    generator: Generator,
    k: int = 5,
    filter: dict[str, Any] | None = None,
) -> LLMResponse:
    query_chunk = TextChunk(text=query_text, index=0)
    embedding = await embedder.arun([query_chunk])
    vector = embedding.embeddings[0].to_list()

    results = await vector_search.arun((vector, k, filter))
    chunks = [chunk for chunk, _score in results]

    reranked = await reranker.arun((query_text, chunks))
    context = "\n\n".join(chunk.text for chunk in reranked)

    prompt = Prompt.build(
        system=f"Answer the question using only this context:\n{context}",
        user=query_text,
    )
    return await generator.arun(prompt)
