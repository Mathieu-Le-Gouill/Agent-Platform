from collections.abc import Sequence

from agent_platform.components.chunker.component import (
    Chunker,
    ChunkerConfigT,
    ChunkerInput,
)
from agent_platform.components.embedder.component import (
    EmbedConfigT,
    Embedder,
    EmbedderInput,
)
from agent_platform.components.loader.component import Loader, LoaderConfigT
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.interfaces.vector_store.port import VectorStore


async def ingest(
    sources: Sequence[str],
    loader: Loader[LoaderConfigT],
    chunker: Chunker[ChunkerConfigT],
    embedder: Embedder[EmbedConfigT],
    store: VectorStore,
    loader_config: LoaderConfigT | None = None,
    chunker_config: ChunkerConfigT | None = None,
    embedding_config: EmbedConfigT | None = None,
    store_config: VectorStoreConfig | None = None,
) -> None:
    documents = await loader.arun((sources, loader_config))
    chunks = await chunker.arun(ChunkerInput(documents, chunker_config))
    embedding_response = await embedder.arun(EmbedderInput(chunks, embedding_config))
    vectors = [embedding.to_list() for embedding in embedding_response.embeddings]
    await store.add(chunks, vectors, config=store_config)
