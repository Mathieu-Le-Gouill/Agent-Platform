from collections.abc import Sequence

from agent_platform.components.chunker import Chunker
from agent_platform.components.loader import Loader
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.interfaces.vector_store.port import VectorStore


async def ingest(
    sources: Sequence[str],
    loader: Loader,
    chunker: Chunker,
    store: VectorStore,
    config: VectorStoreConfig | None = None,
) -> None:
    documents = await loader.arun(sources)
    chunks = await chunker.arun(documents)
    await store.add(chunks, config=config)
