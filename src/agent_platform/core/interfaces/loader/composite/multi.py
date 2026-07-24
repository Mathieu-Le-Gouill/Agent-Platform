from collections.abc import AsyncIterator, Sequence

from agent_platform.core.interfaces.loader.composite.auto import AutoLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import Document


class MultiLoader:
    """Loads heterogeneous document sources (mixed media types) in one call."""

    def __init__(self) -> None:
        self._auto = AutoLoader()

    async def load(
        self, source: str, config: LoaderConfig | None = None
    ) -> Sequence[Document]:
        return await self._auto.load(source, config)

    async def load_many(
        self,
        sources: list[str],
        config: LoaderConfig | None = None,
    ) -> AsyncIterator[Sequence[Document]]:
        async for docs in self._auto.load_many(sources, config):
            yield docs
