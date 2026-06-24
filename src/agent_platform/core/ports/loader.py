from typing import AsyncIterator, Protocol
from core.entities.document import Document


class LoaderPort(Protocol):
    """Loads raw source material and returns a populated Document."""

    async def load(self, source: str, **kwargs) -> Document: ...

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Document]: ...