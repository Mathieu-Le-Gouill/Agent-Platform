from typing import AsyncIterator, Protocol
from agent_platform.models.document import Document


class BaseLoader(Protocol):
    """Loads raw source material and returns a populated Document."""

    async def load(self, source: str, **kwargs) -> Document: ...

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Document]: ...