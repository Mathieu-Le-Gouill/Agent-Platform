from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import TextDocument

LoaderConfigT = TypeVar("LoaderConfigT", bound=LoaderConfig)

LoaderInput = tuple[Sequence[str], LoaderConfigT | None]


class Loader(
    Component[LoaderInput[LoaderConfigT], list[TextDocument]],
    Generic[LoaderConfigT],
):
    def __init__(
        self,
        backend: BaseMediaLoader[TextDocument, LoaderConfigT],
    ) -> None:
        self._backend = backend

    async def arun(self, input: LoaderInput[LoaderConfigT]) -> list[TextDocument]:
        sources, config = input
        documents: list[TextDocument] = []
        async for batch in self._backend.load_many(list(sources), config):
            documents.extend(batch)
        return documents
