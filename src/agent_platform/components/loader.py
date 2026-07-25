from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import TextDocument

LoaderConfigT = TypeVar("LoaderConfigT", bound=LoaderConfig)


class Loader(
    Component[Sequence[str], list[TextDocument]],
    Generic[LoaderConfigT],
):
    def __init__(
        self,
        backend: BaseMediaLoader[TextDocument, LoaderConfigT],
        config: LoaderConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: Sequence[str]) -> list[TextDocument]:
        documents: list[TextDocument] = []
        async for batch in self._backend.load_many(list(input), self._config):
            documents.extend(batch)
        return documents
