from collections.abc import AsyncIterator, Sequence

import pytest

from agent_platform.components.ingestion.loader.component import Loader, LoaderInput
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import TextDocument


class _FakeLoader(BaseMediaLoader[TextDocument, LoaderConfig]):
    def __init__(self) -> None:
        self.calls: list[tuple[str, LoaderConfig | None]] = []

    async def load(
        self, source: str, config: LoaderConfig | None = None
    ) -> Sequence[TextDocument]:
        self.calls.append((source, config))
        return [TextDocument(text=f"{source}-page{i}") for i in range(2)]


class _FailingLoader(BaseMediaLoader[TextDocument, LoaderConfig]):
    async def load(
        self, source: str, config: LoaderConfig | None = None
    ) -> Sequence[TextDocument]:
        raise NotImplementedError

    async def load_many(
        self, sources: list[str], config: LoaderConfig | None = None
    ) -> AsyncIterator[Sequence[TextDocument]]:
        raise ProviderError("backend down")
        yield []


class TestLoader:
    async def test_flattens_documents_of_every_source_in_order(self):
        backend = _FakeLoader()
        config = LoaderConfig()

        result = await Loader(backend).arun((("a.pdf", "b.pdf"), config))

        assert [doc.text for doc in result] == [
            "a.pdf-page0",
            "a.pdf-page1",
            "b.pdf-page0",
            "b.pdf-page1",
        ]
        assert backend.calls == [("a.pdf", config), ("b.pdf", config)]

    async def test_forwards_none_config(self):
        backend = _FakeLoader()

        await Loader(backend).arun((["a.pdf"], None))

        assert backend.calls == [("a.pdf", None)]

    async def test_no_sources_returns_empty_list(self):
        assert await Loader(_FakeLoader()).arun(([], None)) == []

    async def test_accepts_a_named_input(self):
        backend = _FakeLoader()

        await Loader(backend).arun(LoaderInput(sources=["a.pdf"], config=None))

        assert backend.calls == [("a.pdf", None)]

    async def test_backend_error_propagates(self):
        with pytest.raises(ProviderError, match="backend down"):
            await Loader(_FailingLoader()).arun((["a.pdf"], None))
