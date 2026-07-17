import pytest
from typing import Sequence

from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.schemas.document import Document
from agent_platform.core.schemas.chunk import TextChunk


class TestBaseMediaLoaderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseMediaLoader()

    def test_abstract_methods(self):
        assert "load" in BaseMediaLoader.__abstractmethods__

    def test_load_many_yields_sequence(self):
        class ConcreteLoader(BaseMediaLoader):
            async def load(self, source: str, config=None):
                return []

        loader = ConcreteLoader()
        import inspect

        assert inspect.iscoroutinefunction(loader.load)

    @pytest.mark.asyncio
    async def test_load_many_iterates_sources(self):
        class TestLoader(BaseMediaLoader):
            async def load(self, source: str, config=None) -> Sequence[TextChunk]:
                return [TextChunk(text=source)]

        loader = TestLoader()
        results = [docs async for docs in loader.load_many(["a", "b", "c"])]
        assert len(results) == 3
        assert results[0][0].text == "a"
        assert results[1][0].text == "b"
        assert results[2][0].text == "c"

    @pytest.mark.asyncio
    async def test_load_many_skips_exceptions(self):
        class FailingLoader(BaseMediaLoader):
            async def load(self, source: str, config=None) -> Sequence[TextChunk]:
                if source == "fail":
                    raise ValueError("Failed to load")
                return [TextChunk(text=source)]

        loader = FailingLoader()
        results = [docs async for docs in loader.load_many(["ok", "fail", "also_ok"])]
        assert len(results) == 2
        assert results[0][0].text == "ok"
        assert results[1][0].text == "also_ok"
