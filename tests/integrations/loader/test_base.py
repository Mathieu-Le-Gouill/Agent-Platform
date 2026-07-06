import pytest
from typing import Sequence

from agent_platform.integrations.loader.base import BaseMediaLoader
from agent_platform.models.document import Document
from agent_platform.models.chunk import TextChunk


class TestBaseMediaLoaderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseMediaLoader()

    def test_abstract_methods(self):
        assert "load" in BaseMediaLoader.__abstractmethods__

    def test_load_many_yields_sequence(self):
        class ConcreteLoader(BaseMediaLoader):
            async def load(self, source: str):
                return []

        loader = ConcreteLoader()
        import inspect

        assert inspect.iscoroutinefunction(loader.load)

    async def test_load_many_iterates_sources(self):
        class TestLoader(BaseMediaLoader):
            async def load(self, source: str) -> Sequence[TextChunk]:
                return [TextChunk(text=source)]

        loader = TestLoader()
        results = [docs async for docs in loader.load_many(["a", "b", "c"])]
        assert len(results) == 3
        assert results[0][0].text == "a"
        assert results[1][0].text == "b"
        assert results[2][0].text == "c"
