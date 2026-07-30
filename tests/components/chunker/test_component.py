from unittest.mock import MagicMock

from agent_platform.components.chunker.component import Chunker, ChunkerInput
from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument


class TestChunker:
    async def test_arun_delegates_to_backend(self):
        backend = MagicMock()
        expected = [TextChunk(text="a"), TextChunk(text="b")]
        backend.chunk = MagicMock(return_value=expected)
        documents = [TextDocument(text="doc")]

        chunker = Chunker(backend=backend)
        result = await chunker.arun(ChunkerInput(documents, None))

        assert result == expected
        backend.chunk.assert_called_once_with(documents, None)

    async def test_arun_passes_config_through(self):
        backend = MagicMock()
        backend.chunk = MagicMock(return_value=[])
        config = ChunkerConfig(chunk_size=100)
        documents = [TextDocument(text="doc")]

        chunker = Chunker(backend=backend)
        await chunker.arun(ChunkerInput(documents, config))

        backend.chunk.assert_called_once_with(documents, config)

    async def test_arun_empty_documents(self):
        backend = MagicMock()
        backend.chunk = MagicMock(return_value=[])

        chunker = Chunker(backend=backend)
        result = await chunker.arun(ChunkerInput([], None))

        assert result == []
