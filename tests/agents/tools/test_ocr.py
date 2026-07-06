from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import OCRInput, OCRTool, ToolError
from agent_platform.models.chunk import TextChunk


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.extract = AsyncMock(
        return_value=[
            TextChunk(id=uuid4(), text="Hello", index=0, metadata={"confidence": 0.95}),
            TextChunk(id=uuid4(), text="World", index=1, metadata={"confidence": 0.87}),
        ]
    )
    return provider


@pytest.fixture
def tool(mock_provider):
    return OCRTool(provider=mock_provider)


class TestOCRInput:
    def test_valid_input(self):
        inp = OCRInput(source="/path/to/image.png")
        assert inp.source == "/path/to/image.png"
        assert inp.language == "eng"
        assert inp.min_confidence == 0.0

    def test_min_confidence_range(self):
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", min_confidence=1.5)
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", min_confidence=-0.1)

    def test_language_too_short(self):
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", language="")

    def test_language_too_long(self):
        with pytest.raises(ValidationError):
            OCRInput(source="img.png", language="toolonglang")

    def test_defaults(self):
        inp = OCRInput(source="img.png")
        assert inp.language == "eng"
        assert inp.min_confidence == 0.0


class TestOCRTool:
    def test_name_and_description(self, tool):
        assert tool.name == "ocr"
        assert tool.description

    def test_input_schema(self, tool):
        assert tool.input_schema is OCRInput

    @pytest.mark.asyncio
    async def test_run_success(self, tool, mock_provider):
        results = await tool.run(source="/path/to/image.png")
        assert len(results) == 2
        assert results[0].text == "Hello"
        assert results[1].text == "World"
        mock_provider.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_language(self, tool, mock_provider):
        await tool.run(source="/path/to/image.png", language="fra")
        call_config = mock_provider.extract.call_args[1].get("config")
        assert call_config is not None
        assert call_config.language == "fra"

    @pytest.mark.asyncio
    async def test_run_with_min_confidence(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(
            return_value=[
                TextChunk(
                    id=uuid4(),
                    text="Low",
                    index=0,
                    metadata={"confidence": 0.3},
                ),
                TextChunk(
                    id=uuid4(),
                    text="High",
                    index=1,
                    metadata={"confidence": 0.9},
                ),
            ]
        )
        results = await tool.run(source="/path/to/image.png", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].text == "High"

    @pytest.mark.asyncio
    async def test_run_empty_results(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(return_value=[])
        results = await tool.run(source="/path/to/image.png")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_all_filtered_out(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(
            return_value=[
                TextChunk(
                    id=uuid4(),
                    text="Low",
                    index=0,
                    metadata={"confidence": 0.1},
                ),
            ]
        )
        results = await tool.run(source="/path/to/image.png", min_confidence=0.5)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_provider_error_wrapped(self, tool, mock_provider):
        mock_provider.extract = AsyncMock(side_effect=RuntimeError("ocr failed"))
        with pytest.raises(ToolError, match="OCR extraction failed"):
            await tool.run(source="/path/to/image.png")

    @pytest.mark.asyncio
    async def test_run_missing_source_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(language="eng")
