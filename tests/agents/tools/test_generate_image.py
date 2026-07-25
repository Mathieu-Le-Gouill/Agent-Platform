from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import GenerateImageInput, GenerateImageTool, ToolError
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat


@pytest.fixture
def mock_generator():
    generator = AsyncMock()
    generator.generate = AsyncMock(
        return_value=ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
    )
    return generator


@pytest.fixture
def tool(mock_generator):
    return GenerateImageTool(generator=mock_generator)


class TestGenerateImageInput:
    def test_valid_input(self):
        inp = GenerateImageInput(prompt="a cat")
        assert inp.prompt == "a cat"
        assert inp.size is None
        assert inp.format == ImageFormat.PNG

    def test_empty_prompt_raises(self):
        with pytest.raises(ValidationError):
            GenerateImageInput(prompt="")


class TestGenerateImageTool:
    def test_name_and_description(self, tool):
        assert tool.name == "generate_image"
        assert tool.description

    async def test_run_success(self, tool, mock_generator):
        result = await tool.run(prompt="a cat")
        assert result.format == ImageFormat.PNG
        mock_generator.generate.assert_awaited_once_with(
            "a cat", size=None, format=ImageFormat.PNG
        )

    async def test_run_with_size_and_format(self, tool, mock_generator):
        await tool.run(prompt="a cat", size="512x512", format=ImageFormat.JPEG)
        mock_generator.generate.assert_awaited_once_with(
            "a cat", size="512x512", format=ImageFormat.JPEG
        )

    async def test_run_provider_error_wrapped(self, tool, mock_generator):
        mock_generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ToolError, match="Image generation failed"):
            await tool.run(prompt="a cat")

    async def test_run_missing_prompt_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run()
