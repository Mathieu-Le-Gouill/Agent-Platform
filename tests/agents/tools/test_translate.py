from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import ToolError, TranslateInput, TranslateTool
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language


@pytest.fixture
def mock_translator():
    translator = AsyncMock()
    translator.translate = AsyncMock(return_value=TextChunk(text="Bonjour"))
    return translator


@pytest.fixture
def tool(mock_translator):
    return TranslateTool(translator=mock_translator)


class TestTranslateInput:
    def test_valid_input(self):
        inp = TranslateInput(text="Hello", target=Language.FR)
        assert inp.text == "Hello"
        assert inp.target == Language.FR
        assert inp.source is None

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            TranslateInput(text="", target=Language.FR)

    def test_missing_target_raises(self):
        with pytest.raises(ValidationError):
            TranslateInput(text="Hello")


class TestTranslateTool:
    def test_name_and_description(self, tool):
        assert tool.name == "translate"
        assert tool.description

    async def test_run_success(self, tool, mock_translator):
        result = await tool.run(text="Hello", target=Language.FR)
        assert result.text == "Bonjour"
        mock_translator.translate.assert_awaited_once()
        call = mock_translator.translate.await_args
        assert call.args[0].text == "Hello"
        assert call.kwargs == {"target": Language.FR, "source": None}

    async def test_run_with_source(self, tool, mock_translator):
        await tool.run(text="Hello", target=Language.FR, source=Language.EN)
        call = mock_translator.translate.await_args
        assert call.kwargs == {"target": Language.FR, "source": Language.EN}

    async def test_run_provider_error_wrapped(self, tool, mock_translator):
        mock_translator.translate = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ToolError, match="Translation failed"):
            await tool.run(text="Hello", target=Language.FR)

    async def test_run_missing_target_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(text="Hello")
