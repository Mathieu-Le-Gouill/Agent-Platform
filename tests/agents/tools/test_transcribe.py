from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import TranscribeInput, TranscribeTool, ToolError
from agent_platform.models.conversation import Transcript, Utterance


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.transcribe = AsyncMock(
        return_value=Transcript(
            id=uuid4(),
            utterances=[Utterance(text="hello world", confidence=0.95)],
        )
    )
    return provider


@pytest.fixture
def tool(mock_provider):
    return TranscribeTool(provider=mock_provider)


class TestTranscribeInput:
    def test_valid_input(self):
        inp = TranscribeInput(data=b"\x00\x01", sample_rate=16000, channels=1)
        assert inp.data == b"\x00\x01"
        assert inp.sample_rate == 16000

    def test_sample_rate_too_low(self):
        with pytest.raises(ValidationError):
            TranscribeInput(data=b"\x00\x01", sample_rate=500)

    def test_sample_rate_too_high(self):
        with pytest.raises(ValidationError):
            TranscribeInput(data=b"\x00\x01", sample_rate=200000)

    def test_channels_too_low(self):
        with pytest.raises(ValidationError):
            TranscribeInput(data=b"\x00\x01", channels=0)

    def test_channels_too_high(self):
        with pytest.raises(ValidationError):
            TranscribeInput(data=b"\x00\x01", channels=16)

    def test_defaults(self):
        inp = TranscribeInput(data=b"\x00\x01")
        assert inp.sample_rate == 16000
        assert inp.channels == 1


class TestTranscribeTool:
    def test_name_and_description(self, tool):
        assert tool.name == "transcribe"
        assert tool.description

    def test_input_schema(self, tool):
        assert tool.input_schema is TranscribeInput

    def test_output_schema(self, tool):
        assert tool.output_schema is Transcript

    @pytest.mark.asyncio
    async def test_run_success(self, tool, mock_provider):
        result = await tool.run(data=b"\x00\x01")
        assert isinstance(result, Transcript)
        assert len(result.utterances) == 1
        assert result.utterances[0].text == "hello world"
        mock_provider.transcribe.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_sample_rate(self, tool, mock_provider):
        await tool.run(data=b"\x00\x01", sample_rate=44100)
        call_args = mock_provider.transcribe.call_args[0][0]
        assert call_args.sample_rate == 44100

    @pytest.mark.asyncio
    async def test_run_with_channels(self, tool, mock_provider):
        await tool.run(data=b"\x00\x01", channels=2)
        call_args = mock_provider.transcribe.call_args[0][0]
        assert call_args.channels == 2

    @pytest.mark.asyncio
    async def test_run_empty_utterances_raises(self, tool, mock_provider):
        mock_provider.transcribe = AsyncMock(
            return_value=Transcript(id=uuid4(), utterances=[])
        )
        with pytest.raises(ToolError, match="no utterances"):
            await tool.run(data=b"\x00\x01")

    @pytest.mark.asyncio
    async def test_run_invalid_input_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(data=b"\x00\x01", sample_rate=-1)

    @pytest.mark.asyncio
    async def test_run_provider_error_wrapped(self, tool, mock_provider):
        mock_provider.transcribe = AsyncMock(
            side_effect=RuntimeError("provider failure")
        )
        with pytest.raises(ToolError, match="Speech-to-text failed"):
            await tool.run(data=b"\x00\x01")

    @pytest.mark.asyncio
    async def test_run_missing_data_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(sample_rate=16000)
