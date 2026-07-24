from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage as LCAIMessage,
)
from langchain_core.messages import (
    HumanMessage as LCHumanMessage,
)
from langchain_core.messages import (
    SystemMessage as LCSystemMessage,
)
from langchain_core.messages import (
    ToolMessage as LCToolMessage,
)

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
)
from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, FinishReason, ImageFormat
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ImageBlock,
    Prompt,
    SystemMessage,
    TextBlock,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.llm.langchain_base import (
    LangChainLLMProvider,
    _block_to_langchain,
    _content_to_langchain,
    _from_langchain,
    _to_langchain,
)


class TestBlockToLangchain:
    def test_text_block(self):
        result = _block_to_langchain(TextBlock(text="hello"))
        assert result == {"type": "text", "text": "hello"}

    def test_image_block_url(self):
        result = _block_to_langchain(ImageBlock(image="https://example.com/a.png"))
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
        }

    def test_image_block_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        result = _block_to_langchain(ImageBlock(image=doc))
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_block_document_unknown_format_mime(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.UNKNOWN)
        result = _block_to_langchain(ImageBlock(image=doc))
        assert result["image_url"]["url"].startswith("data:image/unknown;base64,")

    def test_audio_block(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        result = _block_to_langchain(AudioBlock(audio=doc))
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "wav"
        assert result["input_audio"]["data"]


class TestContentToLangchain:
    def test_plain_string_content_passthrough(self):
        msg = UserMessage(content="hi there")
        assert _content_to_langchain(msg) == "hi there"

    def test_block_list_content_maps_each_block(self):
        msg = UserMessage(
            content=[
                TextBlock(text="describe this"),
                ImageBlock(image="https://x/y.png"),
            ]
        )
        result = _content_to_langchain(msg)
        assert result == [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]

    def test_multimodal_message_reaches_to_langchain(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        result = _to_langchain(prompt)
        assert len(result) == 1
        assert result[0].content == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]


class TestToLangchain:
    def test_system_message(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        result = _to_langchain(prompt)
        assert len(result) == 1
        assert isinstance(result[0], LCSystemMessage)
        assert result[0].content == "Be helpful."

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        result = _to_langchain(prompt)
        assert len(result) == 1
        assert isinstance(result[0], LCHumanMessage)
        assert result[0].content == "Hello"

    def test_assistant_message(self):
        prompt = Prompt(messages=[AssistantMessage(content="Hi there!")])
        result = _to_langchain(prompt)
        assert len(result) == 1
        assert isinstance(result[0], LCAIMessage)
        assert result[0].content == "Hi there!"

    def test_tool_message(self):
        prompt = Prompt(
            messages=[
                ToolMessage(
                    result=ToolResult(
                        tool_call_id="call_123",
                        name="get_weather",
                        content='{"temp": 72}',
                    ),
                ),
            ],
        )
        result = _to_langchain(prompt)
        assert len(result) == 1
        assert isinstance(result[0], LCToolMessage)
        assert result[0].content == '{"temp": 72}'
        assert result[0].tool_call_id == "call_123"

    def test_mixed_messages(self):
        prompt = Prompt(
            messages=[
                SystemMessage(content="System prompt"),
                UserMessage(content="User query"),
                AssistantMessage(content="Assistant reply"),
            ],
        )
        result = _to_langchain(prompt)
        assert len(result) == 3
        assert isinstance(result[0], LCSystemMessage)
        assert isinstance(result[1], LCHumanMessage)
        assert isinstance(result[2], LCAIMessage)

    def test_empty_prompt(self):
        prompt = Prompt(messages=[])
        result = _to_langchain(prompt)
        assert result == []


class TestFromLangchain:
    def test_simple_message(self):
        lc_msg = LCAIMessage(content="Hello world")
        result = _from_langchain(lc_msg, model="gpt-4")
        assert isinstance(result, LLMResponse)
        assert result.message is not None
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "gpt-4"
        assert result.finish_reason == FinishReason.STOP
        assert isinstance(result.usage, TokenUsage)

    def test_with_tool_calls(self):
        lc_msg = LCAIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                    "type": "tool_call",
                },
            ],
        )
        result = _from_langchain(lc_msg, model="claude-3")
        assert result.message is not None
        assert result.message.content == ""
        assert len(result.message.tool_calls) == 1
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_with_usage_metadata(self):
        lc_msg = LCAIMessage(
            content="Done",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        )
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_without_usage_metadata(self):
        lc_msg = LCAIMessage(content="No usage")
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_empty_tool_calls_list(self):
        lc_msg = LCAIMessage(content="Hello", tool_calls=[])
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message is not None
        assert result.message.tool_calls == []
        assert result.message.content == "Hello"

    def test_multiple_tool_calls(self):
        lc_msg = LCAIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "func_a", "args": {"x": 1}, "type": "tool_call"},
                {"id": "2", "name": "func_b", "args": {"y": 2}, "type": "tool_call"},
            ],
        )
        result = _from_langchain(lc_msg, model="gpt-4")
        assert len(result.message.tool_calls) == 2
        assert result.message.tool_calls[1].name == "func_b"

    def test_null_tool_calls(self):
        lc_msg = LCAIMessage(content="Hi")
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message.tool_calls == []

    def test_content_as_list_of_mixed_parts(self):
        lc_msg = LCAIMessage(
            content=["Hello", " ", {"text": "World"}],
            tool_calls=[],
        )
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message.content == "Hello World"

    def test_content_as_list_with_missing_text_keys(self):
        lc_msg = LCAIMessage(
            content=[{"type": "text", "text": "Hi"}, {"type": "image"}],
            tool_calls=[],
        )
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message.content == "Hi"

    def test_content_as_list_of_only_dicts(self):
        lc_msg = LCAIMessage(
            content=[{"text": "part1"}, {"text": "part2"}],
            tool_calls=[],
        )
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message.content == "part1part2"

    def test_content_as_empty_list(self):
        lc_msg = LCAIMessage(content=[], tool_calls=[])
        result = _from_langchain(lc_msg, model="gpt-4")
        assert result.message.content == ""


class _TestLLMProvider(LangChainLLMProvider):
    def _client(self, config):
        raise NotImplementedError

    def _tool_to_schema(self, tool):
        raise NotImplementedError

    def _default_config(self):
        return GenerationConfig()


class TestLangChainLLMProviderStream:
    async def test_stream_with_string_content(self):
        chunk1 = MagicMock(spec=[])
        chunk1.content = "Hello"
        chunk1.usage_metadata = None
        chunk2 = MagicMock(spec=[])
        chunk2.content = " World"
        chunk2.usage_metadata = None

        async def _gen():
            yield chunk1
            yield chunk2

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 3
        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].delta == ""
        assert results[2].finish_reason == FinishReason.STOP


class TestLangChainLLMProviderAgenerateRetryAndTranslation:
    async def test_retries_transient_failure_then_succeeds(self, monkeypatch):
        import agent_platform.core.errors as errors_mod

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

        calls = {"n": 0}
        lc_response = LCAIMessage(content="ok")

        mock_model = MagicMock()

        async def flaky_ainvoke(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient network error")
            return lc_response

        mock_model.ainvoke = flaky_ainvoke

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=GenerationConfig(model="gpt-4")
        )

        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_translates_permanent_failure_to_provider_error(self, monkeypatch):
        import agent_platform.core.errors as errors_mod
        from agent_platform.core.errors import ProviderError

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

        mock_model = MagicMock()

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_model.ainvoke = always_fails

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(prompt, config=GenerationConfig(model="gpt-4"))

    async def test_stream_skips_empty_string_content(self):
        chunk1 = MagicMock(spec=[])
        chunk1.content = ""
        chunk1.usage_metadata = None
        chunk2 = MagicMock(spec=[])
        chunk2.content = "Actual"
        chunk2.usage_metadata = None

        async def _gen():
            yield chunk1
            yield chunk2

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 2
        assert results[0].delta == "Actual"
        assert results[1].delta == ""
        assert results[1].finish_reason == FinishReason.STOP

    async def test_stream_with_content_as_list_of_items(self):
        chunk = MagicMock(spec=[])
        chunk.content = ["Hello", {"text": " "}, "World"]
        chunk.usage_metadata = None

        async def _gen():
            yield chunk

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 4
        assert results[0].delta == "Hello"
        assert results[1].delta == " "
        assert results[2].delta == "World"
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_skips_empty_strings_in_list(self):
        chunk = MagicMock(spec=[])
        chunk.content = ["A", "", "B", {"text": ""}, "C"]
        chunk.usage_metadata = None

        async def _gen():
            yield chunk

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 4
        assert [r.delta for r in results[:3]] == ["A", "B", "C"]
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_with_usage_metadata_yields_finish_reason(self):
        chunk = MagicMock(spec=[])
        chunk.content = "Done"
        chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 10}

        async def _gen():
            yield chunk

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 3
        assert results[0].delta == "Done"
        assert results[1].delta == ""
        assert results[1].finish_reason == FinishReason.STOP
        assert results[1].usage is not None
        assert results[1].usage.input_tokens == 5
        assert results[1].usage.output_tokens == 10
        assert results[2].delta == ""
        assert results[2].finish_reason == FinishReason.STOP

    async def test_stream_usage_only_with_empty_content(self):
        chunk = MagicMock(spec=[])
        chunk.content = ""
        chunk.usage_metadata = {"input_tokens": 3, "output_tokens": 6}

        async def _gen():
            yield chunk

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 2
        assert results[0].delta == ""
        assert results[0].finish_reason == FinishReason.STOP
        assert results[1].delta == ""
        assert results[1].finish_reason == FinishReason.STOP

    async def test_stream_ends_with_final_stop_chunk(self):
        async def _gen():
            if False:
                yield

        mock_model = MagicMock()
        mock_model.astream = MagicMock(return_value=_gen())

        provider = _TestLLMProvider()
        provider._client = MagicMock(return_value=mock_model)

        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = GenerationConfig(model="gpt-4")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 1
        assert results[0].delta == ""
        assert results[0].finish_reason == FinishReason.STOP
