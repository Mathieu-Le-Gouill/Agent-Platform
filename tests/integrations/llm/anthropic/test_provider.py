from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.batch import BatchRequest, BatchStatus
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, FinishReason, ImageFormat
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ImageBlock,
    Prompt,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.anthropic.provider import (
    AnthropicLLM,
    _block_to_native,
    _from_native_response,
    _to_native_messages,
    _to_native_params,
)


def _creds(key: str = "sk-ant-test") -> AnthropicCredentials:
    return AnthropicCredentials(api_key=SecretStr(key))


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _response(
    content: list[SimpleNamespace], input_tokens: int = 0, output_tokens: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class TestBlockToNative:
    def test_text_block(self):
        assert _block_to_native(TextBlock(text="hello")) == {
            "type": "text",
            "text": "hello",
        }

    def test_image_block_url(self):
        result = _block_to_native(ImageBlock(image="https://example.com/a.png"))
        assert result == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        }

    def test_image_block_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        result = _block_to_native(ImageBlock(image=doc))
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"]

    def test_audio_block_unsupported(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        with pytest.raises(ProviderError, match="does not support audio"):
            _block_to_native(AudioBlock(audio=doc))


class TestToNative:
    def test_system_message_extracted_separately(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        system, messages = _to_native_messages(prompt)
        assert system == "Be helpful."
        assert messages == []

    def test_multiple_system_messages_joined(self):
        prompt = Prompt(
            messages=[
                SystemMessage(content="First."),
                SystemMessage(content="Second."),
            ]
        )
        system, _ = _to_native_messages(prompt)
        assert system == "First.\n\nSecond."

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        _, messages = _to_native_messages(prompt)
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_without_tool_calls(self):
        prompt = Prompt(messages=[AssistantMessage(content="Hi there!")])
        _, messages = _to_native_messages(prompt)
        assert messages == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
        ]

    def test_assistant_message_with_tool_calls(self):
        prompt = Prompt().add_assistant(
            "",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"loc": "Paris"})
            ],
        )
        _, messages = _to_native_messages(prompt)
        assert messages[0]["role"] == "assistant"
        tool_use = messages[0]["content"][-1]
        assert tool_use == {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"loc": "Paris"},
        }

    def test_tool_message(self):
        prompt = Prompt(
            messages=[
                ToolMessage(
                    result=ToolResult(
                        tool_call_id="call_123",
                        name="get_weather",
                        content='{"temp": 72}',
                    )
                )
            ]
        )
        _, messages = _to_native_messages(prompt)
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": '{"temp": 72}',
                        "is_error": False,
                    }
                ],
            }
        ]

    def test_empty_prompt(self):
        system, messages = _to_native_messages(Prompt(messages=[]))
        assert system is None
        assert messages == []

    def test_multimodal_user_message(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        _, messages = _to_native_messages(prompt)
        assert messages[0]["content"] == [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        ]


class TestToNativeParams:
    def test_default_config(self):
        cfg = AnthropicGenerationConfig()
        result = _to_native_params(cfg)
        assert result == {"max_tokens": 1024, "temperature": 0.7}

    def test_with_max_tokens(self):
        cfg = AnthropicGenerationConfig(max_tokens=500)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 500

    def test_with_all_fields(self):
        cfg = AnthropicGenerationConfig(
            temperature=0.1,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            stop_sequences=["stop1"],
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.1
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["stop1"]

    def test_thinking(self):
        cfg = AnthropicGenerationConfig(thinking=True)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 8192
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_thinking_with_budget(self):
        cfg = AnthropicGenerationConfig(thinking=True, thinking_budget=10000)
        result = _to_native_params(cfg)
        assert result["thinking"]["budget_tokens"] == 10000
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_thinking_budget_exceeds_explicit_max_tokens(self):
        cfg = AnthropicGenerationConfig(thinking=True, max_tokens=1024)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 1024
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_effort(self):
        cfg = AnthropicGenerationConfig(effort="high")
        result = _to_native_params(cfg)
        assert result["output_config"] == {"effort": "high"}
        assert "thinking" not in result
        assert result["max_tokens"] == 1024

    def test_effort_takes_precedence_over_thinking(self):
        cfg = AnthropicGenerationConfig(effort="max", thinking=True)
        result = _to_native_params(cfg)
        assert result["output_config"] == {"effort": "max"}
        assert "thinking" not in result

    def test_cache_control(self):
        cfg = AnthropicGenerationConfig(cache_control=True)
        result = _to_native_params(cfg)
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_disabled_by_default(self):
        cfg = AnthropicGenerationConfig()
        result = _to_native_params(cfg)
        assert "cache_control" not in result

    def test_extra_params_applied(self):
        cfg = AnthropicGenerationConfig(extra_params={"metadata": {"user_id": "u1"}})
        result = _to_native_params(cfg)
        assert result["metadata"] == {"user_id": "u1"}

    def test_response_format_text_is_noop(self):
        cfg = AnthropicGenerationConfig(response_format=ResponseFormat.TEXT)
        result = _to_native_params(cfg)
        assert "output_config" not in result

    def test_response_format_json_raises(self):
        # Anthropic has no schema-less JSON mode; must raise, not silently drop.
        cfg = AnthropicGenerationConfig(response_format=ResponseFormat.JSON)
        with pytest.raises(ValueError, match="no native schema-less JSON"):
            _to_native_params(cfg)

    def test_response_format_json_schema_missing_raises(self):
        cfg = AnthropicGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_response_format_json_schema_wires_output_config(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cfg = AnthropicGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["output_config"] == {
            "format": {"type": "json_schema", "schema": schema}
        }

    def test_response_format_json_schema_merges_with_effort(self):
        schema = {"type": "object"}
        cfg = AnthropicGenerationConfig(
            effort="high",
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_native_params(cfg)
        assert result["output_config"] == {
            "effort": "high",
            "format": {"type": "json_schema", "schema": schema},
        }


class TestFromNative:
    def test_simple_message(self):
        response = _response(
            [_text_block("Hello world")], input_tokens=1, output_tokens=2
        )
        result = _from_native_response(response, model="claude-sonnet-4-6")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "claude-sonnet-4-6"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls(self):
        response = _response(
            [_tool_use_block("call_abc", "get_weather", {"location": "Paris"})]
        )
        result = _from_native_response(response, model="claude-3")
        assert result.message.content == ""
        assert len(result.message.tool_calls) == 1
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_mixed_text_and_tool_use(self):
        response = _response(
            [_text_block("Let me check."), _tool_use_block("1", "func_a", {"x": 1})]
        )
        result = _from_native_response(response, model="claude-3")
        assert result.message.content == "Let me check."
        assert len(result.message.tool_calls) == 1


class TestAnthropicLLMConstruction:
    def test_construct(self):
        provider = AnthropicLLM(_creds("sk-ant-123"))
        assert provider._credentials.api_key.get_secret_value() == "sk-ant-123"

    def test_default_config(self):
        provider = AnthropicLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, AnthropicGenerationConfig)
        assert cfg.model == "claude-sonnet-4-6"

    def test_tool_to_schema(self):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "test_tool"
            description = "A test tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        provider = AnthropicLLM(_creds())
        schema = provider._tool_to_schema(_SchemaTool())
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "input_schema" in schema
        assert "required" not in schema

    def test_missing_credential_error(self):
        provider = AnthropicLLM()
        provider._credentials = AnthropicCredentials()
        cfg = AnthropicGenerationConfig(model="claude-3")
        with pytest.raises(
            MissingCredentialError, match="Anthropic API key is required"
        ):
            provider._async_client(cfg)

    def test_client_kwargs_default_max_retries(self):
        provider = AnthropicLLM(_creds())
        cfg = AnthropicGenerationConfig(max_retries=None)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout_and_retries(self):
        provider = AnthropicLLM(_creds())
        cfg = AnthropicGenerationConfig(timeout=30.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 30.0
        assert kwargs["max_retries"] == 5


class TestAnthropicLLMGenerate:
    def test_sync_generate(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.Anthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _response(
            [_text_block("Sync hello")], input_tokens=1, output_tokens=1
        )

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        result = provider.generate(prompt, config=config)

        assert result.message.content == "Sync hello"
        assert result.model == "claude-sonnet-4-6"
        mock_client.messages.create.assert_called_once()

    def test_sync_generate_with_system_and_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.Anthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _response([_text_block("ok")])

        provider = AnthropicLLM(_creds())
        prompt = Prompt(
            messages=[SystemMessage(content="Be nice."), UserMessage(content="Hi")]
        )
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        provider.generate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["system"] == "Be nice."
        assert kwargs["tools"][0]["name"] == "search"

    async def test_agenerate(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=_response([_text_block("Async hello")])
        )

        provider = AnthropicLLM(_creds())
        prompt = Prompt(
            messages=[SystemMessage(content="Be nice."), UserMessage(content="Hi")]
        )
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        result = await provider.agenerate(prompt, config=config)

        assert result.message.content == "Async hello"
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["system"] == "Be nice."

    async def test_agenerate_with_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=_response([_text_block("ok")])
        )

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        await provider.agenerate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["tools"][0]["name"] == "search"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response([_text_block("ok")])

        mock_client.messages.create = flaky

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=AnthropicGenerationConfig(model="claude-sonnet-4-6")
        )
        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.messages.create = always_fails

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(
                prompt, config=AnthropicGenerationConfig(model="claude-sonnet-4-6")
            )


class TestAnthropicLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5)),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text=" World"),
            ),
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=10),
            ),
        ]

        async def _gen():
            for event in events:
                yield event

        mock_client.messages.create = AsyncMock(return_value=_gen())

        provider = AnthropicLLM(_creds())
        prompt = Prompt(
            messages=[SystemMessage(content="Be nice."), UserMessage(content="Hi")]
        )
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        results = [c async for c in provider.stream(prompt, config=config)]

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["system"] == "Be nice."
        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].delta == ""
        assert results[2].usage.output_tokens == 10
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_with_no_events(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        async def _gen():
            if False:
                yield

        mock_client.messages.create = AsyncMock(return_value=_gen())

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 1
        assert results[0].delta == ""
        assert results[0].finish_reason == FinishReason.STOP

    async def test_stream_with_tools_yields_tool_call_deltas(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        events = [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use", id="call_1", name="search"
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta", partial_json='{"query": "hi"}'
                ),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
        ]

        async def _gen():
            for event in events:
                yield event

        mock_client.messages.create = AsyncMock(return_value=_gen())

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        results = [
            c
            async for c in provider.stream(prompt, config=config, tools=[_SchemaTool()])
        ]

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["tools"][0]["name"] == "search"

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "search"
        assert deltas[1].arguments_delta == '{"query": "hi"}'


class TestAnthropicLLMBatch:
    async def test_submit_batch(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.batches.create = AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1",
                processing_status="in_progress",
                request_counts=SimpleNamespace(
                    processing=1, succeeded=0, errored=0, canceled=0, expired=0
                ),
            )
        )

        provider = AnthropicLLM(_creds())
        requests = [
            BatchRequest(
                custom_id="r1", prompt=Prompt(messages=[UserMessage(content="Hi")])
            )
        ]
        job = await provider.submit_batch(requests)

        assert job.id == "batch_1"
        assert job.status == BatchStatus.IN_PROGRESS
        assert job.request_count == 1
        assert job.completed_count == 0

        _, kwargs = mock_client.messages.batches.create.call_args
        assert kwargs["requests"][0]["custom_id"] == "r1"

    async def test_get_batch_status(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1", processing_status="ended", request_counts=None
            )
        )

        provider = AnthropicLLM(_creds())
        job = await provider.get_batch_status("batch_1")

        assert job.status == BatchStatus.COMPLETED
        assert job.request_count is None

    async def test_fetch_batch_results(self, mocker):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        succeeded_message = SimpleNamespace(
            content=[_text_block("hi")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="claude-sonnet-4-6",
        )
        entries = [
            SimpleNamespace(
                custom_id="r1",
                result=SimpleNamespace(type="succeeded", message=succeeded_message),
            ),
            SimpleNamespace(
                custom_id="r2",
                result=SimpleNamespace(type="errored", error="boom"),
            ),
        ]

        async def _gen():
            for entry in entries:
                yield entry

        mock_client.messages.batches.results = AsyncMock(return_value=_gen())

        provider = AnthropicLLM(_creds())
        results = await provider.fetch_batch_results("batch_1")

        assert results[0].custom_id == "r1"
        assert results[0].response is not None
        assert results[0].response.message.content == "hi"
        assert results[1].custom_id == "r2"
        assert results[1].error is not None
