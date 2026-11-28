import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.core.credentials import ClientOptions
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.batch import BatchRequest, BatchStatus
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas import model_schema
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
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.anthropic.mappers import (
    block_to_native as _block_to_native,
)
from agent_platform.integrations.llm.anthropic.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.anthropic.mappers import (
    to_native_messages as _to_native_messages,
)
from agent_platform.integrations.llm.anthropic.mappers import (
    to_native_params as _to_native_params,
)
from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM


def _creds(key: str = "sk-ant-test") -> AnthropicCredentials:
    return AnthropicCredentials(api_key=SecretStr(key))


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _response(
    content: list[SimpleNamespace],
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str | None = "end_turn",
    id: str = "msg-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        content=content,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
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

    def test_image_block_unknown_format_falls_back_to_png(self):
        doc = ImageDocument(content=b"\x89PNG")
        result = _block_to_native(ImageBlock(image=doc))
        assert result["source"]["media_type"] == "image/png"

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
        # Fits the 4096-token cap of Claude 3 models and finishes within
        # the default 120 s timeout of a non-streaming call.
        assert result == {"max_tokens": 4096}

    def test_with_max_tokens(self):
        cfg = AnthropicGenerationConfig(max_tokens=500)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 500

    def test_with_all_fields(self):
        cfg = AnthropicGenerationConfig(
            model="claude-sonnet-4-6",
            temperature=0.1,
            max_tokens=200,
            top_k=40,
            stop_sequences=["stop1"],
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.1
        assert result["max_tokens"] == 200
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["stop1"]

    def test_top_p_alone_is_sent_on_a_model_that_accepts_sampling(self):
        cfg = AnthropicGenerationConfig(model="claude-haiku-4-5", top_p=0.9)
        assert _to_native_params(cfg)["top_p"] == 0.9

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-0",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ],
    )
    def test_temperature_and_top_p_together_send_only_temperature(self, model, caplog):
        cfg = AnthropicGenerationConfig(model=model, temperature=0.3, top_p=0.9)
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert result["temperature"] == 0.3
        assert "top_p" not in result
        assert "top_p is ignored" in caplog.text

    @pytest.mark.parametrize(
        "model",
        ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
    )
    def test_claude_3_models_get_both(self, model, caplog):
        cfg = AnthropicGenerationConfig(model=model, temperature=0.3, top_p=0.9)
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert (result["temperature"], result["top_p"]) == (0.3, 0.9)
        assert "top_p is ignored" not in caplog.text

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-fable-5-2",
            "claude-mythos-5-2",
            "claude-future-9",
        ],
    )
    def test_models_that_reject_sampling_get_none_and_warn(self, model, caplog):
        cfg = AnthropicGenerationConfig(
            model=model, temperature=0.3, top_p=0.9, top_k=5
        )
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert not {"temperature", "top_p", "top_k"} & result.keys()
        assert caplog.text.count("does not accept sampling parameters") == 3
        assert "0.3" not in caplog.text

    def test_thinking_drops_temperature_and_top_k_but_keeps_top_p(self, caplog):
        cfg = AnthropicGenerationConfig(
            model="claude-sonnet-4-6",
            temperature=0.3,
            top_p=0.97,
            top_k=5,
            extra_params={"thinking": {"type": "adaptive"}},
        )
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert "temperature" not in result
        assert "top_k" not in result
        assert result["top_p"] == 0.97
        assert caplog.text.count("incompatible with thinking") == 2

    def test_effort(self):
        cfg = AnthropicGenerationConfig(effort="high")
        result = _to_native_params(cfg)
        assert result["output_config"] == {"effort": "high"}
        assert "thinking" not in result
        assert result["max_tokens"] == 4096

    def test_generic_reasoning_effort_maps_to_effort(self):
        cfg = AnthropicGenerationConfig(reasoning_effort="medium")
        assert _to_native_params(cfg)["output_config"] == {"effort": "medium"}

    @pytest.mark.parametrize("level", ["none", "minimal"])
    def test_lowest_generic_levels_map_to_low_and_log(self, level, caplog):
        cfg = AnthropicGenerationConfig(reasoning_effort=level)
        with caplog.at_level("DEBUG"):
            result = _to_native_params(cfg)
        assert result["output_config"] == {"effort": "low"}
        assert f"'{level}'" in caplog.text
        assert "'low'" in caplog.text

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
        ],
    )
    def test_model_without_effort_support_sends_no_effort_and_warns(
        self, model, caplog
    ):
        cfg = AnthropicGenerationConfig(model=model, reasoning_effort="high")
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert "output_config" not in result
        assert caplog.text.count("reasoning_effort is ignored") == 1
        assert model in caplog.text

    @pytest.mark.parametrize(
        ("model", "level", "sent"),
        [
            ("claude-opus-4-6", "xhigh", "high"),
            ("claude-opus-4-5-20251101", "xhigh", "high"),
            ("claude-sonnet-4-6", "xhigh", "high"),
            ("claude-opus-4-5-20251101", "none", "low"),
            ("claude-opus-4-7", "xhigh", "xhigh"),
            ("claude-future-9", "xhigh", "xhigh"),
            ("Claude-Opus-4-6", "xhigh", "high"),
        ],
    )
    def test_effort_is_clamped_to_the_levels_the_model_accepts(
        self, model, level, sent
    ):
        cfg = AnthropicGenerationConfig(model=model, reasoning_effort=level)
        assert _to_native_params(cfg)["output_config"] == {"effort": sent}

    def test_provider_effort_is_sent_unchanged_on_any_model(self):
        cfg = AnthropicGenerationConfig(model="claude-opus-4-6", effort="xhigh")
        assert _to_native_params(cfg)["output_config"] == {"effort": "xhigh"}
        cfg = AnthropicGenerationConfig(model="claude-haiku-4-5", effort="high")
        assert _to_native_params(cfg)["output_config"] == {"effort": "high"}

    def test_provider_effort_wins_over_generic_reasoning_effort(self):
        cfg = AnthropicGenerationConfig(effort="max", reasoning_effort="low")
        assert _to_native_params(cfg)["output_config"] == {"effort": "max"}

    def test_no_effort_is_sent_by_default(self):
        assert "output_config" not in _to_native_params(AnthropicGenerationConfig())

    @pytest.mark.parametrize("value", [True, False])
    def test_parallel_tool_calls_with_tools(self, value):
        cfg = AnthropicGenerationConfig(parallel_tool_calls=value)
        result = _to_native_params(cfg, has_tools=True)
        assert result["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": not value,
        }

    def test_parallel_tool_calls_without_tools_is_not_sent(self):
        cfg = AnthropicGenerationConfig(parallel_tool_calls=False)
        assert "tool_choice" not in _to_native_params(cfg)

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
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
            }
        }

    def test_json_schema_is_reduced_to_the_supported_subset(self):
        class _Answer(BaseModel):
            score: int = Field(ge=1, le=10)
            label: str = Field(max_length=5)

        schema = model_schema(_Answer)
        original = copy.deepcopy(schema)
        cfg = AnthropicGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        sent = _to_native_params(cfg)["output_config"]["format"]["schema"]

        assert sent["additionalProperties"] is False
        assert "minimum" not in sent["properties"]["score"]
        assert "maximum" not in sent["properties"]["score"]
        assert "maxLength" not in sent["properties"]["label"]
        assert "minimum: 1" in sent["properties"]["score"]["description"]
        assert schema == original

    def test_json_schema_the_sdk_cannot_read_is_sent_unchanged(self):
        schema = {"type": "object", "properties": {"anything": {}}}
        cfg = AnthropicGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        sent = _to_native_params(cfg)["output_config"]["format"]["schema"]
        assert sent == schema

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
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
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

    def test_the_bound_model_and_options_resolve_into_the_provider_config(self):
        provider = AnthropicLLM(model="claude-sonnet-4-6", options={"effort": "high"})
        cfg = provider._resolve(None)
        assert isinstance(cfg, AnthropicGenerationConfig)
        assert (cfg.model, cfg.effort) == ("claude-sonnet-4-6", "high")

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
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 120.0

    def test_client_kwargs_explicit_timeout(self):
        provider = AnthropicLLM(_creds())
        cfg = AnthropicGenerationConfig(timeout=30.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 30.0
        assert kwargs["max_retries"] == 0

    def test_generate_clients_make_no_sdk_retries(self, mocker):
        async_client = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        sync_client = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.Anthropic"
        )
        provider = AnthropicLLM(_creds(), ClientOptions(max_retries=5))
        cfg = AnthropicGenerationConfig()

        provider._async_client(cfg)
        provider._sync_client(cfg)

        assert async_client.call_args.kwargs["max_retries"] == 0
        assert sync_client.call_args.kwargs["max_retries"] == 0

    def test_stream_client_keeps_the_configured_sdk_retries(self, mocker):
        async_client = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        provider = AnthropicLLM(_creds(), ClientOptions(max_retries=5))

        provider._stream_client(AnthropicGenerationConfig())
        assert async_client.call_args.kwargs["max_retries"] == 5

        provider._stream_client(AnthropicGenerationConfig(max_retries=2))
        assert async_client.call_args.kwargs["max_retries"] == 2
        assert async_client.call_args.kwargs["timeout"] == 120.0

    @staticmethod
    def _status_error(status: int) -> anthropic.APIStatusError:
        response = httpx.Response(status, request=httpx.Request("POST", "http://x"))
        return anthropic.APIStatusError("boom", response=response, body=None)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_client_error_is_not_transient(self, status):
        assert AnthropicLLM(_creds())._is_transient(self._status_error(status)) is False

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
    def test_a_temporary_status_is_transient(self, status):
        assert AnthropicLLM(_creds())._is_transient(self._status_error(status)) is True

    def test_a_timeout_or_connection_error_is_transient(self):
        request = httpx.Request("POST", "http://x")
        provider = AnthropicLLM(_creds())

        assert provider._is_transient(anthropic.APITimeoutError(request=request))
        assert provider._is_transient(anthropic.APIConnectionError(request=request))
        assert provider._is_transient(ValueError("other")) is False


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
        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(
                prompt, config=AnthropicGenerationConfig(model="claude-sonnet-4-6")
            )
        assert caught.value.code == "provider_unavailable"


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
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=5, output_tokens=1)
                ),
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
                delta=SimpleNamespace(stop_reason="max_tokens"),
                usage=SimpleNamespace(input_tokens=None, output_tokens=10),
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
        assert [c.delta for c in results] == ["Hello", " World", ""]
        assert [c.usage for c in results if c.usage is not None] == [
            TokenUsage(input_tokens=5, output_tokens=10)
        ]
        assert results[-1].finish_reason == FinishReason.LENGTH

    async def test_stream_takes_cumulative_counts_from_the_last_message_delta(
        self, mocker
    ):
        mock_anthropic = mocker.patch(
            "agent_platform.integrations.llm.anthropic.provider.AsyncAnthropic"
        )
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # With server tools, `message_delta` reports a cumulative input count
        # that replaces the one from `message_start`.
        async def _gen():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=100, output_tokens=2)
                ),
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason=None),
                usage=SimpleNamespace(input_tokens=400, output_tokens=20),
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(input_tokens=400, output_tokens=50),
            )

        mock_client.messages.create = AsyncMock(return_value=_gen())

        provider = AnthropicLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = AnthropicGenerationConfig(model="claude-sonnet-4-6")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert [c.usage for c in results if c.usage is not None] == [
            TokenUsage(input_tokens=400, output_tokens=50)
        ]
        assert results[-1].finish_reason == FinishReason.STOP

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
        assert results[0].usage is None
        assert results[0].finish_reason == FinishReason.UNKNOWN

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
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(input_tokens=None, output_tokens=3),
            ),
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
        assert results[-1].finish_reason == FinishReason.TOOL_CALL


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

        provider = AnthropicLLM(_creds(), model="claude-sonnet-4-6")
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
            id="msg-1",
            content=[_text_block("hi")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
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
