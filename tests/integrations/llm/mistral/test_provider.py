from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.core.credentials import ClientOptions
from agent_platform.core.errors import MissingCredentialError, ProviderError
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
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.mistral.mappers import (
    block_to_native as _block_to_native,
)
from agent_platform.integrations.llm.mistral.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.mistral.mappers import (
    to_native_messages as _to_native_messages,
)
from agent_platform.integrations.llm.mistral.mappers import (
    to_native_params as _to_native_params,
)
from agent_platform.integrations.llm.mistral.provider import MistralLLM


def _creds(key: str = "sk-mist-test") -> MistralCredentials:
    return MistralCredentials(api_key=SecretStr(key))


def _tool_call(id: str, name: str, arguments) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _response(
    content: str | None,
    tool_calls: list[SimpleNamespace] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class TestBlockToNative:
    def test_text_block(self):
        assert _block_to_native(TextBlock(text="hello")) == {
            "type": "text",
            "text": "hello",
        }

    def test_image_block_url(self):
        result = _block_to_native(ImageBlock(image="https://example.com/a.png"))
        assert result == {"type": "image_url", "image_url": "https://example.com/a.png"}

    def test_image_block_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        result = _block_to_native(ImageBlock(image=doc))
        assert result["type"] == "image_url"
        assert result["image_url"].startswith("data:image/png;base64,")

    def test_audio_block(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        result = _block_to_native(AudioBlock(audio=doc))
        assert result["type"] == "audio"
        assert result["input_audio"]


class TestToNative:
    def test_system_message(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_without_tool_calls(self):
        prompt = Prompt(messages=[AssistantMessage(content="Hi there!")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "assistant", "content": "Hi there!"}]

    def test_assistant_message_with_tool_calls(self):
        prompt = Prompt().add_assistant(
            "",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"loc": "Paris"})
            ],
        )
        result = _to_native_messages(prompt)
        assert result[0]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'},
            }
        ]

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
        result = _to_native_messages(prompt)
        assert result == [
            {
                "role": "tool",
                "content": '{"temp": 72}',
                "tool_call_id": "call_123",
                "name": "get_weather",
            }
        ]

    def test_multimodal_user_message(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        result = _to_native_messages(prompt)
        assert result[0]["content"] == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": "https://x/y.png"},
        ]

    def test_empty_prompt(self):
        assert _to_native_messages(Prompt(messages=[])) == []


class TestToNativeParams:
    def test_default_config(self):
        cfg = MistralGenerationConfig()
        result = _to_native_params(cfg)
        assert result == {}

    def test_with_fields(self):
        cfg = MistralGenerationConfig(
            temperature=0.3,
            max_tokens=200,
            top_p=0.9,
            stop_sequences=["stop"],
            seed=123,
            frequency_penalty=0.2,
            presence_penalty=0.1,
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["stop"] == ["stop"]
        assert result["random_seed"] == 123
        assert result["frequency_penalty"] == 0.2
        assert result["presence_penalty"] == 0.1

    def test_response_format_json(self):
        cfg = MistralGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_native_params(cfg)
        assert result["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object"}
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "response", "strict": True},
        }

    def test_response_format_json_schema_custom_name_and_strict(self):
        schema = {"type": "object"}
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
            json_schema_name="my_schema",
            json_schema_strict=False,
        )
        result = _to_native_params(cfg)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "my_schema", "strict": False},
        }

    def test_json_schema_missing_raises(self):
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_extra_params_applied(self):
        cfg = MistralGenerationConfig(extra_params={"safe_prompt": True})
        result = _to_native_params(cfg)
        assert result["safe_prompt"] is True


class TestFromNative:
    def test_simple_message(self):
        response = _response("Hello world", prompt_tokens=1, completion_tokens=2)
        result = _from_native_response(response, model="mistral-medium-latest")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "mistral-medium-latest"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls_string_arguments(self):
        response = _response(
            None,
            tool_calls=[_tool_call("call_abc", "get_weather", '{"location": "Paris"}')],
        )
        result = _from_native_response(response, model="mistral-medium-latest")
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_with_tool_calls_dict_arguments(self):
        response = _response(
            None,
            tool_calls=[_tool_call("call_abc", "get_weather", {"location": "Paris"})],
        )
        result = _from_native_response(response, model="mistral-medium-latest")
        assert result.message.tool_calls[0].arguments == {"location": "Paris"}

    def test_no_tool_calls(self):
        response = _response("Hi")
        result = _from_native_response(response, model="mistral-medium-latest")
        assert result.message.tool_calls == []


class TestMistralLLMConstruction:
    def test_construct(self):
        provider = MistralLLM(_creds("mist-123"))
        assert provider._credentials.api_key.get_secret_value() == "mist-123"

    def test_default_config(self):
        provider = MistralLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, MistralGenerationConfig)
        assert cfg.model == "mistral-medium-latest"

    def test_tool_to_schema(self):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "test_tool"
            description = "A test tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        provider = MistralLLM(_creds())
        schema = provider._tool_to_schema(_SchemaTool())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"

    def test_missing_credential_error(self):
        provider = MistralLLM()
        provider._credentials = MistralCredentials()
        cfg = MistralGenerationConfig(model="mistral-large")
        with pytest.raises(MissingCredentialError, match="Mistral API key is required"):
            provider._async_client(cfg)


class TestMistralLLMClient:
    def test_client_creation(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        provider = MistralLLM(_creds())
        config = MistralGenerationConfig(model="mistral-large", timeout=30.0)
        client = provider._async_client(config)
        assert client is mock_mistral.return_value
        _, kwargs = mock_mistral.call_args
        assert kwargs["timeout_ms"] == 30000

    def test_client_with_base_url(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        provider = MistralLLM(
            _creds(), client_options=ClientOptions(base_url="https://custom.mistral")
        )
        provider._async_client(MistralGenerationConfig())
        _, kwargs = mock_mistral.call_args
        assert kwargs["server_url"] == "https://custom.mistral"


class TestMistralLLMGenerate:
    def test_sync_generate(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete.return_value = _response("Sync hello")

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        result = provider.generate(prompt, config=config)

        assert result.message.content == "Sync hello"
        mock_client.chat.complete.assert_called_once()

    def test_sync_generate_with_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete.return_value = _response("ok")

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        provider.generate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.complete.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete_async = AsyncMock(
            return_value=_response("Async hello")
        )

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        result = await provider.agenerate(prompt, config=config)

        assert result.message.content == "Async hello"

    async def test_agenerate_with_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.chat.complete_async = AsyncMock(return_value=_response("ok"))

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        await provider.agenerate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.complete_async.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response("ok")

        mock_client.chat.complete_async = flaky

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=MistralGenerationConfig(model="mistral-medium-latest")
        )
        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.chat.complete_async = always_fails

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(
                prompt, config=MistralGenerationConfig(model="mistral-medium-latest")
            )


class TestMistralLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        events = [
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Hello", tool_calls=None)
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=" World", tool_calls=None)
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
                )
            ),
        ]

        async def _gen():
            for event in events:
                yield event

        mock_client.chat.stream_async = AsyncMock(return_value=_gen())

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].delta == ""
        assert results[2].usage.input_tokens == 5
        assert results[2].usage.output_tokens == 10
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_with_no_events(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        async def _gen():
            if False:
                yield

        mock_client.chat.stream_async = AsyncMock(return_value=_gen())

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
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

        mock_mistral = mocker.patch(
            "agent_platform.integrations.llm.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        events = [
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[_tool_call("call_1", "search", "")],
                            )
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[_tool_call(None, None, '{"query": "hi"}')],
                            )
                        )
                    ],
                    usage=None,
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )
            ),
        ]

        async def _gen():
            for event in events:
                yield event

        mock_client.chat.stream_async = AsyncMock(return_value=_gen())

        provider = MistralLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = MistralGenerationConfig(model="mistral-medium-latest")
        results = [
            c
            async for c in provider.stream(prompt, config=config, tools=[_SchemaTool()])
        ]

        _, kwargs = mock_client.chat.stream_async.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "search"
        assert deltas[1].arguments_delta == '{"query": "hi"}'
