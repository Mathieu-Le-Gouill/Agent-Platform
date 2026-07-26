from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import FinishReason, ImageFormat
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
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.provider import (
    OpenAILLM,
    _block_to_native,
    _from_native_response,
    _to_native_messages,
    _to_native_params,
)


def _creds(key: str = "sk-test") -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr(key))


def _tool_call(id: str, name: str, arguments: str) -> SimpleNamespace:
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
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
        }

    def test_image_block_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        result = _block_to_native(ImageBlock(image=doc))
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_audio_block(self):
        from agent_platform.core.schemas.document import AudioDocument
        from agent_platform.core.schemas.enums import AudioFormat

        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        result = _block_to_native(AudioBlock(audio=doc))
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "wav"
        assert result["input_audio"]["data"]


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
            }
        ]

    def test_multimodal_user_message(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        result = _to_native_messages(prompt)
        assert result[0]["content"] == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]

    def test_empty_prompt(self):
        assert _to_native_messages(Prompt(messages=[])) == []


class TestToNativeParams:
    def test_default_config_omits_default_temperature(self):
        cfg = OpenAIGenerationConfig()
        result = _to_native_params(cfg)
        assert result == {}

    def test_with_fields(self):
        cfg = OpenAIGenerationConfig(
            temperature=0.2,
            max_tokens=100,
            top_p=0.8,
            stop_sequences=["stop"],
            seed=42,
            frequency_penalty=0.3,
            presence_penalty=0.4,
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.8
        assert result["stop"] == ["stop"]
        assert result["seed"] == 42
        assert result["frequency_penalty"] == 0.3
        assert result["presence_penalty"] == 0.4

    def test_temperature_at_default_is_omitted(self):
        cfg = OpenAIGenerationConfig(temperature=0.7)
        result = _to_native_params(cfg)
        assert "temperature" not in result

    def test_temperature_non_default_non_reasoning_model_is_sent(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", temperature=0.2)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2

    @pytest.mark.parametrize(
        "model", ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"]
    )
    def test_temperature_dropped_for_reasoning_models(self, model):
        cfg = OpenAIGenerationConfig(model=model, temperature=0.2)
        result = _to_native_params(cfg)
        assert "temperature" not in result

    def test_temperature_kept_for_gpt5_chat(self):
        cfg = OpenAIGenerationConfig(model="gpt-5-chat", temperature=0.2)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2

    def test_reasoning_effort(self):
        cfg = OpenAIGenerationConfig(model="o3", reasoning_effort="high")
        result = _to_native_params(cfg)
        assert result["reasoning_effort"] == "high"

    def test_reasoning_effort_dropped_for_non_reasoning_model(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", reasoning_effort="high")
        result = _to_native_params(cfg)
        assert "reasoning_effort" not in result

    def test_parallel_tool_calls_false(self):
        cfg = OpenAIGenerationConfig(parallel_tool_calls=False)
        result = _to_native_params(cfg)
        assert result["parallel_tool_calls"] is False

    def test_parallel_tool_calls_default_omitted(self):
        cfg = OpenAIGenerationConfig()
        result = _to_native_params(cfg)
        assert "parallel_tool_calls" not in result

    def test_response_format_json(self):
        cfg = OpenAIGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_native_params(cfg)
        assert result["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }

    def test_response_format_json_schema_strict_false(self):
        schema = {"type": "object"}
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
            strict=False,
        )
        result = _to_native_params(cfg)
        assert result["response_format"]["json_schema"]["strict"] is False

    def test_json_schema_missing_raises(self):
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_extra_params_applied(self):
        cfg = OpenAIGenerationConfig(extra_params={"user": "u1"})
        result = _to_native_params(cfg)
        assert result["user"] == "u1"


class TestFromNative:
    def test_simple_message(self):
        response = _response("Hello world", prompt_tokens=1, completion_tokens=2)
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "gpt-4.1"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls(self):
        response = _response(
            None,
            tool_calls=[_tool_call("call_abc", "get_weather", '{"location": "Paris"}')],
        )
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.content == ""
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_no_tool_calls(self):
        response = _response("Hi")
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.tool_calls == []


class TestOpenAILLMConstruction:
    def test_construct(self):
        provider = OpenAILLM(_creds("sk-123"))
        assert provider._credentials.api_key.get_secret_value() == "sk-123"

    def test_default_config(self):
        provider = OpenAILLM()
        cfg = provider._default_config()
        assert isinstance(cfg, OpenAIGenerationConfig)
        assert cfg.model == "gpt-4.1"

    def test_tool_to_schema(self):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "test_tool"
            description = "A test tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        provider = OpenAILLM(_creds())
        schema = provider._tool_to_schema(_SchemaTool())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "parameters" in schema["function"]

    def test_missing_credential_error(self):
        provider = OpenAILLM()
        provider._credentials = OpenAICredentials()
        cfg = OpenAIGenerationConfig(model="gpt-4")
        with pytest.raises(MissingCredentialError, match="OPENAI API key is required"):
            provider._async_client(cfg)

    def test_client_kwargs_default_max_retries(self):
        provider = OpenAILLM(_creds())
        cfg = OpenAIGenerationConfig(max_retries=None)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout(self):
        provider = OpenAILLM(_creds())
        cfg = OpenAIGenerationConfig(timeout=15.0)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0


class TestOpenAILLMGenerate:
    def test_sync_generate(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _response("Sync hello")

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        result = provider.generate(prompt, config=config)

        assert result.message.content == "Sync hello"
        assert result.model == "gpt-4.1"
        mock_client.chat.completions.create.assert_called_once()

    def test_sync_generate_with_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _response("ok")

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        provider.generate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=_response("Async hello")
        )

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
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

        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=_response("ok"))

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        await provider.agenerate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response("ok")

        mock_client.chat.completions.create = flaky

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=OpenAIGenerationConfig(model="gpt-4.1")
        )
        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.chat.completions.create = always_fails

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(
                prompt, config=OpenAIGenerationConfig(model="gpt-4.1")
            )


class TestOpenAILLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=" World"))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create = AsyncMock(return_value=_gen())

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].delta == ""
        assert results[2].usage.input_tokens == 5
        assert results[2].usage.output_tokens == 10
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_with_no_chunks(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        async def _gen():
            if False:
                yield

        mock_client.chat.completions.create = AsyncMock(return_value=_gen())

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert len(results) == 1
        assert results[0].delta == ""
        assert results[0].finish_reason == FinishReason.STOP
