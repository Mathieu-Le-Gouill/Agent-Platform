from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.tools.base import Tool
from agent_platform.core.errors import ProviderError
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
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.ollama.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.ollama.mappers import (
    message_content as _message_content,
)
from agent_platform.integrations.llm.ollama.mappers import (
    to_native_messages as _to_native_messages,
)
from agent_platform.integrations.llm.ollama.mappers import (
    to_native_params as _to_native_params,
)
from agent_platform.integrations.llm.ollama.provider import OllamaLLM


def _tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _response(
    content: str | None,
    tool_calls: list[SimpleNamespace] | None = None,
    prompt_eval_count: int = 0,
    eval_count: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


class TestMessageContent:
    def test_plain_string_content(self):
        content, images = _message_content(UserMessage(content="hi there"))
        assert content == "hi there"
        assert images == []

    def test_text_and_image_blocks(self):
        msg = UserMessage(
            content=[
                TextBlock(text="describe this"),
                ImageBlock(image="https://x/y.png"),
            ]
        )
        content, images = _message_content(msg)
        assert content == "describe this"
        assert images == ["https://x/y.png"]

    def test_image_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        msg = UserMessage(content=[ImageBlock(image=doc)])
        _, images = _message_content(msg)
        assert len(images) == 1
        assert images[0]

    def test_audio_block_unsupported(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        msg = UserMessage(content=[AudioBlock(audio=doc)])
        with pytest.raises(ProviderError, match="does not support audio"):
            _message_content(msg)


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
            {"function": {"name": "get_weather", "arguments": {"loc": "Paris"}}}
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
            {"role": "tool", "content": '{"temp": 72}', "tool_name": "get_weather"}
        ]

    def test_user_message_with_images(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        result = _to_native_messages(prompt)
        assert result == [
            {
                "role": "user",
                "content": "what is this?",
                "images": ["https://x/y.png"],
            }
        ]

    def test_system_message_with_images(self):
        prompt = Prompt(
            messages=[
                SystemMessage(
                    content=[
                        TextBlock(text="context"),
                        ImageBlock(image="https://x/y.png"),
                    ]
                )
            ]
        )
        result = _to_native_messages(prompt)
        assert result[0]["images"] == ["https://x/y.png"]

    def test_assistant_message_with_images(self):
        prompt = Prompt(
            messages=[
                AssistantMessage(
                    content=[
                        TextBlock(text="here"),
                        ImageBlock(image="https://x/y.png"),
                    ]
                )
            ]
        )
        result = _to_native_messages(prompt)
        assert result[0]["images"] == ["https://x/y.png"]

    def test_empty_prompt(self):
        assert _to_native_messages(Prompt(messages=[])) == []


class TestToNativeParams:
    def test_default_config(self):
        cfg = OllamaGenerationConfig()
        result = _to_native_params(cfg)
        assert result == {"options": {"temperature": 0.7}}

    def test_with_fields(self):
        cfg = OllamaGenerationConfig(
            temperature=0.5,
            max_tokens=500,
            top_p=0.95,
            top_k=40,
            seed=99,
            stop_sequences=["stop"],
        )
        result = _to_native_params(cfg)
        options = result["options"]
        assert options["temperature"] == 0.5
        assert options["num_predict"] == 500
        assert options["top_p"] == 0.95
        assert options["top_k"] == 40
        assert options["seed"] == 99
        assert options["stop"] == ["stop"]

    def test_frequency_penalty_is_not_remapped_to_repeat_penalty(self):
        cfg = OllamaGenerationConfig(frequency_penalty=0.3)
        result = _to_native_params(cfg)
        assert "repeat_penalty" not in result["options"]

    def test_repeat_penalty_native_field(self):
        cfg = OllamaGenerationConfig(repeat_penalty=1.2)
        result = _to_native_params(cfg)
        assert result["options"]["repeat_penalty"] == 1.2

    def test_mirostat_fields(self):
        cfg = OllamaGenerationConfig(mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)
        result = _to_native_params(cfg)
        assert result["options"]["mirostat"] == 2
        assert result["options"]["mirostat_tau"] == 5.0
        assert result["options"]["mirostat_eta"] == 0.1

    def test_num_ctx_field(self):
        cfg = OllamaGenerationConfig(num_ctx=4096)
        result = _to_native_params(cfg)
        assert result["options"]["num_ctx"] == 4096

    def test_optional_fields_omitted_when_unset(self):
        cfg = OllamaGenerationConfig()
        result = _to_native_params(cfg)
        for key in (
            "repeat_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "num_ctx",
        ):
            assert key not in result["options"]

    def test_response_format_json(self):
        cfg = OllamaGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_native_params(cfg)
        assert result["format"] == "json"

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        cfg = OllamaGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["format"] == schema

    def test_json_schema_missing_raises(self):
        cfg = OllamaGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_extra_params_applied(self):
        cfg = OllamaGenerationConfig(extra_params={"think": True})
        result = _to_native_params(cfg)
        assert result["think"] is True


class TestFromNative:
    def test_simple_message(self):
        response = _response("Hello world", prompt_eval_count=1, eval_count=2)
        result = _from_native_response(response, model="llama3.2")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "llama3.2"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls_generates_id(self):
        response = _response(
            None, tool_calls=[_tool_call("get_weather", {"location": "Paris"})]
        )
        result = _from_native_response(response, model="llama3.2")
        tc = result.message.tool_calls[0]
        assert tc.id
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_no_tool_calls(self):
        response = _response("Hi")
        result = _from_native_response(response, model="llama3.2")
        assert result.message.tool_calls == []


class TestOllamaLLMConstruction:
    def test_construct(self):
        provider = OllamaLLM(OllamaCredentials())
        assert hasattr(provider, "_credentials")

    def test_default_config(self):
        provider = OllamaLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, OllamaGenerationConfig)
        assert cfg.model == "llama3.2"

    def test_tool_to_schema(self):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "test_tool"
            description = "A test tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        provider = OllamaLLM()
        schema = provider._tool_to_schema(_SchemaTool())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"

    def test_no_credential_needed(self):
        provider = OllamaLLM(OllamaCredentials())
        cfg = OllamaGenerationConfig(model="llama3")
        client = provider._async_client(cfg)
        assert client is not None

    def test_client_kwargs_default_no_timeout(self):
        provider = OllamaLLM()
        kwargs = provider._client_kwargs(OllamaGenerationConfig())
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout(self):
        provider = OllamaLLM()
        kwargs = provider._client_kwargs(OllamaGenerationConfig(timeout=30.0))
        assert kwargs["timeout"] == 30.0

    def test_max_retries_default(self):
        provider = OllamaLLM()
        assert provider._max_retries(OllamaGenerationConfig()) == 3

    def test_max_retries_explicit(self):
        provider = OllamaLLM()
        cfg = OllamaGenerationConfig(max_retries=5)
        assert provider._max_retries(cfg) == 5

    def test_client_wires_max_retries_into_async_transport(self):
        provider = OllamaLLM()
        client = provider._async_client(OllamaGenerationConfig(max_retries=2))
        transport = client._client._transport
        assert transport._pool._retries == 2

    def test_sync_client_wires_max_retries_into_transport(self):
        provider = OllamaLLM()
        client = provider._sync_client(OllamaGenerationConfig(max_retries=4))
        transport = client._client._transport
        assert transport._pool._retries == 4


class TestOllamaLLMGenerate:
    def test_sync_generate(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = _response("Sync hello")

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
        result = provider.generate(prompt, config=config)

        assert result.message.content == "Sync hello"
        mock_client.chat.assert_called_once()

    def test_sync_generate_with_tools(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = _response("ok")

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
        provider.generate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat = AsyncMock(return_value=_response("Async hello"))

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
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

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat = AsyncMock(return_value=_response("ok"))

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
        await provider.agenerate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response("ok")

        mock_client.chat = flaky

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=OllamaGenerationConfig(model="llama3.2")
        )
        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.chat = always_fails

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(
                prompt, config=OllamaGenerationConfig(model="llama3.2")
            )


class TestOllamaLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            SimpleNamespace(
                message=SimpleNamespace(content="Hello", tool_calls=None), done=False
            ),
            SimpleNamespace(
                message=SimpleNamespace(content=" World", tool_calls=None), done=False
            ),
            SimpleNamespace(
                message=None,
                done=True,
                prompt_eval_count=5,
                eval_count=10,
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.chat = AsyncMock(return_value=_gen())

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
        results = [c async for c in provider.stream(prompt, config=config)]

        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].delta == ""
        assert results[2].usage.input_tokens == 5
        assert results[2].usage.output_tokens == 10
        assert results[3].delta == ""
        assert results[3].finish_reason == FinishReason.STOP

    async def test_stream_with_no_chunks(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def _gen():
            if False:
                yield

        mock_client.chat = AsyncMock(return_value=_gen())

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
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

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[_tool_call("search", {"query": "hi"})],
                ),
                done=False,
            ),
            SimpleNamespace(
                message=None,
                done=True,
                prompt_eval_count=1,
                eval_count=1,
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.chat = AsyncMock(return_value=_gen())

        provider = OllamaLLM()
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OllamaGenerationConfig(model="llama3.2")
        results = [
            c
            async for c in provider.stream(prompt, config=config, tools=[_SchemaTool()])
        ]

        _, kwargs = mock_client.chat.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].name == "search"
        assert deltas[0].arguments_delta == '{"query": "hi"}'
