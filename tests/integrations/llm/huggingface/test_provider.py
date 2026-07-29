from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, FinishReason, ImageFormat
from agent_platform.core.schemas.message import (
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
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.llm.huggingface.mappers import (
    block_to_native as _block_to_native,
)
from agent_platform.integrations.llm.huggingface.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.huggingface.mappers import (
    to_native_messages as _to_native_messages,
)
from agent_platform.integrations.llm.huggingface.mappers import (
    to_native_params as _to_native_params,
)
from agent_platform.integrations.llm.huggingface.provider import HuggingFaceLLM


def _creds(key: str = "hf_test") -> HuggingFaceCredentials:
    return HuggingFaceCredentials(api_key=SecretStr(key))


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
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_audio_block_unsupported(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        with pytest.raises(ProviderError, match="does not support audio"):
            _block_to_native(AudioBlock(audio=doc))


class TestToNative:
    def test_system_message(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "user", "content": "Hello"}]

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
    def test_default_config(self):
        cfg = HuggingFaceGenerationConfig()
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.7
        assert "extra_body" not in result

    def test_temperature_zero_is_sent(self):
        # Regression: `if config.temperature:` used to drop an explicit 0.0.
        cfg = HuggingFaceGenerationConfig(temperature=0.0)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.0

    def test_max_tokens_zero_is_sent(self):
        cfg = HuggingFaceGenerationConfig(max_tokens=0)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 0

    def test_with_fields(self):
        cfg = HuggingFaceGenerationConfig(
            temperature=0.5, max_tokens=200, top_p=0.9, stop_sequences=["stop"], seed=7
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["stop"] == ["stop"]
        assert result["seed"] == 7

    def test_new_fields_forwarded_via_extra_body(self):
        cfg = HuggingFaceGenerationConfig(
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True,
            typical_p=0.9,
            return_full_text=False,
        )
        result = _to_native_params(cfg)
        assert result["extra_body"] == {
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "typical_p": 0.9,
            "return_full_text": False,
        }

    def test_do_sample_false_is_forwarded(self):
        cfg = HuggingFaceGenerationConfig(do_sample=False)
        result = _to_native_params(cfg)
        assert result["extra_body"]["do_sample"] is False

    def test_return_full_text_false_is_forwarded(self):
        cfg = HuggingFaceGenerationConfig(return_full_text=False)
        result = _to_native_params(cfg)
        assert result["extra_body"]["return_full_text"] is False

    def test_extra_params_applied(self):
        cfg = HuggingFaceGenerationConfig(extra_params={"n": 2})
        result = _to_native_params(cfg)
        assert result["n"] == 2

    def test_frequency_penalty_forwarded(self):
        cfg = HuggingFaceGenerationConfig(frequency_penalty=0.3)
        result = _to_native_params(cfg)
        assert result["frequency_penalty"] == 0.3

    def test_presence_penalty_forwarded(self):
        cfg = HuggingFaceGenerationConfig(presence_penalty=0.4)
        result = _to_native_params(cfg)
        assert result["presence_penalty"] == 0.4

    def test_penalties_omitted_when_unset(self):
        cfg = HuggingFaceGenerationConfig()
        result = _to_native_params(cfg)
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result

    def test_response_format_json(self):
        cfg = HuggingFaceGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_native_params(cfg)
        assert result["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema_missing_raises(self):
        cfg = HuggingFaceGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_response_format_json_schema_wires_response_format(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cfg = HuggingFaceGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }


class TestFromNative:
    def test_simple_message(self):
        response = _response("Hello world", prompt_tokens=1, completion_tokens=2)
        result = _from_native_response(response, model="test/model")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "test/model"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls(self):
        response = _response(
            None,
            tool_calls=[_tool_call("call_abc", "get_weather", '{"location": "Paris"}')],
        )
        result = _from_native_response(response, model="test/model")
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_no_tool_calls(self):
        response = _response("Hi")
        result = _from_native_response(response, model="test/model")
        assert result.message.tool_calls == []


class TestHuggingFaceLLMConstruction:
    def test_construct(self):
        provider = HuggingFaceLLM(_creds("hf_123"))
        assert provider._credentials.api_key.get_secret_value() == "hf_123"

    def test_default_config(self):
        provider = HuggingFaceLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, HuggingFaceGenerationConfig)
        assert cfg.repo_id == "deepseek-ai/DeepSeek-R1-0528"
        assert cfg.provider == "auto"

    def test_tool_to_schema(self):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "test_tool"
            description = "A test tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        provider = HuggingFaceLLM(_creds())
        schema = provider._tool_to_schema(_SchemaTool())
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"

    def test_missing_credential_error(self):
        provider = HuggingFaceLLM()
        provider._credentials = HuggingFaceCredentials()
        cfg = HuggingFaceGenerationConfig(repo_id="test/model")
        with pytest.raises(
            MissingCredentialError, match="Hugging Face Hub API token is required"
        ):
            provider._async_client(cfg)

    def test_client_kwargs(self, mocker):
        provider = HuggingFaceLLM(_creds())
        cfg = HuggingFaceGenerationConfig(repo_id="test/model", provider="cerebras")
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["model"] == "test/model"
        assert kwargs["provider"] == "cerebras"
        assert kwargs["token"] == "hf_test"
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout(self):
        provider = HuggingFaceLLM(_creds())
        cfg = HuggingFaceGenerationConfig(timeout=30.0)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 30.0


class TestHuggingFaceLLMGenerate:
    def test_sync_generate(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.huggingface.provider.InferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat_completion.return_value = _response("Sync hello")

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
        result = provider.generate(prompt, config=config)

        assert result.message.content == "Sync hello"
        mock_client.chat_completion.assert_called_once()

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
            "agent_platform.integrations.llm.huggingface.provider.InferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat_completion.return_value = _response("ok")

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
        provider.generate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat_completion.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat_completion = AsyncMock(return_value=_response("Async hello"))

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
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
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat_completion = AsyncMock(return_value=_response("ok"))

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
        await provider.agenerate(prompt, config=config, tools=[_SchemaTool()])

        _, kwargs = mock_client.chat_completion.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response("ok")

        mock_client.chat_completion = flaky

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(
            prompt, config=HuggingFaceGenerationConfig(repo_id="test/model")
        )
        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.chat_completion = always_fails

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(
                prompt, config=HuggingFaceGenerationConfig(repo_id="test/model")
            )


class TestHuggingFaceLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello", tool_calls=None)
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" World", tool_calls=None)
                    )
                ],
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

        mock_client.chat_completion = AsyncMock(return_value=_gen())

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
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
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def _gen():
            if False:
                yield

        mock_client.chat_completion = AsyncMock(return_value=_gen())

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
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
            "agent_platform.integrations.llm.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="search", arguments=""
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(
                                        name=None, arguments='{"query": "hi"}'
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.chat_completion = AsyncMock(return_value=_gen())

        provider = HuggingFaceLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = HuggingFaceGenerationConfig(repo_id="test/model")
        results = [
            c
            async for c in provider.stream(prompt, config=config, tools=[_SchemaTool()])
        ]

        _, kwargs = mock_client.chat_completion.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "search"
        assert deltas[1].arguments_delta == '{"query": "hi"}'
