from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr

pytest.importorskip("google.genai")

from agent_platform.agents.tools.base import Tool
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, ImageFormat
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
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig
from agent_platform.integrations.llm.google.provider import (
    GoogleLLM,
    _from_native_response,
    _to_native_config,
    _to_native_contents,
)
from tests.helpers import assert_custom_construction_stored, assert_default_construction


def _creds(key: str = "gm-test") -> GoogleCredentials:
    return GoogleCredentials(api_key=SecretStr(key))


def _part(text=None, function_call=None):
    return SimpleNamespace(text=text, function_call=function_call)


def _function_call(id, name, args):
    return SimpleNamespace(id=id, name=name, args=args)


def _response(parts, prompt_tokens=None, candidates_tokens=None):
    usage = (
        SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=candidates_tokens
        )
        if prompt_tokens is not None or candidates_tokens is not None
        else None
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=usage,
    )


class TestGoogleLLMConstruction:
    def test_default_credentials_and_config(self):
        assert_default_construction(GoogleLLM, GoogleGenerationConfig)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(GoogleLLM, _creds())

    def test_missing_api_key_raises(self):
        provider = GoogleLLM(GoogleCredentials(api_key=None))
        with pytest.raises(MissingCredentialError):
            provider._client(GoogleGenerationConfig())


class TestToNativeContents:
    def test_system_message_extracted_separately(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        system, contents = _to_native_contents(prompt)
        assert system == "Be helpful."
        assert contents == []

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        _, contents = _to_native_contents(prompt)
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"

    def test_assistant_message_with_tool_calls(self):
        prompt = Prompt().add_assistant(
            "",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"loc": "Paris"})
            ],
        )
        _, contents = _to_native_contents(prompt)
        assert contents[0].role == "model"
        fc = contents[0].parts[-1].function_call
        assert fc.name == "get_weather"
        assert fc.args == {"loc": "Paris"}

    def test_tool_message_becomes_function_response(self):
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
        _, contents = _to_native_contents(prompt)
        assert contents[0].role == "user"
        fr = contents[0].parts[0].function_response
        assert fr.name == "get_weather"

    def test_empty_prompt(self):
        system, contents = _to_native_contents(Prompt(messages=[]))
        assert system is None
        assert contents == []

    def test_multimodal_user_message_with_image_and_audio(self):
        image_doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        audio_doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        prompt = Prompt().add_user_content(
            [
                TextBlock(text="what is this?"),
                ImageBlock(image=image_doc),
                ImageBlock(image="https://example.com/a.png"),
                AudioBlock(audio=audio_doc),
            ]
        )
        _, contents = _to_native_contents(prompt)
        parts = contents[0].parts
        assert parts[0].text == "what is this?"
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[2].file_data.file_uri == "https://example.com/a.png"
        assert parts[3].inline_data.mime_type == "audio/wav"


class TestToNativeConfig:
    def test_default_config(self):
        cfg = GoogleGenerationConfig()
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.temperature == cfg.temperature

    def test_all_sampling_fields_forwarded(self):
        cfg = GoogleGenerationConfig(
            max_tokens=100,
            top_p=0.9,
            top_k=40,
            stop_sequences=["stop"],
            seed=42,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        native = _to_native_config(cfg, "system prompt", None, lambda t: t)
        assert native.max_output_tokens == 100
        assert native.top_p == 0.9
        assert native.top_k == 40
        assert native.stop_sequences == ["stop"]
        assert native.seed == 42
        assert native.presence_penalty == 0.1
        assert native.frequency_penalty == 0.2
        assert native.system_instruction == "system prompt"

    def test_json_response_format(self):
        cfg = GoogleGenerationConfig(response_format=ResponseFormat.JSON)
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.response_mime_type == "application/json"

    def test_json_schema_requires_schema(self):
        cfg = GoogleGenerationConfig(response_format=ResponseFormat.JSON_SCHEMA)
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_config(cfg, None, None, lambda t: t)

    def test_json_schema_sets_response_mime_type(self):
        cfg = GoogleGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema={"type": "object"},
        )
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.response_mime_type == "application/json"
        assert native.response_json_schema == {"type": "object"}

    def test_thinking_budget_sets_thinking_config(self):
        cfg = GoogleGenerationConfig(thinking_budget=100)
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.thinking_config.thinking_budget == 100


class TestFromNativeResponse:
    def test_extracts_text_and_tool_calls(self):
        response = _response(
            [
                _part(text="hi"),
                _part(function_call=_function_call("c1", "search", {"q": 1})),
            ]
        )
        result = _from_native_response(response, "gemini-2.5-flash")
        assert result.message.content == "hi"
        assert result.message.tool_calls[0].name == "search"
        assert result.message.tool_calls[0].arguments == {"q": 1}


class TestGoogleLLMGenerate:
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
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _response([_part(text="ok")])

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = provider.generate(prompt, tools=[_SchemaTool()])

        assert result.message.content == "ok"
        _, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["config"].tools[0].function_declarations[0].name == "search"

    async def test_agenerate(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_response([_part(text="async hi")])
        )

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(prompt)

        assert result.message.content == "async hi"

    async def test_agenerate_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _response([_part(text="ok")])

        mock_client.aio.models.generate_content = flaky

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        result = await provider.agenerate(prompt)

        assert calls["n"] == 2
        assert result.message.content == "ok"

    async def test_agenerate_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.aio.models.generate_content = always_fails

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError, match="LLM generation failed"):
            await provider.agenerate(prompt)


class TestGoogleLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            _response([_part(text="Hello")]),
            _response([_part(text=" World")], prompt_tokens=5, candidates_tokens=10),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt)]

        assert results[0].delta == "Hello"
        assert results[1].delta == " World"
        assert results[2].usage.input_tokens == 5
        assert results[2].usage.output_tokens == 10

    async def test_stream_with_tools_yields_tool_call_delta(self, mocker):
        class _Input(BaseModel):
            query: str

        class _SchemaTool(Tool):
            name = "search"
            description = "search tool"
            input_schema = _Input

            async def run(self, **kwargs):
                return "ok"

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        chunks = [
            _response(
                [_part(function_call=_function_call("c1", "search", {"query": "hi"}))]
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

        provider = GoogleLLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt, tools=[_SchemaTool()])]

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].name == "search"
        assert deltas[0].arguments_delta == '{"query": "hi"}'
