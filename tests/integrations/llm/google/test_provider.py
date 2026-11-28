from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, SecretStr

pytest.importorskip("google.genai")

from google.genai import errors as genai_errors

from agent_platform.agents.tools.base import Tool
from agent_platform.core.credentials import ClientOptions
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
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig
from agent_platform.integrations.llm.google.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.google.mappers import (
    to_native_config as _to_native_config,
)
from agent_platform.integrations.llm.google.mappers import (
    to_native_contents as _to_native_contents,
)
from agent_platform.integrations.llm.google.provider import GoogleLLM
from tests.helpers import assert_custom_construction_stored


def _creds(key: str = "gm-test") -> GoogleCredentials:
    return GoogleCredentials(api_key=SecretStr(key))


def _part(text=None, function_call=None, thought=None):
    return SimpleNamespace(text=text, function_call=function_call, thought=thought)


def _function_call(id, name, args):
    return SimpleNamespace(id=id, name=name, args=args)


def _response(parts, prompt_tokens=None, candidates_tokens=None, finish_reason="STOP"):
    usage = (
        SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=candidates_tokens
        )
        if prompt_tokens is not None or candidates_tokens is not None
        else None
    )
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts), finish_reason=finish_reason
            )
        ],
        usage_metadata=usage,
        response_id="resp-1",
    )


class TestGoogleLLMConstruction:
    def test_the_bound_model_and_options_resolve_into_the_provider_config(self):
        provider = GoogleLLM(model="gemini-2.5-flash", options={"thinking_budget": 256})
        cfg = provider._resolve(None)
        assert isinstance(cfg, GoogleGenerationConfig)
        assert (cfg.model, cfg.thinking_budget) == ("gemini-2.5-flash", 256)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(GoogleLLM, _creds())

    def test_missing_api_key_raises(self):
        provider = GoogleLLM(GoogleCredentials(api_key=None))
        with pytest.raises(MissingCredentialError):
            provider._client(GoogleGenerationConfig())

    def test_client_gets_the_default_timeout_in_milliseconds(self, mocker):
        client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )

        GoogleLLM(_creds())._client(GoogleGenerationConfig())

        http_options = client_cls.call_args.kwargs["http_options"]
        assert http_options.timeout == 120000

    def test_client_timeout_comes_from_the_config_then_the_client_options(self, mocker):
        client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        provider = GoogleLLM(_creds(), ClientOptions(timeout=30.0))

        provider._client(GoogleGenerationConfig())
        assert client_cls.call_args.kwargs["http_options"].timeout == 30000

        provider._client(GoogleGenerationConfig(timeout=5.0))
        assert client_cls.call_args.kwargs["http_options"].timeout == 5000

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_client_error_is_not_transient(self, status):
        error = genai_errors.APIError(status, {"error": {"message": "bad"}})
        assert GoogleLLM(_creds())._is_transient(error) is False

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
    def test_a_temporary_status_is_transient(self, status):
        error = genai_errors.APIError(status, {"error": {"message": "busy"}})
        assert GoogleLLM(_creds())._is_transient(error) is True

    def test_a_timeout_or_connection_error_is_transient(self):
        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")

        assert provider._is_transient(httpx.ReadTimeout("slow"))
        assert provider._is_transient(httpx.ConnectError("down"))
        assert provider._is_transient(ConnectionError("down"))
        assert provider._is_transient(ValueError("other")) is False

    def test_an_aiohttp_connection_error_is_transient(self):
        aiohttp = pytest.importorskip("aiohttp")
        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")

        assert provider._is_transient(aiohttp.ClientConnectionError("reset"))
        assert provider._is_transient(aiohttp.ServerDisconnectedError())


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

    def test_unknown_media_formats_fall_back_to_png_and_wav(self):
        prompt = Prompt().add_user_content(
            [
                ImageBlock(image=ImageDocument(content=b"\x89PNG")),
                AudioBlock(audio=AudioDocument(content=b"RIFF....")),
            ]
        )
        _, contents = _to_native_contents(prompt)
        parts = contents[0].parts
        assert parts[0].inline_data.mime_type == "image/png"
        assert parts[1].inline_data.mime_type == "audio/wav"


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

    @pytest.mark.parametrize(
        ("model", "level", "native_level"),
        [
            ("gemini-2.5-flash", "none", "LOW"),
            ("gemini-2.5-flash", "medium", "MEDIUM"),
            ("gemini-3-pro-preview", "medium", "HIGH"),
            ("gemini-3-pro-preview", "minimal", "LOW"),
            ("gemini-4-neon", "xhigh", "HIGH"),
            ("gemini-4-neon", "minimal", "LOW"),
            ("gemini-3.1-flash-lite-image", "low", "MINIMAL"),
            ("gemini-3.1-flash-lite", "low", "LOW"),
            ("gemini-3.1-pro-preview", "minimal", "LOW"),
            ("gemini-3.8-flash", "none", "LOW"),
            ("gemini-future-9", "none", "MINIMAL"),
            ("gemini-future-9", "xhigh", "HIGH"),
            ("gemini-future-9", "medium", "MEDIUM"),
        ],
    )
    def test_generic_reasoning_effort_maps_to_a_supported_thinking_level(
        self, model, level, native_level
    ):
        cfg = GoogleGenerationConfig(model=model, reasoning_effort=level)
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.thinking_config.thinking_level.value == native_level
        assert native.thinking_config.thinking_budget is None

    def test_a_clamped_level_is_logged(self, caplog):
        cfg = GoogleGenerationConfig(model="gemini-3-pro", reasoning_effort="xhigh")
        with caplog.at_level("DEBUG"):
            _to_native_config(cfg, None, None, lambda t: t)
        assert "'xhigh'" in caplog.text
        assert "'high'" in caplog.text

    def test_provider_thinking_budget_wins_over_generic_reasoning_effort(self):
        cfg = GoogleGenerationConfig(
            model="gemini-3-pro", thinking_budget=7, reasoning_effort="high"
        )
        native = _to_native_config(cfg, None, None, lambda t: t)
        assert native.thinking_config.thinking_budget == 7
        assert native.thinking_config.thinking_level is None

    def test_no_thinking_config_by_default(self):
        native = _to_native_config(GoogleGenerationConfig(), None, None, lambda t: t)
        assert native.thinking_config is None

    def test_parallel_tool_calls_with_tools_warns_and_is_not_sent(self, caplog):
        cfg = GoogleGenerationConfig(parallel_tool_calls=False)
        with caplog.at_level("WARNING"):
            native = _to_native_config(cfg, None, [object()], lambda t: t)
        assert "parallel_tool_calls is ignored" in caplog.text
        assert native.tool_config is None

    def test_parallel_tool_calls_without_tools_does_not_warn(self, caplog):
        cfg = GoogleGenerationConfig(parallel_tool_calls=False)
        with caplog.at_level("WARNING"):
            _to_native_config(cfg, None, None, lambda t: t)
        assert caplog.text == ""


class TestFromNativeResponse:
    def test_thought_parts_are_not_part_of_the_answer(self):
        response = _response(
            [_part(text="Reasoning summary", thought=True), _part(text="Answer")]
        )
        result = _from_native_response(response, "gemini-2.5-flash")
        assert result.message.content == "Answer"

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

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
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

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
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

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
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

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(prompt)
        assert caught.value.code == "provider_unavailable"


class TestGoogleLLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Gemini repeats the cumulative usage on every chunk.
        chunks = [
            _response(
                [_part(text="Hello")],
                prompt_tokens=5,
                candidates_tokens=4,
                finish_reason=None,
            ),
            _response(
                [_part(text=" World")],
                prompt_tokens=5,
                candidates_tokens=10,
                finish_reason="MAX_TOKENS",
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt)]

        assert [c.delta for c in results] == ["Hello", " World", ""]
        assert [c.usage for c in results if c.usage is not None] == [
            TokenUsage(input_tokens=5, output_tokens=10)
        ]
        assert results[-1].finish_reason == FinishReason.LENGTH

    async def test_stream_keeps_thought_parts_out_of_the_text(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def _gen():
            yield _response([_part(text="Thinking...", thought=True)])
            yield _response([_part(text="Answer")])

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt)]

        assert "".join(c.delta for c in results) == "Answer"

    async def test_stream_tool_call_indices_count_across_chunks(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.llm.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def _gen():
            yield _response(
                [
                    _part(text="Let me look."),
                    _part(function_call=_function_call(None, "search", {"q": "a"})),
                ]
            )
            yield _response(
                [_part(function_call=_function_call(None, "lookup", {"q": "b"}))]
            )

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gen())

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt)]

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert [(d.index, d.name) for d in deltas] == [(0, "search"), (1, "lookup")]

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

        provider = GoogleLLM(_creds(), model="gemini-2.5-flash")
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        results = [c async for c in provider.stream(prompt, tools=[_SchemaTool()])]

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].name == "search"
        assert deltas[0].arguments_delta == '{"query": "hi"}'
