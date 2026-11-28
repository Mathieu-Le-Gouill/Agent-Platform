import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.generate_image.tool import GenerateImageInput
from agent_platform.core.credentials import ClientOptions
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.batch import BatchRequest, BatchStatus
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.enums import FinishReason
from agent_platform.core.schemas.message import Prompt, UserMessage
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.provider import OpenAILLM
from tests.helpers import make_chat_completion, make_chat_completion_chunk


def _creds(key: str = "sk-test") -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr(key))


class TestOpenAILLMConstruction:
    def test_construct(self):
        provider = OpenAILLM(_creds("sk-123"))
        assert provider._credentials.api_key.get_secret_value() == "sk-123"

    def test_the_bound_model_and_options_resolve_into_the_provider_config(self):
        provider = OpenAILLM(model="gpt-4.1", options={"temperature": 0.3})
        cfg = provider._resolve(None)
        assert isinstance(cfg, OpenAIGenerationConfig)
        assert (cfg.model, cfg.temperature) == ("gpt-4.1", 0.3)

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

    @staticmethod
    def _tool_with(input_schema):
        class _T(Tool):
            name = "t"
            description = "d"

            async def run(self, **kwargs):
                return "ok"

        _T.input_schema = input_schema
        return _T()

    def test_tool_with_defaulted_fields_is_not_strict(self):
        function = OpenAILLM(_creds())._tool_to_schema(
            self._tool_with(GenerateImageInput)
        )["function"]

        assert function["strict"] is False
        assert function["parameters"] == model_schema(GenerateImageInput)

    def test_tool_with_all_required_fields_is_strict_on_every_object(self):
        class _Inner(BaseModel):
            city: str

        class _Outer(BaseModel):
            query: str
            inner: _Inner

        original = copy.deepcopy(model_schema(_Outer))

        function = OpenAILLM(_creds())._tool_to_schema(self._tool_with(_Outer))[
            "function"
        ]

        params = function["parameters"]
        assert function["strict"] is True
        assert params["additionalProperties"] is False
        assert params["$defs"]["_Inner"]["additionalProperties"] is False
        assert model_schema(_Outer) == original

    def test_parameterless_tool_is_strict_with_empty_required(self):
        function = OpenAILLM(_creds())._tool_to_schema(self._tool_with(BaseModel))[
            "function"
        ]

        assert function["strict"] is True
        assert function["parameters"] == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_tool_with_required_dict_field_is_not_strict(self):
        class _WithDict(BaseModel):
            options: dict[str, str]

        function = OpenAILLM(_creds())._tool_to_schema(self._tool_with(_WithDict))[
            "function"
        ]

        assert function["strict"] is False

    def test_tool_with_required_any_field_is_not_strict(self):
        class _WithAny(BaseModel):
            payload: Any

        function = OpenAILLM(_creds())._tool_to_schema(self._tool_with(_WithAny))[
            "function"
        ]

        assert function["strict"] is False

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
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 120.0

    def test_client_kwargs_explicit_timeout(self):
        provider = OpenAILLM(_creds())
        cfg = OpenAIGenerationConfig(timeout=15.0)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0

    def test_generate_clients_make_no_sdk_retries(self, mocker):
        async_client = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        sync_client = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.OpenAI"
        )
        provider = OpenAILLM(_creds(), ClientOptions(max_retries=5))
        cfg = OpenAIGenerationConfig()

        provider._async_client(cfg)
        provider._sync_client(cfg)

        assert async_client.call_args.kwargs["max_retries"] == 0
        assert sync_client.call_args.kwargs["max_retries"] == 0

    def test_stream_client_keeps_the_configured_sdk_retries(self, mocker):
        async_client = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        provider = OpenAILLM(_creds(), ClientOptions(max_retries=5))

        provider._stream_client(OpenAIGenerationConfig())
        assert async_client.call_args.kwargs["max_retries"] == 5

        provider._stream_client(OpenAIGenerationConfig(max_retries=2))
        assert async_client.call_args.kwargs["max_retries"] == 2
        assert async_client.call_args.kwargs["timeout"] == 120.0

    @staticmethod
    def _status_error(status: int) -> openai.APIStatusError:
        response = httpx.Response(status, request=httpx.Request("POST", "http://x"))
        return openai.APIStatusError("boom", response=response, body=None)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_client_error_is_not_transient(self, status):
        assert OpenAILLM(_creds())._is_transient(self._status_error(status)) is False

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
    def test_a_temporary_status_is_transient(self, status):
        assert OpenAILLM(_creds())._is_transient(self._status_error(status)) is True

    def test_a_timeout_or_connection_error_is_transient(self):
        request = httpx.Request("POST", "http://x")
        provider = OpenAILLM(_creds())

        assert provider._is_transient(openai.APITimeoutError(request=request))
        assert provider._is_transient(openai.APIConnectionError(request=request))
        assert provider._is_transient(ValueError("other")) is False


class TestOpenAILLMGenerate:
    def test_sync_generate(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_chat_completion(
            "Sync hello"
        )

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
        mock_client.chat.completions.create.return_value = make_chat_completion("ok")

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
            return_value=make_chat_completion("Async hello")
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
        mock_client.chat.completions.create = AsyncMock(
            return_value=make_chat_completion("ok")
        )

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
            return make_chat_completion("ok")

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
        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(
                prompt, config=OpenAIGenerationConfig(model="gpt-4.1")
            )
        assert caught.value.code == "provider_unavailable"


class TestOpenAILLMStream:
    async def test_stream_yields_text_and_usage(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        chunks = [
            make_chat_completion_chunk(content="Hello"),
            make_chat_completion_chunk(content=" World", finish_reason="length"),
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

        assert [c.delta for c in results] == ["Hello", " World", ""]
        assert [c.usage for c in results[:-1]] == [None, None]
        assert results[-1].usage == TokenUsage(input_tokens=5, output_tokens=10)
        assert results[-1].finish_reason == FinishReason.LENGTH

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

    async def test_stream_with_tools_yields_tool_call_deltas(self, mocker):
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

        chunks = [
            make_chat_completion_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(name="search", arguments=""),
                    )
                ]
            ),
            make_chat_completion_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(
                            name=None, arguments='{"query": "hi"}'
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            ),
        ]

        async def _gen():
            for chunk in chunks:
                yield chunk

        mock_client.chat.completions.create = AsyncMock(return_value=_gen())

        provider = OpenAILLM(_creds())
        prompt = Prompt(messages=[UserMessage(content="Hi")])
        config = OpenAIGenerationConfig(model="gpt-4.1")
        results = [
            c
            async for c in provider.stream(prompt, config=config, tools=[_SchemaTool()])
        ]

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["tools"][0]["function"]["name"] == "search"

        deltas = [d for c in results for d in c.tool_call_deltas]
        assert deltas[0].index == 0
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "search"
        assert deltas[1].arguments_delta == '{"query": "hi"}'
        assert results[-1].finish_reason == FinishReason.TOOL_CALL
        assert [c.usage for c in results if c.usage is not None] == [
            TokenUsage(input_tokens=1, output_tokens=1)
        ]


class TestOpenAILLMBatch:
    async def test_submit_batch(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file_1"))
        mock_client.batches.create = AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1",
                status="validating",
                request_counts=SimpleNamespace(total=1, completed=0),
            )
        )

        provider = OpenAILLM(_creds(), model="gpt-4.1")
        requests = [
            BatchRequest(
                custom_id="r1", prompt=Prompt(messages=[UserMessage(content="Hi")])
            )
        ]
        job = await provider.submit_batch(requests)

        assert job.id == "batch_1"
        assert job.status == BatchStatus.PENDING
        assert job.request_count == 1

        _, kwargs = mock_client.files.create.call_args
        assert kwargs["purpose"] == "batch"
        _, batch_kwargs = mock_client.batches.create.call_args
        assert batch_kwargs["input_file_id"] == "file_1"

    async def test_get_batch_status(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(
                id="batch_1", status="completed", request_counts=None
            )
        )

        provider = OpenAILLM(_creds())
        job = await provider.get_batch_status("batch_1")

        assert job.status == BatchStatus.COMPLETED
        assert job.request_count is None

    async def test_fetch_batch_results(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.llm.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.batches.retrieve = AsyncMock(
            return_value=SimpleNamespace(id="batch_1", output_file_id="out_1")
        )
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        result_line = json.dumps(
            {"custom_id": "r1", "response": {"body": body}, "error": None}
        )
        error_line = json.dumps(
            {"custom_id": "r2", "response": None, "error": {"message": "boom"}}
        )
        mock_client.files.content = AsyncMock(
            return_value=SimpleNamespace(text=f"{result_line}\n{error_line}\n")
        )

        provider = OpenAILLM(_creds())
        results = await provider.fetch_batch_results("batch_1")

        assert results[0].custom_id == "r1"
        assert results[0].response is not None
        assert results[0].response.message.content == "hi"
        assert results[1].custom_id == "r2"
        assert results[1].error is not None
