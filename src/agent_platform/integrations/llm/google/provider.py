from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# google-genai sends async calls through aiohttp whenever it is installed, and
# aiohttp's connection errors are neither `ConnectionError` nor httpx errors.
try:
    from aiohttp import ClientConnectionError as _AiohttpConnectionError

    _AIOHTTP_CONNECTION_ERROR: type[Exception] | None = _AiohttpConnectionError
except ImportError:  # pragma: no cover - aiohttp is optional
    _AIOHTTP_CONNECTION_ERROR = None

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig
from agent_platform.integrations.llm.google.mappers import (
    from_native_response,
    to_native_config,
    to_native_contents,
)

if TYPE_CHECKING:
    from agent_platform.core.interfaces.llm.tool_spec import ToolSpec


class GoogleLLM(
    NativeLLMProvider[
        GoogleGenerationConfig,
        genai.Client,
        genai.Client,
        types.GenerateContentResponse,
    ]
):
    _provider_name = "google"
    _missing_api_key_message = "Google API key is required but was not provided"
    # The first of the names `GoogleCredentials` reads.
    _api_key_env = "GOOGLE_API_KEY"
    _config_cls = GoogleGenerationConfig

    def __init__(
        self,
        credentials: GoogleCredentials | None = None,
        client_options: ClientOptions | None = None,
        *,
        model: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, GoogleCredentials)
        self._client_options = resolve_client_options(client_options)
        self._bind(model, options)

    def _tool_to_schema(self, tool: ToolSpec) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=model_schema(tool.input_schema),
                )
            ]
        )

    def _client(self, config: GoogleGenerationConfig) -> genai.Client:
        api_key = require_secret(
            self._credentials.api_key, self._missing_api_key_message
        )
        return genai.Client(
            api_key=api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(self._timeout(config) * 1000)),
        )

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        if _AIOHTTP_CONNECTION_ERROR is not None and isinstance(
            exc, _AIOHTTP_CONNECTION_ERROR
        ):
            return True
        return super()._is_transient(exc)

    def _status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, genai_errors.APIError):
            return exc.code
        return None

    def _is_timeout(self, exc: Exception) -> bool:
        return isinstance(exc, httpx.TimeoutException) or super()._is_timeout(exc)

    def _async_client(self, config: GoogleGenerationConfig) -> genai.Client:
        # Single client class, no sync/async split (like Mistral): async calls
        # go through `client.aio.models.*`, sync through `client.models.*`.
        return self._client(config)

    def _sync_client(self, config: GoogleGenerationConfig) -> genai.Client:
        return self._client(config)

    def _invoke_sync(
        self,
        client: genai.Client,
        prompt: Prompt,
        config: GoogleGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> types.GenerateContentResponse:
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(config, system, tools, self._tool_to_schema)
        return client.models.generate_content(
            model=config.model,
            contents=contents,
            config=native_config,
        )

    async def _invoke_async(
        self,
        client: genai.Client,
        prompt: Prompt,
        config: GoogleGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> types.GenerateContentResponse:
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(config, system, tools, self._tool_to_schema)
        return await client.aio.models.generate_content(
            model=config.model,
            contents=contents,
            config=native_config,
        )

    def _from_native(
        self, response: types.GenerateContentResponse, model: str
    ) -> LLMResponse:
        return from_native_response(response, model)

    async def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resolved = self._resolve(config)
        client = self._async_client(resolved)
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(resolved, system, tools, self._tool_to_schema)

        with self._observe(resolved) as call:
            usage_totals = TokenUsage.zero()
            events = await client.aio.models.generate_content_stream(
                model=resolved.model,
                contents=contents,
                config=native_config,
            )
            async for chunk in events:
                if chunk.candidates:
                    content = chunk.candidates[0].content
                    parts = content.parts if content else None
                    for idx, part in enumerate(parts or []):
                        if part.text:
                            yield StreamChunk(delta=part.text)
                        if part.function_call:
                            fc = part.function_call
                            # Gemini emits each function call whole in one
                            # chunk (no argument fragmentation like OpenAI/
                            # Anthropic), and has no native call id.
                            yield StreamChunk(
                                delta="",
                                tool_call_deltas=[
                                    ToolCallDelta(
                                        index=idx,
                                        id=fc.id or uuid4().hex,
                                        name=fc.name,
                                        arguments_delta=json.dumps(dict(fc.args or {})),
                                    )
                                ],
                            )
                if chunk.usage_metadata is not None:
                    chunk_usage = TokenUsage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    )
                    usage_totals = chunk_usage
                    yield StreamChunk(
                        delta="",
                        finish_reason=FinishReason.STOP,
                        usage=chunk_usage,
                    )

            self._record_usage(call, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)
