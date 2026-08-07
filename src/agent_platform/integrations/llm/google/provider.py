from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from google import genai
from google.genai import types

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import record_token_usage
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig
from agent_platform.integrations.llm.google.mappers import (
    from_native_response,
    to_native_config,
    to_native_contents,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


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

    def __init__(self, credentials: GoogleCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, GoogleCredentials)

    def _tool_to_schema(self, tool: Tool) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=model_schema(tool.input_schema),
                )
            ]
        )

    def _default_config(self) -> GoogleGenerationConfig:
        return GoogleGenerationConfig()

    def _client(self, config: GoogleGenerationConfig) -> genai.Client:
        api_key = require_secret(
            self._credentials.api_key, self._missing_api_key_message
        )
        return genai.Client(api_key=api_key.get_secret_value())

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
        tools: list[Tool] | None,
    ) -> types.GenerateContentResponse:
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(config, system, tools, self._tool_to_schema)
        return client.models.generate_content(
            model=config.model,
            contents=contents,  # type: ignore[arg-type]
            config=native_config,
        )

    async def _invoke_async(
        self,
        client: genai.Client,
        prompt: Prompt,
        config: GoogleGenerationConfig,
        tools: list[Tool] | None,
    ) -> types.GenerateContentResponse:
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(config, system, tools, self._tool_to_schema)
        return await client.aio.models.generate_content(
            model=config.model,
            contents=contents,  # type: ignore[arg-type]
            config=native_config,
        )

    def _from_native(
        self, response: types.GenerateContentResponse, model: str
    ) -> LLMResponse:
        return from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: GoogleGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        system, contents = to_native_contents(prompt)
        native_config = to_native_config(config, system, tools, self._tool_to_schema)

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = await client.aio.models.generate_content_stream(
                model=config.model,
                contents=contents,  # type: ignore[arg-type]
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

            record_token_usage(span, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)
