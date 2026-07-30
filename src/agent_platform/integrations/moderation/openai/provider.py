from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.moderation_create_response import ModerationCreateResponse

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.moderation.response import (
    ModerationCategory,
    ModerationResult,
)
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.moderation._base import NativeModerationProvider
from agent_platform.integrations.moderation.openai.config import OpenAIModerationConfig


class OpenAIModeration(
    NativeModerationProvider[
        OpenAIModerationConfig, AsyncOpenAI, OpenAI, ModerationCreateResponse
    ]
):
    def __init__(
        self,
        credentials: OpenAICredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> OpenAIModerationConfig:
        return OpenAIModerationConfig()

    def _client_kwargs(self, config: OpenAIModerationConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "OpenAI API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "base_url": self._client_options.base_url,
            "max_retries": resolve_max_retries(
                config.max_retries, self._client_options
            ),
        }
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _async_client(self, config: OpenAIModerationConfig) -> AsyncOpenAI:
        return AsyncOpenAI(**self._client_kwargs(config))

    def _sync_client(self, config: OpenAIModerationConfig) -> OpenAI:
        return OpenAI(**self._client_kwargs(config))

    def _invoke_sync(
        self, client: OpenAI, text: str, config: OpenAIModerationConfig
    ) -> ModerationCreateResponse:
        return client.moderations.create(model=config.model, input=text)

    async def _invoke_async(
        self, client: AsyncOpenAI, text: str, config: OpenAIModerationConfig
    ) -> ModerationCreateResponse:
        return await client.moderations.create(model=config.model, input=text)

    def _from_native(self, result: ModerationCreateResponse) -> ModerationResult:
        moderation = result.results[0]
        categories = moderation.categories.model_dump()
        scores = moderation.category_scores.model_dump()
        return ModerationResult(
            flagged=moderation.flagged,
            categories=[
                ModerationCategory(
                    name=name,
                    flagged=bool(flagged),
                    score=Score.confidence(scores.get(name, 0.0)),
                )
                for name, flagged in categories.items()
            ],
        )
