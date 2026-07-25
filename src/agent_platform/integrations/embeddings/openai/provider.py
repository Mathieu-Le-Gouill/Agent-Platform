from typing import Any

from langchain_openai import OpenAIEmbeddings

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(LangChainEmbedder[OpenAIEmbeddingConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _client(self, config: OpenAIEmbeddingConfig) -> OpenAIEmbeddings:
        api_key = require_secret(
            self._credentials.api_key,
            "OpenAI API key is required but was not provided",
        )

        return OpenAIEmbeddings(
            model=config.model,
            api_key=api_key,
            **_to_langchain_openai(config, self._credentials),
        )

    def _default_config(self) -> OpenAIEmbeddingConfig:
        return OpenAIEmbeddingConfig()


def _to_langchain_openai(
    config: OpenAIEmbeddingConfig,
    credentials: OpenAICredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    # `OpenAIEmbeddings.model_kwargs` defaults to `dict` (non-Optional), so
    # `None` must never be passed through directly. `encoding_format` has no
    # dedicated field on the client and is routed through this same channel.
    if config.model_kwargs is not None or config.encoding_format is not None:
        model_kwargs: dict[str, Any] = dict(config.model_kwargs or {})
        if config.encoding_format is not None:
            model_kwargs["encoding_format"] = config.encoding_format
        params["model_kwargs"] = model_kwargs

    if config.dimensions is not None:
        params["dimensions"] = config.dimensions

    # Maps the base `batch_size` field to the client's request-batching knob.
    params["chunk_size"] = config.batch_size
    params["check_embedding_ctx_length"] = config.check_embedding_ctx_length

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    if credentials.organization is not None:
        params["organization"] = credentials.organization

    return params
