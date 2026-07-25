from typing import Any

from langchain_mistralai import MistralAIEmbeddings

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig


class MistralEmbeddingProvider(LangChainEmbedder[MistralEmbeddingConfig]):
    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)

    def _client(self, config: MistralEmbeddingConfig) -> MistralAIEmbeddings:
        api_key = require_secret(
            self._credentials.api_key,
            "Mistral API key is required but was not provided",
        )

        return MistralAIEmbeddings(
            model=config.model,
            api_key=api_key,
            **_to_langchain_mistral(config, self._credentials),
        )

    def _default_config(self) -> MistralEmbeddingConfig:
        return MistralEmbeddingConfig()


def _to_langchain_mistral(
    config: MistralEmbeddingConfig,
    credentials: MistralCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {"endpoint": config.endpoint}

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = int(timeout)

    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    # `dimensions` is intentionally never forwarded: `MistralAIEmbeddings` has
    # `extra="forbid"` and no such field (see config.py note).
    if config.wait_time is not None:
        params["wait_time"] = config.wait_time
    if config.max_concurrent_requests is not None:
        params["max_concurrent_requests"] = config.max_concurrent_requests

    return params
