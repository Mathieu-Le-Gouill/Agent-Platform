from langchain_mistralai import MistralAIEmbeddings
from typing import Any

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig
from agent_platform.integrations.credentials.mistral import MistralCredentials
from agent_platform.core.credentials import (
    resolve_timeout,
    resolve_max_retries,
)

from agent_platform.core.errors import MissingCredentialError


class MistralEmbeddingProvider(
    LangChainEmbedder[MistralCredentials, MistralEmbeddingConfig]
):
    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else MistralCredentials()
        )

    def _client(self, config: MistralEmbeddingConfig) -> MistralAIEmbeddings:

        if self._credentials.api_key is None:
            raise MissingCredentialError(
                "Mistral API key is required but was not provided"
            )

        return MistralAIEmbeddings(
            model=config.model,
            api_key=self._credentials.api_key,
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

    if config.dimensions is not None:
        params["dimensions"] = config.dimensions
    if config.wait_time is not None:
        params["wait_time"] = config.wait_time
    if config.max_concurrent_requests is not None:
        params["max_concurrent_requests"] = config.max_concurrent_requests

    return params
