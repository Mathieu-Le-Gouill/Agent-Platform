from langchain_openai import OpenAIEmbeddings
from typing import Any

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig
from agent_platform.integrations.credentials.openai import OpenAICredentials
from agent_platform.core.credentials import (
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import MissingCredentialError


class OpenAIEmbeddingProvider(
    LangChainEmbedder[OpenAICredentials, OpenAIEmbeddingConfig]
):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else OpenAICredentials()
        )

    def _client(self, config: OpenAIEmbeddingConfig) -> OpenAIEmbeddings:

        if self._credentials.api_key is None:
            raise MissingCredentialError(
                "OpenAI API key is required but was not provided"
            )

        return OpenAIEmbeddings(
            model=config.model,
            api_key=self._credentials.api_key,
            **_to_langchain_openai(config, self._credentials),
        )

    def _default_config(self) -> OpenAIEmbeddingConfig:
        return OpenAIEmbeddingConfig()


def _to_langchain_openai(
    config: OpenAIEmbeddingConfig,
    credentials: OpenAICredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model_kwargs": config.model_kwargs}
    if config.dimensions is not None:
        params["dimensions"] = config.dimensions

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    if credentials.organization is not None:
        params["organization"] = credentials.organization

    return params
