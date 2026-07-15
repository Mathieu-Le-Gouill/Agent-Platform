from langchain_ollama import OllamaEmbeddings
from typing import Any

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig
from agent_platform.integrations.credentials.ollama import OllamaCredentials
from agent_platform.core.credentials import (
    resolve_timeout,
)


class OllamaEmbeddingProvider(
    LangChainEmbedder[OllamaCredentials, OllamaEmbeddingConfig]
):
    def __init__(self, credentials: OllamaCredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else OllamaCredentials()
        )

    def _client(self, config: OllamaEmbeddingConfig) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=config.model,
            base_url=self._credentials.base_url,
            **_to_langchain_ollama(config, self._credentials),
        )

    def _default_config(self) -> OllamaEmbeddingConfig:
        return OllamaEmbeddingConfig()


def _to_langchain_ollama(
    config: OllamaEmbeddingConfig,
    credentials: OllamaCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.dimensions is not None:
        params["dimensions"] = config.dimensions
    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout

    return params
