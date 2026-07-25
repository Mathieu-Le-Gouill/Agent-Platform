from typing import Any

from langchain_ollama import OllamaEmbeddings

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig


class OllamaEmbeddingProvider(LangChainEmbedder[OllamaEmbeddingConfig]):
    def __init__(self, credentials: OllamaCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OllamaCredentials)

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
    if config.keep_alive is not None:
        params["keep_alive"] = config.keep_alive

    # `OllamaEmbeddings` has `extra="forbid"` and no top-level `timeout`
    # field; route it through `client_kwargs`, which is merged into the
    # underlying `ollama.Client`/`AsyncClient` (httpx-based) constructor.
    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["client_kwargs"] = {"timeout": timeout}

    return params
