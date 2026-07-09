from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Any

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)
from agent_platform.integrations.credentials import (
    HuggingFaceCredentials,
    resolve_timeout,
)
from agent_platform.core.errors import MissingCredentialError


class HuggingFaceEmbeddingProvider(
    LangChainEmbedder[HuggingFaceCredentials, HuggingFaceEmbeddingConfig]
):

    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        super().__init__(credentials if credentials is not None else HuggingFaceCredentials())

    def _client(self, config: HuggingFaceEmbeddingConfig) -> Embeddings:
        if config.mode is HuggingFaceEmbeddingMode.HOSTED:
            if self._credentials.api_key is None:
                raise MissingCredentialError(
                    "Hugging Face Hub API token is required for hosted inference"
                )
            return HuggingFaceEndpointEmbeddings(
                model=config.model,
                huggingfacehub_api_token=self._credentials.api_key.get_secret_value(),
                **_to_langchain_huggingface_hosted(config, self._credentials),
            )

        return HuggingFaceEmbeddings(
            model_name=config.model,
            **_to_langchain_huggingface_local(config),
        )

    def _default_config(self) -> HuggingFaceEmbeddingConfig: 
        return HuggingFaceEmbeddingConfig()


def _to_langchain_huggingface_local(
    config: HuggingFaceEmbeddingConfig,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.model_kwargs is not None:
        params["model_kwargs"] = config.model_kwargs
    if config.encode_kwargs is not None:
        params["encode_kwargs"] = config.encode_kwargs
    return params


def _to_langchain_huggingface_hosted(
    config: HuggingFaceEmbeddingConfig,
    credentials: HuggingFaceCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.provider is not None:
        params["provider"] = config.provider

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout

    return params