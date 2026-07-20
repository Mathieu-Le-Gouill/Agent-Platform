from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Any

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.core.errors import MissingCredentialError


class HuggingFaceEmbeddingProvider(LangChainEmbedder[HuggingFaceEmbeddingConfig]):
    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else HuggingFaceCredentials()
        )

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

    # `HuggingFaceEndpointEmbeddings` has no `timeout` field (verified
    # against the installed package source) and `InferenceClient` offers no
    # equivalent kwargs channel for hosted embeddings either, so `timeout` is
    # intentionally dropped in hosted mode.

    model_kwargs: dict[str, Any] = {}
    if config.model_kwargs is not None:
        model_kwargs.update(config.model_kwargs)
    if config.encode_kwargs is not None:
        model_kwargs.update(config.encode_kwargs)
    if config.dimensions is not None:
        model_kwargs["dimensions"] = config.dimensions
    if config.truncate is not None:
        model_kwargs["truncate"] = config.truncate
    if config.normalize is not None:
        model_kwargs["normalize"] = config.normalize
    if model_kwargs:
        params["model_kwargs"] = model_kwargs

    return params
