from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from agent_platform.core.schemas import model_schema
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.credentials.huggingface import HuggingFaceCredentials
from agent_platform.core.credentials import (
    resolve_max_retries,
    resolve_timeout,
)

from agent_platform.core.errors import MissingCredentialError

from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class HuggingFaceLLM(
    LangChainLLMProvider[HuggingFaceCredentials, HuggingFaceGenerationConfig]
):
    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else HuggingFaceCredentials()
        )

    def _tool_to_schema(self, tool: "Tool") -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }

    def _client(self, config: HuggingFaceGenerationConfig) -> ChatHuggingFace:
        if self._credentials.api_key is None:
            raise MissingCredentialError(
                "Hugging Face Hub API token is required but was not provided"
            )

        llm = HuggingFaceEndpoint(
            task=config.task,
            repo_id=config.repo_id,
            provider=config.provider,
            huggingfacehub_api_token=self._credentials.api_key.get_secret_value(),
            **_to_langchain_hugging_face(config, self._credentials),
        )

        return ChatHuggingFace(llm=llm)

    def _default_config(self) -> HuggingFaceGenerationConfig:
        return HuggingFaceGenerationConfig()


def _to_langchain_hugging_face(
    config: HuggingFaceGenerationConfig,
    credentials: HuggingFaceCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.temperature:
        params["temperature"] = config.temperature
    if config.max_tokens:
        params["max_new_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = int(timeout)
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    return params
