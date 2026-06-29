from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from integrations.llm.config import GenerationConfig
from bridges.langchain.generation_config import to_langchain_anthropic
from integrations.llm.langchain_base import LangChainLLMProvider


class AnthropicLLM(LangChainLLMProvider):

    def __init__(self, api_key: str) -> None:
        self._api_key = SecretStr(api_key)


    def _client(
        self,
        model: str,
        config: GenerationConfig | None
    ) -> ChatAnthropic:
        
        cfg = config or GenerationConfig()

        return ChatAnthropic(
            model_name=model,
            api_key=self._api_key,
            **to_langchain_anthropic(cfg),
        )