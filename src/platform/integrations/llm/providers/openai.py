from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from integrations.llm.config import GenerationConfig
from bridges.langchain.generation_config import to_langchain_openai
from integrations.llm.langchain_base import LangChainLLMProvider


class OpenAILLM(LangChainLLMProvider):

    def __init__(self, api_key: str) -> None:
        self._api_key = SecretStr(api_key)


    def _client(
        self,
        model: str,
        config: GenerationConfig | None
    ) -> ChatOpenAI:
        
        cfg = config or GenerationConfig()

        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            **to_langchain_openai(cfg),
        )