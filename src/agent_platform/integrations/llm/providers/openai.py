from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.bridges.langchain.generation_config import to_langchain_openai
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider


class OpenAILLM(LangChainLLMProvider):

    def __init__(self, api_key: SecretStr) -> None:
        self._api_key = api_key


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