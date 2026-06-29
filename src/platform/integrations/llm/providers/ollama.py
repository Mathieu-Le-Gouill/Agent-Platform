from langchain_ollama import ChatOllama
from integrations.llm.config import GenerationConfig
from bridges.langchain.generation_config import to_langchain_ollama
from integrations.llm.langchain_base import LangChainLLMProvider

class OllamaLLM(LangChainLLMProvider):

    def _client(
        self,
        model: str,
        config: GenerationConfig | None
    ) -> ChatOllama:
        
        cfg = config or GenerationConfig()

        return ChatOllama(
            model=model,
            **to_langchain_ollama(cfg),
        )
