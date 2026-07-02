from abc import abstractmethod
from typing import AsyncIterator

from agent_platform.integrations.llm.prompt import Prompt
from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.integrations.llm.response import LLMResponse, StreamChunk, FinishReason
from agent_platform.models.token import TokenUsage

from agent_platform.integrations.llm.base import BaseLLMProvider
from agent_platform.bridges.langchain.prompt import to_langchain
from agent_platform.bridges.langchain.response import from_langchain
from langchain_core.language_models.chat_models import BaseChatModel


class LangChainLLMProvider(BaseLLMProvider):

    @abstractmethod
    def _client(self, model: str, config: GenerationConfig | None) -> BaseChatModel:
        ...

    @abstractmethod
    def _to_prompt(self, prompt: Prompt):
        ...

    @abstractmethod
    def _from_response(self, response, model: str) -> LLMResponse:
        ...

    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        lc = self._client(model, config)
        response = await lc.ainvoke(to_langchain(prompt))

        return from_langchain(response, model)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None
    ) -> AsyncIterator[StreamChunk]:
        
        lc = self._client(model, config)

        async for chunk in lc.astream(to_langchain(prompt)):
            content = chunk.content

            if isinstance(content, str):
                if content:
                    yield StreamChunk(delta=content)

            elif isinstance(content, list):
                for item in content:

                    if isinstance(item, str) and item:
                        yield StreamChunk(delta=item)

                    elif isinstance(item, dict):
                        text = item.get("text", "")
                        if isinstance(text, str) and text:
                            yield StreamChunk(delta=text)
            
            usage = chunk.usage_metadata
            if usage:
                yield StreamChunk(
                    delta="",
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
                )

        yield StreamChunk(delta="", finish_reason=FinishReason.STOP)