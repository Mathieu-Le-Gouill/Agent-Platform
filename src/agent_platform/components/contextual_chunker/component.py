from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from agent_platform.components.base import Component
from agent_platform.components.chunker.component import Chunker, ChunkerInput
from agent_platform.components.contextual_chunker.config import ContextualChunkerConfig
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.message import Prompt

GenConfigT = TypeVar("GenConfigT", bound=GenerationConfig)


class ContextualChunkerInput(NamedTuple, Generic[GenConfigT]):
    documents: list[TextDocument]
    config: ContextualChunkerConfig | None
    generation_config: GenConfigT | None


class ContextualChunker(
    Component[ContextualChunkerInput[GenConfigT], list[TextChunk]],
    Generic[GenConfigT],
):
    def __init__(
        self,
        chunker: Chunker,
        llm: BaseLLMProvider[GenConfigT],
    ) -> None:
        self._chunker = chunker
        self._llm = llm

    async def arun(self, input: ContextualChunkerInput[GenConfigT]) -> list[TextChunk]:
        documents, config, generation_config = input
        config = config or ContextualChunkerConfig()
        result: list[TextChunk] = []
        for doc in documents:
            chunks = await self._chunker.arun(ChunkerInput([doc], None))
            document_text = doc.text[: config.max_document_chars]
            for chunk in chunks:
                context = await self._generate_context(
                    document_text, chunk.text, config, generation_config
                )
                result.append(
                    chunk.model_copy(
                        update={
                            "text": f"{context}\n\n{chunk.text}"
                            if context
                            else chunk.text,
                            "metadata": {**chunk.metadata, "context": context},
                        }
                    )
                )
        return result

    async def _generate_context(
        self,
        document: str,
        chunk_text: str,
        config: ContextualChunkerConfig,
        generation_config: GenConfigT | None,
    ) -> str:
        prompt = Prompt.build(
            user=config.context_prompt_template.format(
                document=document, chunk=chunk_text
            )
        )
        response = await self._llm.agenerate(prompt, generation_config)
        if response.message is None:
            return ""
        return response.message.text.strip()
