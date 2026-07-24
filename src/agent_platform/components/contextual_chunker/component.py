from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.components.chunker import Chunker
from agent_platform.components.contextual_chunker.config import ContextualChunkerConfig
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.message import Prompt

GenConfigT = TypeVar("GenConfigT", bound=GenerationConfig)


class ContextualChunker(
    Component[list[TextDocument], list[TextChunk]],
    Generic[GenConfigT],
):
    def __init__(
        self,
        chunker: Chunker,
        llm: BaseLLMProvider[GenConfigT],
        config: ContextualChunkerConfig | None = None,
        generation_config: GenConfigT | None = None,
    ) -> None:
        self._chunker = chunker
        self._llm = llm
        self._config = config or ContextualChunkerConfig()
        self._generation_config = generation_config

    async def arun(self, input: list[TextDocument]) -> list[TextChunk]:
        result: list[TextChunk] = []
        for doc in input:
            chunks = await self._chunker.arun([doc])
            document_text = doc.text[: self._config.max_document_chars]
            for chunk in chunks:
                context = await self._generate_context(document_text, chunk.text)
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

    async def _generate_context(self, document: str, chunk_text: str) -> str:
        prompt = Prompt.build(
            user=self._config.context_prompt_template.format(
                document=document, chunk=chunk_text
            )
        )
        response = await self._llm.agenerate(prompt, self._generation_config)
        if response.message is None:
            return ""
        return response.message.text.strip()
