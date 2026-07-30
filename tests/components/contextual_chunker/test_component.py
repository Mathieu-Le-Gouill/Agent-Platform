import pytest

from agent_platform.components.chunker.component import Chunker
from agent_platform.components.contextual_chunker.component import ContextualChunker
from agent_platform.components.contextual_chunker.config import ContextualChunkerConfig
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.core.schemas.token import TokenUsage


class _FakeChunker(Chunker):
    def __init__(self, chunks: list[TextChunk]) -> None:
        self._chunks = chunks

    async def arun(self, input):
        return self._chunks


class _FakeLLM(BaseLLMProvider):
    def __init__(self, context: str = "This chunk covers section one.") -> None:
        self._context = context
        self.calls: list = []

    def generate(self, prompt, config=None, tools=None):
        raise NotImplementedError

    async def agenerate(self, prompt, config=None, tools=None):
        self.calls.append(prompt)
        return LLMResponse(
            message=AssistantMessage(content=self._context),
            usage=TokenUsage(),
            model="fake",
        )

    def stream(self, prompt, config=None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_prepends_generated_context_to_each_chunk():
    doc = TextDocument(text="Full document body.", format=DocumentFormat.TXT)
    chunk = TextChunk(text="Chunk body.", document_id=doc.id, index=0)
    chunker = _FakeChunker([chunk])
    llm = _FakeLLM("Context sentence.")
    contextual_chunker = ContextualChunker(chunker, llm)

    result = await contextual_chunker.arun(([doc], None, None))

    assert len(result) == 1
    assert result[0].text == "Context sentence.\n\nChunk body."
    assert result[0].metadata["context"] == "Context sentence."
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_truncates_document_to_max_chars_in_prompt():
    doc = TextDocument(text="x" * 100, format=DocumentFormat.TXT)
    chunk = TextChunk(text="Chunk body.", document_id=doc.id, index=0)
    chunker = _FakeChunker([chunk])
    llm = _FakeLLM("Context.")
    config = ContextualChunkerConfig(max_document_chars=10)
    contextual_chunker = ContextualChunker(chunker, llm)

    await contextual_chunker.arun(([doc], config, None))

    prompt_text = llm.calls[0].last_user_message().content
    assert "x" * 100 not in prompt_text
    assert "x" * 10 in prompt_text


@pytest.mark.asyncio
async def test_empty_llm_response_leaves_chunk_text_unchanged():
    doc = TextDocument(text="Full document body.", format=DocumentFormat.TXT)
    chunk = TextChunk(text="Chunk body.", document_id=doc.id, index=0)
    chunker = _FakeChunker([chunk])

    class _EmptyLLM(_FakeLLM):
        async def agenerate(self, prompt, config=None, tools=None):
            return LLMResponse(message=None, usage=TokenUsage(), model="fake")

    contextual_chunker = ContextualChunker(chunker, _EmptyLLM())

    result = await contextual_chunker.arun(([doc], None, None))

    assert result[0].text == "Chunk body."
    assert result[0].metadata["context"] == ""
