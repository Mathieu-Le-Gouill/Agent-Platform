from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.schemas.message import (
    UserMessage,
)
from tests.helpers import make_fake_llm_response


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes the input"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return kwargs.get("text", "")


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(_EchoTool())
    return r


class TestConversationConstruction:
    def test_empty_history(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        assert agent.history == []

    def test_default_max_iterations(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        assert agent._executor.max_iterations == 10

    def test_custom_max_iterations(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm, max_iterations=3)
        assert agent._executor.max_iterations == 3

    def test_max_history_turns_default(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        assert agent._max_history_turns is None

    def test_conversation_id_default(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        assert agent.conversation_id is not None

    def test_conversation_id_custom(self, mock_llm):
        from uuid import uuid4

        cid = uuid4()
        agent = ConversationAgent(name="conv", llm=mock_llm, conversation_id=cid)
        assert agent.conversation_id == cid


class TestConversationHistory:
    def test_add_user_message(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        agent.add_user_message("Hello")
        assert len(agent.history) == 1
        assert agent.history[0].content == "Hello"
        assert agent.history[0].role.value == "user"

    def test_clear_history(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        agent.add_user_message("Hello")
        agent.clear_history()
        assert agent.history == []

    def test_history_is_copy(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        agent.add_user_message("Hello")
        h = agent.history
        h.append(UserMessage(content="Extra"))
        assert len(agent.history) == 1

    def test_multiple_user_messages(self, mock_llm):
        agent = ConversationAgent(name="conv", llm=mock_llm)
        agent.add_user_message("First")
        agent.add_user_message("Second")
        assert len(agent.history) == 2


class TestConversationChat:
    @pytest.mark.asyncio
    async def test_simple_chat_appends_to_history(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hello back")
        agent = ConversationAgent(name="conv", llm=mock_llm)
        result = await agent.chat("Hi")
        assert result == "Hello back"
        assert len(agent.history) == 2
        assert agent.history[0].content == "Hi"
        assert agent.history[1].content == "Hello back"

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Sure")
        agent = ConversationAgent(
            name="sys",
            llm=mock_llm,
            system_prompt="You are helpful.",
        )
        await agent.chat("Do something")
        call_kwargs = mock_llm.agenerate.call_args[1]
        prompt = call_kwargs["prompt"]
        assert prompt.system_prompt() == "You are helpful."

    @pytest.mark.asyncio
    async def test_chat_accumulates_history(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="First response"
        )
        agent = ConversationAgent(name="conv", llm=mock_llm)
        await agent.chat("First message")
        assert len(agent.history) == 2

        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="Second response"
        )
        await agent.chat("Second message")
        assert len(agent.history) == 4
        assert agent.history[2].content == "Second message"
        assert agent.history[3].content == "Second response"

    @pytest.mark.asyncio
    async def test_chat_includes_history_in_prompt(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="R1")
        agent = ConversationAgent(name="conv", llm=mock_llm)
        await agent.chat("First")

        mock_llm.agenerate.return_value = make_fake_llm_response(content="R2")
        await agent.chat("Second")

        call_kwargs = mock_llm.agenerate.call_args[1]
        prompt = call_kwargs["prompt"]
        messages = prompt.messages
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "R1"
        assert messages[2].content == "Second"

    @pytest.mark.asyncio
    async def test_chat_clear_history(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="R1")
        agent = ConversationAgent(name="conv", llm=mock_llm)
        await agent.chat("First")
        agent.clear_history()
        assert len(agent.history) == 0

    @pytest.mark.asyncio
    async def test_chat_with_tool_round(self, mock_llm, registry):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "echo", "args": {"text": "hello"}},
                    ],
                )
            return make_fake_llm_response(content="Echoed: hello")

        mock_llm.agenerate.side_effect = gen

        agent = ConversationAgent(
            name="tool-conv",
            llm=mock_llm,
            tool_registry=registry,
        )
        result = await agent.chat("Say hello")
        assert result == "Echoed: hello"
        assert len(agent.history) >= 3


class TestConversationTruncation:
    @pytest.mark.asyncio
    async def test_truncation_drops_oldest_turns(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Response")
        agent = ConversationAgent(
            name="trunc",
            llm=mock_llm,
            max_history_turns=1,
        )

        await agent.chat("Turn 1")
        assert len(agent.history) == 2

        await agent.chat("Turn 2")
        assert len(agent.history) == 2
        assert agent.history[0].content == "Turn 2"
        assert agent.history[1].content == "Response"

    @pytest.mark.asyncio
    async def test_truncation_preserves_system_prompt(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Response")
        agent = ConversationAgent(
            name="trunc-sys",
            llm=mock_llm,
            system_prompt="System msg",
            max_history_turns=1,
        )

        await agent.chat("Turn 1")
        await agent.chat("Turn 2")
        assert len(agent.history) == 2
        assert agent.history[0].content == "Turn 2"
        assert agent.history[1].content == "Response"

    @pytest.mark.asyncio
    async def test_no_truncation_when_not_set(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="R")
        agent = ConversationAgent(name="no-trunc", llm=mock_llm)

        for i in range(5):
            await agent.chat(f"Turn {i}")
        assert len(agent.history) == 10


@pytest.mark.asyncio
async def test_transcribe_search_summarize_integration():
    from agent_platform.agents.tools import SearchTool, TranscribeTool
    from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
    from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
    from agent_platform.core.interfaces.speech.base import BaseSpeechToText
    from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
    from agent_platform.core.schemas.chunk import TextChunk
    from agent_platform.core.schemas.conversation import Transcript, Utterance
    from agent_platform.core.schemas.embedding import Embedding
    from agent_platform.core.schemas.score import Score

    stt = AsyncMock(spec=BaseSpeechToText)
    stt.transcribe = AsyncMock(
        return_value=Transcript(
            utterances=[
                Utterance(text="What is the capital of France?", confidence=0.98)
            ],
        )
    )

    embedder = AsyncMock(spec=BaseEmbeddingProvider)
    embedder.aembed_document = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[Embedding.from_list([0.1, 0.2, 0.3])],
            model="test",
        )
    )

    store = AsyncMock(spec=BaseVectorStore)
    store.search_with_scores = AsyncMock(
        return_value=[
            (
                TextChunk(text="Paris is the capital of France.", index=0),
                Score.similarity(0.95),
            ),
            (
                TextChunk(text="France is in Western Europe.", index=1),
                Score.similarity(0.85),
            ),
        ]
    )

    transcribe_tool = TranscribeTool(provider=stt)
    search_tool = SearchTool(embedder=embedder, store=store)

    registry = ToolRegistry()
    registry.register(transcribe_tool)
    registry.register(search_tool)

    llm = MagicMock()
    llm.agenerate = AsyncMock()

    call_count = 0

    async def generate_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "transcribe", "args": {"data": b"audio_data"}},
                ],
            )
        if call_count == 2:
            return make_fake_llm_response(
                content="",
                tool_calls=[
                    {
                        "id": "t2",
                        "name": "search",
                        "args": {"query": "capital of France", "k": 3},
                    },
                ],
            )
        return make_fake_llm_response(
            content="Summary: The user asked about the capital of France. "
            "The search results show that Paris is the capital of France."
        )

    llm.agenerate.side_effect = generate_side_effect

    agent = ConversationAgent(
        name="audio-search-summarize",
        llm=llm,
        tool_registry=registry,
        system_prompt="You are a helpful assistant that transcribes audio, "
        "searches for relevant information, and summarizes.",
    )

    result = await agent.chat(
        "Please transcribe and find info about the capital of France"
    )

    assert "Summary" in result
    assert "Paris" in result
    assert llm.agenerate.await_count == 3
    assert stt.transcribe.await_count == 1
    assert embedder.aembed_document.await_count == 1
    assert store.search_with_scores.await_count == 1
