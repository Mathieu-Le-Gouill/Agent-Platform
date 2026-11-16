import asyncio
import gc
import weakref
from uuid import uuid4

import pytest

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.api.conversations import ConversationNotFound, ConversationStore
from agent_platform.core.token_usage import TokenUsageAggregator
from tests.helpers import make_fake_llm_response


@pytest.fixture
def template(mock_llm):
    mock_llm.agenerate.return_value = make_fake_llm_response(content="reply")
    return ConversationAgent(
        name="t", llm=mock_llm, token_usage_aggregator=TokenUsageAggregator()
    )


class TestOwnership:
    def test_owner_gets_its_conversation_back(self, template):
        store = ConversationStore(template, max_per_key=5)
        created = store.create("a")

        assert store.get("a", created.id) is created

    def test_each_conversation_is_a_separate_fork(self, template):
        store = ConversationStore(template, max_per_key=5)

        first, second = store.create("a"), store.create("a")

        assert first.id != second.id
        assert first.agent is not second.agent
        assert first.agent.conversation_id == first.id

    def test_foreign_and_unknown_ids_raise_the_same_error(self, template):
        store = ConversationStore(template, max_per_key=5)
        created = store.create("a")

        with pytest.raises(ConversationNotFound) as foreign:
            store.get("b", created.id)
        with pytest.raises(ConversationNotFound) as unknown:
            store.get("a", uuid4())

        assert str(foreign.value) == str(unknown.value)

    def test_rejects_a_cap_below_one(self, template):
        with pytest.raises(ValueError, match="max_per_key"):
            ConversationStore(template, max_per_key=0)


class TestEviction:
    def test_creating_past_the_cap_evicts_the_least_recently_used(self, template):
        store = ConversationStore(template, max_per_key=2)
        first, second = store.create("a"), store.create("a")
        store.get("a", first.id)

        third = store.create("a")

        assert store.get("a", first.id) is first
        assert store.get("a", third.id) is third
        with pytest.raises(ConversationNotFound):
            store.get("a", second.id)

    def test_eviction_never_touches_another_key(self, template):
        store = ConversationStore(template, max_per_key=1)
        other = store.create("b")

        for _ in range(3):
            store.create("a")

        assert store.get("b", other.id) is other

    def test_evicted_conversation_is_freed(self, template):
        store = ConversationStore(template, max_per_key=1)
        evicted = store.create("a")
        agent_ref = weakref.ref(evicted.agent)
        lock_ref = weakref.ref(evicted.lock)
        del evicted

        store.create("a")
        gc.collect()

        assert agent_ref() is None
        assert lock_ref() is None


class TestTurns:
    async def test_turns_on_one_conversation_run_one_after_the_other(
        self, template, mock_llm
    ):
        events: list[str] = []

        async def slow_generate(**kwargs):
            events.append("start")
            await asyncio.sleep(0.01)
            events.append("end")
            return make_fake_llm_response(content="reply")

        mock_llm.agenerate.side_effect = slow_generate
        conversation = ConversationStore(template, max_per_key=2).create("a")

        async def turn(text: str) -> None:
            async with conversation.lock:
                await conversation.agent.chat(text)

        await asyncio.gather(turn("one"), turn("two"))

        assert events == ["start", "end", "start", "end"]
        assert [m.content for m in conversation.agent.history] == [
            "one",
            "reply",
            "two",
            "reply",
        ]

    async def test_turn_completes_when_its_conversation_is_evicted_midway(
        self, template, mock_llm
    ):
        store = ConversationStore(template, max_per_key=1)
        conversation = store.create("a")
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_generate(**kwargs):
            started.set()
            await release.wait()
            return make_fake_llm_response(content="reply")

        mock_llm.agenerate.side_effect = blocked_generate

        async def turn() -> str:
            async with conversation.lock:
                return await conversation.agent.chat("hi")

        task = asyncio.create_task(turn())
        await started.wait()
        store.create("a")
        with pytest.raises(ConversationNotFound):
            store.get("a", conversation.id)
        release.set()

        assert await task == "reply"
