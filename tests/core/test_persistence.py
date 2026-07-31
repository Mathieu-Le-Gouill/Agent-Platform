import pytest

from agent_platform.core.persistence import InMemoryCheckpointer


class TestInMemoryCheckpointer:
    @pytest.mark.asyncio
    async def test_load_missing_key_returns_none(self):
        checkpointer: InMemoryCheckpointer[dict] = InMemoryCheckpointer()

        assert await checkpointer.load("missing") is None

    @pytest.mark.asyncio
    async def test_save_then_load_returns_state(self):
        checkpointer: InMemoryCheckpointer[dict] = InMemoryCheckpointer()

        await checkpointer.save("session-1", {"turn": 1})

        assert await checkpointer.load("session-1") == {"turn": 1}

    @pytest.mark.asyncio
    async def test_save_overwrites_existing_state(self):
        checkpointer: InMemoryCheckpointer[dict] = InMemoryCheckpointer()

        await checkpointer.save("session-1", {"turn": 1})
        await checkpointer.save("session-1", {"turn": 2})

        assert await checkpointer.load("session-1") == {"turn": 2}

    @pytest.mark.asyncio
    async def test_keys_are_independent(self):
        checkpointer: InMemoryCheckpointer[str] = InMemoryCheckpointer()

        await checkpointer.save("a", "state-a")
        await checkpointer.save("b", "state-b")

        assert await checkpointer.load("a") == "state-a"
        assert await checkpointer.load("b") == "state-b"
