import pytest

from agent_platform.agents.errors import AgentRecoveryExhausted
from agent_platform.agents.validation import retry_once_on_invalid


class TestRetryOnceOnInvalid:
    @pytest.mark.asyncio
    async def test_valid_first_attempt_returns_without_retry(self):
        calls = 0

        async def attempt():
            nonlocal calls
            calls += 1
            return "ok"

        async def correct(error):
            raise AssertionError("correct should not be called")

        result = await retry_once_on_invalid(
            attempt=attempt, check=lambda r: None, correct=correct
        )

        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_invalid_then_valid_retries_once(self):
        calls = 0
        corrected = []

        async def attempt():
            nonlocal calls
            calls += 1
            return "bad" if calls == 1 else "good"

        async def correct(error):
            corrected.append(error)

        result = await retry_once_on_invalid(
            attempt=attempt,
            check=lambda r: None if r == "good" else "was bad",
            correct=correct,
        )

        assert result == "good"
        assert calls == 2
        assert corrected == ["was bad"]

    @pytest.mark.asyncio
    async def test_invalid_twice_raises_recovery_exhausted(self):
        calls = 0

        async def attempt():
            nonlocal calls
            calls += 1
            return "bad"

        async def correct(error):
            pass

        with pytest.raises(AgentRecoveryExhausted) as exc_info:
            await retry_once_on_invalid(
                attempt=attempt, check=lambda r: "still bad", correct=correct
            )

        assert calls == 2
        assert exc_info.value.last_result == "bad"
        assert "still bad" in str(exc_info.value)
