import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.retry import with_retry


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_after_failures(self):
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.001, max_delay=0.001)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        @with_retry(max_attempts=2, base_delay=0.001, max_delay=0.001)
        async def always_fails():
            calls["n"] += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await always_fails()
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self):
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.001, retry_on=(ProviderError,))
        async def raises_other():
            calls["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await raises_other()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_no_retry_needed(self):
        @with_retry()
        async def succeeds():
            return "done"

        assert await succeeds() == "done"
