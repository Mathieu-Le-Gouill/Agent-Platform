import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.retry import retry_async, retry_sync, with_retry


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

    @pytest.mark.asyncio
    async def test_non_retryable_platform_error_raises_immediately(self):
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.001)
        async def fails():
            calls["n"] += 1
            raise ProviderError("permanent", retryable=False)

        with pytest.raises(ProviderError, match="permanent"):
            await fails()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_retryable_platform_error_retries(self):
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.001, max_delay=0.001)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ProviderError("transient", retryable=True)
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 3


class _Flaky:
    """Fails `failures` times with `error`, then returns "ok"."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.calls = 0
        self._failures = failures
        self._error = error

    def sync(self) -> str:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return "ok"

    async def asynchronous(self) -> str:
        return self.sync()


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_honors_max_attempts_per_call(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ValueError("transient"))

        with pytest.raises(ValueError, match="transient"):
            await retry_async(flaky.asynchronous, max_attempts=4)

        assert flaky.calls == 4

    @pytest.mark.asyncio
    async def test_one_attempt_calls_once(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ValueError("transient"))

        with pytest.raises(ValueError):
            await retry_async(flaky.asynchronous, max_attempts=1)

        assert flaky.calls == 1

    @pytest.mark.asyncio
    async def test_returns_after_a_transient_failure(self, no_retry_sleep):
        flaky = _Flaky(failures=2, error=ValueError("transient"))

        assert await retry_async(flaky.asynchronous, max_attempts=3) == "ok"
        assert flaky.calls == 3

    @pytest.mark.asyncio
    async def test_non_retryable_platform_error_raises_after_one_call(
        self, no_retry_sleep
    ):
        flaky = _Flaky(failures=10, error=ProviderError("no", retryable=False))

        with pytest.raises(ProviderError, match="no"):
            await retry_async(flaky.asynchronous, max_attempts=5)

        assert flaky.calls == 1

    @pytest.mark.asyncio
    async def test_only_retries_the_given_exceptions(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ValueError("other"))

        with pytest.raises(ValueError):
            await retry_async(
                flaky.asynchronous, max_attempts=5, retry_on=(ProviderError,)
            )

        assert flaky.calls == 1

    @pytest.mark.asyncio
    async def test_rejects_fewer_than_one_attempt(self):
        with pytest.raises(ValueError, match="max_attempts"):
            await retry_async(_Flaky(0, ValueError()).asynchronous, max_attempts=0)


class TestRetrySync:
    def test_honors_max_attempts_per_call(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ValueError("transient"))

        with pytest.raises(ValueError, match="transient"):
            retry_sync(flaky.sync, max_attempts=4)

        assert flaky.calls == 4

    def test_one_attempt_calls_once(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ValueError("transient"))

        with pytest.raises(ValueError):
            retry_sync(flaky.sync, max_attempts=1)

        assert flaky.calls == 1

    def test_returns_after_a_transient_failure(self, no_retry_sleep):
        flaky = _Flaky(failures=2, error=ValueError("transient"))

        assert retry_sync(flaky.sync, max_attempts=3) == "ok"
        assert flaky.calls == 3

    def test_non_retryable_platform_error_raises_after_one_call(self, no_retry_sleep):
        flaky = _Flaky(failures=10, error=ProviderError("no", retryable=False))

        with pytest.raises(ProviderError, match="no"):
            retry_sync(flaky.sync, max_attempts=5)

        assert flaky.calls == 1

    def test_sleeps_between_attempts_with_time_sleep(self, mocker):
        sleep = mocker.patch("agent_platform.core.retry.time.sleep")
        flaky = _Flaky(failures=2, error=ValueError("transient"))

        retry_sync(flaky.sync, max_attempts=3, base_delay=1.0, max_delay=1.0)

        assert sleep.call_count == 2

    def test_rejects_fewer_than_one_attempt(self):
        with pytest.raises(ValueError, match="max_attempts"):
            retry_sync(_Flaky(0, ValueError()).sync, max_attempts=0)
