import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.resilience import CircuitBreaker, CircuitState, RateLimiter


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_starts_closed_and_allows_calls(self):
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)

        async def succeeds():
            return "ok"

        assert await breaker.acall(succeeds) == "ok"
        assert breaker.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0)

        async def fails():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.acall(fails)

        assert breaker.state is CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_without_calling(self):
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=30.0)
        called = {"n": 0}

        async def fails():
            called["n"] += 1
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await breaker.acall(fails)

        with pytest.raises(ProviderError, match="open"):
            await breaker.acall(fails)

        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_reset_timeout(self, mocker):
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)

        clock = {"t": 0.0}
        mocker.patch(
            "agent_platform.core.resilience.time.monotonic",
            side_effect=lambda: clock["t"],
        )

        async def fails():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await breaker.acall(fails)
        assert breaker.state is CircuitState.OPEN

        clock["t"] += 11.0
        assert breaker.state is CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self, mocker):
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)

        clock = {"t": 0.0}
        mocker.patch(
            "agent_platform.core.resilience.time.monotonic",
            side_effect=lambda: clock["t"],
        )

        async def fails():
            raise ValueError("boom")

        async def succeeds():
            return "recovered"

        with pytest.raises(ValueError):
            await breaker.acall(fails)
        clock["t"] += 11.0

        assert await breaker.acall(succeeds) == "recovered"
        assert breaker.state is CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens_circuit(self, mocker):
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10.0)

        clock = {"t": 0.0}
        mocker.patch(
            "agent_platform.core.resilience.time.monotonic",
            side_effect=lambda: clock["t"],
        )

        async def fails():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await breaker.acall(fails)
        clock["t"] += 11.0
        assert breaker.state is CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            await breaker.acall(fails)
        assert breaker.state is CircuitState.OPEN


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_burst_does_not_wait(self, mocker):
        sleep = mocker.patch("agent_platform.core.resilience.asyncio.sleep")
        limiter = RateLimiter(rate=5.0, burst=5.0)

        await limiter.acquire()
        await limiter.acquire()

        sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_beyond_burst_waits_for_replenishment(self, mocker):
        clock = {"t": 0.0}
        mocker.patch(
            "agent_platform.core.resilience.time.monotonic",
            side_effect=lambda: clock["t"],
        )

        async def fake_sleep(seconds: float) -> None:
            clock["t"] += seconds

        mocker.patch(
            "agent_platform.core.resilience.asyncio.sleep", side_effect=fake_sleep
        )

        limiter = RateLimiter(rate=1.0, burst=1.0)

        await limiter.acquire()
        await limiter.acquire()

        assert clock["t"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_default_burst_equals_rate(self, mocker):
        sleep = mocker.patch("agent_platform.core.resilience.asyncio.sleep")
        limiter = RateLimiter(rate=3.0)

        await limiter.acquire(tokens=3.0)

        sleep.assert_not_called()
