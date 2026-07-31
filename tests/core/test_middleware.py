import pytest

from agent_platform.core.middleware import MiddlewarePipeline


class RecordingMiddleware:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def before(self, ctx):
        self._log.append(f"before:{ctx}")
        return None

    async def after(self, ctx, result):
        self._log.append(f"after:{ctx}:{result}")
        return result


class ShortCircuitMiddleware:
    def __init__(self, value):
        self._value = value

    async def before(self, ctx):
        return self._value

    async def after(self, ctx, result):
        return result


class ResultRewritingMiddleware:
    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        return f"wrapped:{result}"


class TestMiddlewarePipeline:
    @pytest.mark.asyncio
    async def test_runs_operation_when_no_short_circuit(self):
        log: list[str] = []
        pipeline = MiddlewarePipeline([RecordingMiddleware(log)])

        async def operation(ctx):
            return f"result:{ctx}"

        result = await pipeline.run("ctx", operation)

        assert result == "result:ctx"
        assert log == ["before:ctx", "after:ctx:result:ctx"]

    @pytest.mark.asyncio
    async def test_before_short_circuit_skips_operation(self):
        called = {"n": 0}

        async def operation(ctx):
            called["n"] += 1
            return "should-not-run"

        pipeline = MiddlewarePipeline([ShortCircuitMiddleware("short-circuited")])

        result = await pipeline.run("ctx", operation)

        assert result == "short-circuited"
        assert called["n"] == 0

    @pytest.mark.asyncio
    async def test_after_hooks_run_in_reverse_order(self):
        log: list[str] = []

        class OrderedAfter:
            def __init__(self, name: str) -> None:
                self._name = name

            async def before(self, ctx):
                return None

            async def after(self, ctx, result):
                log.append(self._name)
                return result

        pipeline = MiddlewarePipeline([OrderedAfter("first"), OrderedAfter("second")])

        async def operation(ctx):
            return "ok"

        await pipeline.run("ctx", operation)

        assert log == ["second", "first"]

    @pytest.mark.asyncio
    async def test_after_hooks_can_rewrite_result(self):
        pipeline = MiddlewarePipeline([ResultRewritingMiddleware()])

        async def operation(ctx):
            return "raw"

        result = await pipeline.run("ctx", operation)

        assert result == "wrapped:raw"

    @pytest.mark.asyncio
    async def test_empty_pipeline_runs_operation_unmodified(self):
        pipeline = MiddlewarePipeline([])

        async def operation(ctx):
            return "unmodified"

        result = await pipeline.run("ctx", operation)

        assert result == "unmodified"

    @pytest.mark.asyncio
    async def test_first_short_circuit_wins_and_later_before_not_called(self):
        called = {"n": 0}

        class NeverCalled:
            async def before(self, ctx):
                called["n"] += 1
                return None

            async def after(self, ctx, result):
                return result

        pipeline = MiddlewarePipeline([ShortCircuitMiddleware("first"), NeverCalled()])

        async def operation(ctx):
            return "should-not-run"

        result = await pipeline.run("ctx", operation)

        assert result == "first"
        assert called["n"] == 0
