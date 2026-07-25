import asyncio

import pytest

from agent_platform.core.schemas.score import Score
from agent_platform.evals.dataset import EvalDataset
from agent_platform.evals.runner import EvalRunner
from agent_platform.evals.schemas import EvalCase, EvalOutput


class _StubScorer:
    name = "stub"

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    async def score(self, case, output):
        return Score.confidence(self._value)


def _dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(list(cases))


class TestEvalRunnerHappyPath:
    @pytest.mark.asyncio
    async def test_all_cases_pass(self):
        async def target(case):
            return EvalOutput(text="ok")

        runner = EvalRunner(target, [_StubScorer(1.0)])
        dataset = _dataset(EvalCase(id="a", input="x"), EvalCase(id="b", input="y"))

        report = await runner.arun(dataset)

        assert report.pass_rate == 1.0
        assert all(r.output is not None for r in report.results)
        assert all(r.output.latency_ms >= 0 for r in report.results)

    @pytest.mark.asyncio
    async def test_threshold_boundary(self):
        async def target(case):
            return EvalOutput(text="ok")

        case_pass = EvalCase(id="a", input="x", threshold=0.5)
        case_fail = EvalCase(id="b", input="y", threshold=0.9)
        runner = EvalRunner(target, [_StubScorer(0.7)])

        report = await runner.arun(_dataset(case_pass, case_fail))

        results = {r.case_id: r for r in report.results}
        assert results["a"].passed is True
        assert results["b"].passed is False


class TestEvalRunnerIsolation:
    @pytest.mark.asyncio
    async def test_one_failing_case_does_not_abort_run(self):
        async def target(case):
            if case.id == "bad":
                raise RuntimeError("boom")
            return EvalOutput(text="ok")

        runner = EvalRunner(target, [_StubScorer(1.0)])
        dataset = _dataset(
            EvalCase(id="good", input="x"), EvalCase(id="bad", input="y")
        )

        report = await runner.arun(dataset)

        results = {r.case_id: r for r in report.results}
        assert results["good"].passed is True
        assert results["bad"].passed is False
        assert results["bad"].output is None
        assert "boom" in results["bad"].error

    @pytest.mark.asyncio
    async def test_scorer_failure_is_isolated(self):
        class _FailingScorer:
            name = "failing"

            async def score(self, case, output):
                raise ValueError("scorer exploded")

        async def target(case):
            return EvalOutput(text="ok")

        runner = EvalRunner(target, [_FailingScorer()])
        dataset = _dataset(EvalCase(id="a", input="x"))

        report = await runner.arun(dataset)

        assert report.results[0].passed is False
        assert "scorer exploded" in report.results[0].error


class TestEvalRunnerCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        async def target(case):
            raise asyncio.CancelledError()

        runner = EvalRunner(target, [_StubScorer(1.0)])

        with pytest.raises(asyncio.CancelledError):
            await runner.arun(_dataset(EvalCase(id="a", input="x")))


class TestEvalRunnerConcurrency:
    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        active = 0
        max_active = 0

        async def target(case):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return EvalOutput(text="ok")

        cases = [EvalCase(id=str(i), input="x") for i in range(6)]
        runner = EvalRunner(target, [_StubScorer(1.0)], concurrency=2)

        await runner.arun(_dataset(*cases))

        assert max_active <= 2
