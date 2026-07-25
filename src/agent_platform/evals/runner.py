from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID, uuid4

from agent_platform.core.tracing import traced_operation_span
from agent_platform.evals.dataset import EvalDataset
from agent_platform.evals.schemas import EvalCase, EvalReport, EvalResult
from agent_platform.evals.scorer import Scorer
from agent_platform.evals.targets import EvalTarget

logger = logging.getLogger(__name__)

_EVAL_RUN_ID_ATTR = "agent_platform.eval.run_id"
_EVAL_CASE_ID_ATTR = "agent_platform.eval.case_id"


class EvalRunner:
    def __init__(
        self,
        target: EvalTarget,
        scorers: list[Scorer],
        concurrency: int = 5,
    ) -> None:
        self._target = target
        self._scorers = scorers
        self._semaphore = asyncio.Semaphore(concurrency)

    async def arun(self, dataset: EvalDataset) -> EvalReport:
        run_id = uuid4()

        async def _bounded(case: EvalCase) -> EvalResult:
            async with self._semaphore:
                return await self._run_case(run_id, case)

        results = await asyncio.gather(*(_bounded(case) for case in dataset))
        return EvalReport(run_id=run_id, results=list(results))

    async def _run_case(self, run_id: UUID, case: EvalCase) -> EvalResult:
        with traced_operation_span(
            "eval_case",
            **{_EVAL_RUN_ID_ATTR: str(run_id), _EVAL_CASE_ID_ATTR: case.id},
        ):
            try:
                start = time.monotonic()
                output = await self._target(case)
                latency_ms = (time.monotonic() - start) * 1000
                output = output.model_copy(update={"latency_ms": latency_ms})

                scored = await asyncio.gather(
                    *(scorer.score(case, output) for scorer in self._scorers)
                )
                scores = {
                    scorer.name: score for scorer, score in zip(self._scorers, scored)
                }
                passed = all(
                    score.normalized >= case.threshold for score in scores.values()
                )
                return EvalResult(
                    case_id=case.id,
                    output=output,
                    scores=scores,
                    passed=passed,
                    tags=case.tags,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Eval case '%s' failed", case.id)
                return EvalResult(
                    case_id=case.id,
                    output=None,
                    scores={},
                    passed=False,
                    error=str(exc),
                    tags=case.tags,
                )


__all__ = ["EvalRunner"]
