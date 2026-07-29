from __future__ import annotations

import re
from typing import Protocol

from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.enums import SimilarityMetric
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.score import Score
from agent_platform.core.similarity import compute_similarity, similarity_bounds
from agent_platform.evals.errors import EvalRunError
from agent_platform.evals.schemas import EvalCase, EvalOutput

_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Given an expected answer and an actual "
    "answer, respond with a single number between 0 and 1 rating how well "
    "the actual answer matches the expected one, then a short rationale. "
    "Put the number on its own on the first line."
)

_SCORE_PATTERN = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


class Scorer(Protocol):
    name: str

    async def score(self, case: EvalCase, output: EvalOutput) -> Score: ...


class ExactMatchScorer:
    name = "exact_match"

    async def score(self, case: EvalCase, output: EvalOutput) -> Score:
        expected = (case.expected or "").strip()
        value = 1.0 if output.text.strip() == expected else 0.0
        return Score.confidence(value)


class EmbeddingSimilarityScorer:
    name = "embedding_similarity"

    def __init__(
        self,
        backend: BaseEmbeddingProvider,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self._backend = backend
        self._metric = metric

    async def score(self, case: EvalCase, output: EvalOutput) -> Score:
        expected_response = await self._backend.aembed_query(case.expected or "")
        actual_response = await self._backend.aembed_query(output.text)
        value = compute_similarity(
            expected_response.embeddings[0].to_list(),
            actual_response.embeddings[0].to_list(),
            self._metric,
        )
        low, high = similarity_bounds(self._metric)
        return Score.similarity(value, low=low, high=high)


class LLMJudgeScorer:
    name = "llm_judge"

    def __init__(self, llm: BaseLLMProvider, model: str) -> None:
        self._llm = llm
        self._model = model

    async def score(self, case: EvalCase, output: EvalOutput) -> Score:
        prompt = Prompt.build(
            system=_JUDGE_SYSTEM_PROMPT,
            user=(
                f"Expected answer:\n{case.expected or ''}\n\n"
                f"Actual answer:\n{output.text}"
            ),
        )
        response = await self._llm.agenerate(
            prompt=prompt, config=GenerationConfig(model=self._model)
        )
        text = response.message.text if response.message else ""
        match = _SCORE_PATTERN.search(text)
        if match is None:
            raise EvalRunError(f"LLM judge returned an unparseable verdict: {text!r}")
        return Score.quality(float(match.group(1)))


class ToolSelectionScorer:
    name = "tool_selection"

    async def score(self, case: EvalCase, output: EvalOutput) -> Score:
        expected_tools = set((case.reference or {}).get("expected_tools", []))
        value = 1.0 if set(output.tool_calls) == expected_tools else 0.0
        return Score.confidence(value)


__all__ = [
    "Scorer",
    "ExactMatchScorer",
    "EmbeddingSimilarityScorer",
    "LLMJudgeScorer",
    "ToolSelectionScorer",
]
