from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.evals.errors import EvalRunError
from agent_platform.evals.schemas import EvalCase, EvalOutput
from agent_platform.evals.scorer import (
    EmbeddingSimilarityScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    ToolSelectionScorer,
)
from tests.helpers import make_fake_llm_response


def _case(**kwargs) -> EvalCase:
    return EvalCase(id="c1", input="hi", **kwargs)


class TestExactMatchScorer:
    @pytest.mark.asyncio
    async def test_match(self):
        scorer = ExactMatchScorer()
        score = await scorer.score(_case(expected="hello"), EvalOutput(text="hello"))
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_mismatch(self):
        scorer = ExactMatchScorer()
        score = await scorer.score(_case(expected="hello"), EvalOutput(text="world"))
        assert score.value == 0.0


class TestEmbeddingSimilarityScorer:
    @pytest.mark.asyncio
    async def test_scores_similarity_within_bounds(self):
        backend = MagicMock()
        backend.aembed_query = AsyncMock(
            side_effect=[
                EmbeddingResponse(
                    embeddings=[Embedding.from_list([1.0, 0.0])], model="m"
                ),
                EmbeddingResponse(
                    embeddings=[Embedding.from_list([-1.0, 0.0])], model="m"
                ),
            ]
        )
        scorer = EmbeddingSimilarityScorer(backend)

        score = await scorer.score(_case(expected="a"), EvalOutput(text="b"))

        assert score.value == pytest.approx(-1.0)
        assert score.low == -1.0
        assert score.high == 1.0

    @pytest.mark.asyncio
    async def test_does_not_raise_on_negative_cosine(self):
        backend = MagicMock()
        backend.aembed_query = AsyncMock(
            side_effect=[
                EmbeddingResponse(
                    embeddings=[Embedding.from_list([1.0, 0.0])], model="m"
                ),
                EmbeddingResponse(
                    embeddings=[Embedding.from_list([-1.0, -0.5])], model="m"
                ),
            ]
        )
        scorer = EmbeddingSimilarityScorer(backend)

        score = await scorer.score(_case(expected="a"), EvalOutput(text="b"))

        assert -1.0 <= score.value <= 1.0


class TestLLMJudgeScorer:
    @pytest.mark.asyncio
    async def test_parses_verdict(self):
        llm = MagicMock()
        llm.agenerate = AsyncMock(
            return_value=make_fake_llm_response(content="0.8\nClose enough")
        )
        scorer = LLMJudgeScorer(llm, model="test-model")

        score = await scorer.score(_case(expected="a"), EvalOutput(text="b"))

        assert score.value == 0.8

    @pytest.mark.asyncio
    async def test_unparseable_verdict_raises(self):
        llm = MagicMock()
        llm.agenerate = AsyncMock(
            return_value=make_fake_llm_response(content="not a number")
        )
        scorer = LLMJudgeScorer(llm, model="test-model")

        with pytest.raises(EvalRunError):
            await scorer.score(_case(expected="a"), EvalOutput(text="b"))


class TestToolSelectionScorer:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        scorer = ToolSelectionScorer()
        case = _case(reference={"expected_tools": ["search"]})
        score = await scorer.score(case, EvalOutput(text="", tool_calls=["search"]))
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_mismatch(self):
        scorer = ToolSelectionScorer()
        case = _case(reference={"expected_tools": ["search"]})
        score = await scorer.score(case, EvalOutput(text="", tool_calls=[]))
        assert score.value == 0.0
