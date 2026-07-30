from __future__ import annotations

import logging
from typing import Generic, NamedTuple, TypeVar, cast

from pydantic import ValidationError as PydanticValidationError

from agent_platform.components.base import Component
from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.components.llm_classifier.strategies.registry import get_strategy
from agent_platform.core.errors import ValidationError
from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.message import Prompt

logger = logging.getLogger(__name__)

GenConfigT = TypeVar("GenConfigT", bound=GenerationConfig)


class LLMClassifierInput(NamedTuple):
    items: list[TextDocument]
    candidate_labels: list[str]
    config: LLMClassifierConfig | None


class LLMClassifier(
    Component[LLMClassifierInput, ClassificationResponse],
    Generic[GenConfigT],
):
    def __init__(self, llm: BaseLLMProvider[GenConfigT]) -> None:
        self._llm = llm

    async def arun(self, input: LLMClassifierInput) -> ClassificationResponse:
        items, candidate_labels, config = input
        config = config or LLMClassifierConfig()
        strategy = get_strategy(config.classification_mode)

        system_prompt = strategy.build_system_prompt(
            config,
            candidate_labels,
            multi_label=config.multi_label,
        )

        results: list[ClassificationResult] = []
        for item in items:
            prompt = Prompt.build(system=system_prompt, user=item.text)
            # `config` is a provider-specific *LLMClassifierConfig subclass
            # (e.g. OpenAILLMClassifierConfig) that also mixes in the
            # matching GenerationConfig — see components/llm_classifier/config.py.
            response = await self._llm.agenerate(prompt, cast(GenConfigT, config))
            pred = _extract_pred(response, config.multi_label)
            results.append(pred)

        return ClassificationResponse(results=results)


def _extract_pred(
    response: LLMResponse,
    multi_label: bool,
) -> ClassificationResult:
    if response.message is None:
        return ClassificationResult(predictions=[])

    for tool_call in response.message.tool_calls:
        if tool_call.name != "record_classification":
            continue

        arguments = tool_call.arguments

        if multi_label:
            labels = arguments.get("labels", [])
            scores = arguments.get("scores", [])
            predictions = [
                ClassificationPrediction(label=label, score=score)
                for label, score in zip(labels, scores)
            ]
            return ClassificationResult(predictions=predictions)

        try:
            prediction = ClassificationPrediction(
                label=arguments["label"],
                score=arguments["score"],
            )
        except (ValidationError, PydanticValidationError):
            logger.warning(
                "Failed to parse classification prediction: label=%s, score=%s",
                arguments.get("label"),
                arguments.get("score"),
            )
            return ClassificationResult(predictions=[])

        return ClassificationResult(predictions=[prediction])

    return ClassificationResult(predictions=[])
