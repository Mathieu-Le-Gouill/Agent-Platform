from __future__ import annotations

import logging
from typing import Generic, TypeVar

from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResult,
    ClassificationResponse,
)
from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.components.base import Component
from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.components.llm_classifier.strategies.registry import get_strategy
from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.message import Prompt

logger = logging.getLogger(__name__)

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)
GenConfigT = TypeVar("GenConfigT", bound=GenerationConfig)

_ClassifierInput = tuple[list[TextDocument], list[str], LLMClassifierConfig]


class LLMClassifier(
    Component[_ClassifierInput, ClassificationResponse],
    Generic[CredentialsT, GenConfigT],
):
    def __init__(
        self,
        llm: BaseLLMProvider[CredentialsT, GenConfigT],
        config: LLMClassifierConfig | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or LLMClassifierConfig()

    async def arun(self, input: _ClassifierInput) -> ClassificationResponse:
        items, candidate_labels, config = input
        config = config or self._config
        strategy = get_strategy(config.classification_mode)

        system_prompt = strategy.build_system_prompt(
            config,
            candidate_labels,
            multi_label=config.multi_label,
        )

        results: list[ClassificationResult] = []
        for item in items:
            prompt = Prompt.build(system=system_prompt, user=item.text)
            response = await self._llm.agenerate(prompt, config)
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
        except ValidationError:
            logger.warning(
                "Failed to parse classification prediction: label=%s, score=%s",
                arguments.get("label"),
                arguments.get("score"),
            )
            return ClassificationResult(predictions=[])

        return ClassificationResult(predictions=[prediction])

    return ClassificationResult(predictions=[])
