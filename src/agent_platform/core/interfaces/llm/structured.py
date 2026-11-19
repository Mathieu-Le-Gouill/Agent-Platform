from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from agent_platform.core.errors import RecoveryExhausted
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, ResponseFormat
from agent_platform.core.retry import retry_once_on_invalid
from agent_platform.core.schemas.message import AssistantMessage, Prompt, UserMessage
from agent_platform.core.schemas.token import TokenUsage

__all__ = [
    "StructuredOutputError",
    "StructuredResult",
    "generate_structured",
    "schema_config",
    "schema_correction",
    "schema_error",
]

T = TypeVar("T")

_EMPTY_CORRECTION = (
    "Your previous response was empty. Respond again with a non-empty answer."
)


class StructuredResult(BaseModel, Generic[T], frozen=True):
    """A typed model answer.

    `usage` sums every attempt, so a re-ask is paid for in the total.
    `response` is the accepted attempt only.
    """

    output: T
    usage: TokenUsage
    response: LLMResponse


class StructuredOutputError(RecoveryExhausted):
    """Raised when the re-asked answer is still invalid.

    `last_result` is the last (rejected) `LLMResponse`; `usage` sums every
    attempt.
    """

    def __init__(
        self, message: str = "", *, last_result: Any = None, usage: TokenUsage
    ) -> None:
        super().__init__(message, last_result=last_result)
        self.usage = usage


def schema_config(
    config: GenerationConfig | None, schema_type: type[BaseModel]
) -> GenerationConfig:
    """Returns `config` asking for JSON that matches `schema_type`'s JSON schema."""
    return (config or GenerationConfig()).model_copy(
        update={
            "response_format": ResponseFormat.JSON_SCHEMA,
            "json_schema": schema_type.model_json_schema(),
        }
    )


def schema_error(schema_type: type[BaseModel], text: str) -> str | None:
    """Returns why `text` does not match `schema_type`, or `None` when it does."""
    try:
        schema_type.model_validate_json(text)
    except PydanticValidationError as exc:
        return str(exc)
    return None


def schema_correction(error: str) -> UserMessage:
    """Returns the user message that asks the model to answer again after `error`."""
    return UserMessage(
        content=(
            "Your previous response did not match the required JSON "
            f"schema: {error}. Respond again with valid JSON only, "
            "matching the schema exactly."
        )
    )


def _response_text(response: LLMResponse) -> str:
    return response.message.text if response.message is not None else ""


async def generate_structured(
    llm: BaseLLMProvider[Any],
    prompt: Prompt,
    output_type: type[T],
    config: GenerationConfig | None = None,
) -> StructuredResult[T]:
    """Calls `llm` and returns its answer as `output_type`, re-asking once if invalid.

    `output_type` is a `BaseModel` (the model answers in JSON that matches its
    schema, and the answer is validated) or `str` (plain text that must not be
    empty). The re-ask appends the rejected answer and the validation error to
    the conversation. The model comes from `config.model`, else from the model
    the provider is bound to.

    Raises:
        StructuredOutputError: the re-asked answer is still invalid.
        TypeError: `output_type` is neither `str` nor a `BaseModel` subclass.
    """
    schema_type: type[BaseModel] | None = None
    call_config = config
    if issubclass(output_type, BaseModel):
        schema_type = output_type
        call_config = schema_config(config, schema_type)
    elif output_type is str:
        if config is not None and config.response_format is not ResponseFormat.TEXT:
            call_config = config.model_copy(
                update={"response_format": ResponseFormat.TEXT}
            )
    else:
        raise TypeError(
            f"output_type must be str or a BaseModel subclass, got {output_type!r}"
        )

    responses: list[LLMResponse] = []
    working = prompt

    async def attempt() -> LLMResponse:
        response = await llm.agenerate(working, call_config)
        responses.append(response)
        return response

    def check(response: LLMResponse) -> str | None:
        text = _response_text(response)
        if schema_type is None:
            return None if text.strip() else "the response was empty"
        return schema_error(schema_type, text)

    async def correct(error: str) -> None:
        nonlocal working
        rejected = responses[-1].message or AssistantMessage(content="")
        correction = (
            UserMessage(content=_EMPTY_CORRECTION)
            if schema_type is None
            else schema_correction(error)
        )
        working = Prompt(messages=[*prompt.messages, rejected, correction])

    try:
        response = await retry_once_on_invalid(
            attempt=attempt, check=check, correct=correct
        )
    except RecoveryExhausted as exc:
        raise StructuredOutputError(
            str(exc),
            last_result=exc.last_result,
            usage=_total_usage(responses),
        ) from exc

    text = _response_text(response)
    output: Any = text if schema_type is None else schema_type.model_validate_json(text)
    return StructuredResult(
        output=output, usage=_total_usage(responses), response=response
    )


def _total_usage(responses: list[LLMResponse]) -> TokenUsage:
    return sum((r.usage for r in responses), TokenUsage.zero())
