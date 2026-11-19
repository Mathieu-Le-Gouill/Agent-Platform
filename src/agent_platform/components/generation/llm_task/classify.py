from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, overload

from pydantic import BaseModel, create_model, field_validator

from agent_platform.components.generation.llm_task.component import LLMTask
from agent_platform.core.interfaces.llm.base import BaseLLMProvider

__all__ = [
    "ClassificationExample",
    "MultiLabelOutput",
    "SingleLabelOutput",
    "classify_task",
]


class ClassificationExample(BaseModel):
    text: str
    label: str


class SingleLabelOutput(BaseModel):
    label: str


class MultiLabelOutput(BaseModel):
    labels: list[str]

    @field_validator("labels")
    @classmethod
    def _drop_duplicates(cls, labels: list[str]) -> list[str]:
        # `uniqueItems` is left out of the schema (some strict modes reject it),
        # so repeats are removed here, keeping the first occurrence of each.
        return list(dict.fromkeys(labels))


_SINGLE_RULES = (
    "Classify the text into exactly ONE of these categories:\n{labels}\n\n"
    "Rules:\n"
    "- Select exactly one category from the list.\n"
    "- Base your decision only on the content of the text.\n"
    "- Do not use outside knowledge."
)

_MULTI_RULES = (
    "Classify the text into zero, one or several of these categories:\n{labels}\n\n"
    "Rules:\n"
    "- Select every category that is clearly supported by the text.\n"
    "- Categories are independent and several may apply.\n"
    "- Do not select a category on a weak assumption.\n"
    "- Base your decision only on the content of the text.\n"
    "- If no category applies, return an empty list."
)


@overload
def classify_task(
    llm: BaseLLMProvider,
    labels: list[str],
    *,
    multi_label: Literal[False] = ...,
    examples: Sequence[ClassificationExample] = ...,
    instructions: str | None = ...,
) -> LLMTask[SingleLabelOutput]: ...


@overload
def classify_task(
    llm: BaseLLMProvider,
    labels: list[str],
    *,
    multi_label: Literal[True],
    examples: Sequence[ClassificationExample] = ...,
    instructions: str | None = ...,
) -> LLMTask[MultiLabelOutput]: ...


def classify_task(
    llm: BaseLLMProvider,
    labels: list[str],
    *,
    multi_label: bool = False,
    examples: Sequence[ClassificationExample] = (),
    instructions: str | None = None,
) -> LLMTask[Any]:
    """Returns a task that classifies a text into `labels`.

    The output schema restricts each label to `labels` (a JSON-schema `enum`),
    so an answer outside the list is rejected and re-asked once. `instructions`
    replaces the default wording; `examples` are appended to it.

    Raises:
        ValueError: `labels` is empty.
    """
    if not labels:
        raise ValueError("labels must not be empty")

    label_enum = Literal.__getitem__(tuple(labels))
    output_type: type[BaseModel]
    if multi_label:
        output_type = create_model(
            "Classification",
            __base__=MultiLabelOutput,
            labels=(list[label_enum], ...),  # type: ignore[valid-type]
        )
        rules = _MULTI_RULES
    else:
        output_type = create_model(
            "Classification",
            __base__=SingleLabelOutput,
            label=(label_enum, ...),
        )
        rules = _SINGLE_RULES

    text = instructions or rules.format(labels=", ".join(labels))
    if examples:
        rendered = "\n\n".join(f"Text: {e.text}\nLabel: {e.label}" for e in examples)
        text = f"{text}\n\nExamples:\n{rendered}"
    return LLMTask(llm, text, output_type)
