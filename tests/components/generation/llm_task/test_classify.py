from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.components.generation.llm_task.classify import (
    ClassificationExample,
    classify_task,
)
from agent_platform.components.generation.llm_task.component import LLMTaskInput
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.interfaces.llm.structured import StructuredOutputError
from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.core.schemas.token import TokenUsage

LABELS = ["billing", "support", "sales"]


def _response(text: str, tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(content=text),
        usage=TokenUsage(input_tokens=tokens, output_tokens=1),
        model="m",
    )


def _llm(*responses: LLMResponse) -> MagicMock:
    llm = MagicMock()
    llm.agenerate = AsyncMock(side_effect=list(responses))
    return llm


class TestSingleLabel:
    async def test_a_valid_label_is_accepted(self):
        llm = _llm(_response('{"label": "support"}'))

        result = await classify_task(llm, LABELS).arun(LLMTaskInput("help", None))

        assert result.output.label == "support"
        assert llm.agenerate.await_count == 1

    async def test_the_schema_sent_restricts_labels_to_an_enum(self):
        llm = _llm(_response('{"label": "sales"}'))

        await classify_task(llm, LABELS).arun(LLMTaskInput("buy", None))

        schema = llm.agenerate.await_args.args[1].json_schema
        assert schema["properties"]["label"]["enum"] == LABELS
        assert "score" not in schema["properties"]

    async def test_a_label_outside_the_enum_is_re_asked_once_then_raises(self):
        llm = _llm(
            _response('{"label": "other"}', tokens=10),
            _response('{"label": "unknown"}', tokens=6),
        )

        with pytest.raises(StructuredOutputError) as exc_info:
            await classify_task(llm, LABELS).arun(LLMTaskInput("x", None))

        assert llm.agenerate.await_count == 2
        assert exc_info.value.usage == TokenUsage(input_tokens=16, output_tokens=2)

    async def test_a_re_ask_can_recover_and_usage_is_summed(self):
        accepted = _response('{"label": "billing"}', tokens=6)
        llm = _llm(_response('{"label": "other"}', tokens=10), accepted)

        result = await classify_task(llm, LABELS).arun(LLMTaskInput("x", None))

        assert result.output.label == "billing"
        assert result.response is accepted
        assert result.usage == TokenUsage(input_tokens=16, output_tokens=2)


class TestMultiLabel:
    async def test_returns_a_list_of_labels_without_duplicates(self):
        llm = _llm(_response('{"labels": ["sales", "billing", "sales"]}'))

        result = await classify_task(llm, LABELS, multi_label=True).arun(
            LLMTaskInput("x", None)
        )

        assert result.output.labels == ["sales", "billing"]

    async def test_an_empty_list_is_valid(self):
        llm = _llm(_response('{"labels": []}'))

        result = await classify_task(llm, LABELS, multi_label=True).arun(
            LLMTaskInput("x", None)
        )

        assert result.output.labels == []

    async def test_the_schema_restricts_items_without_unique_items(self):
        llm = _llm(_response('{"labels": []}'))

        await classify_task(llm, LABELS, multi_label=True).arun(LLMTaskInput("x", None))

        items = llm.agenerate.await_args.args[1].json_schema["properties"]["labels"]
        assert items["items"]["enum"] == LABELS
        assert "uniqueItems" not in items

    async def test_a_label_outside_the_enum_raises_after_one_re_ask(self):
        llm = _llm(_response('{"labels": ["other"]}'), _response('{"labels": ["x"]}'))

        with pytest.raises(StructuredOutputError):
            await classify_task(llm, LABELS, multi_label=True).arun(
                LLMTaskInput("x", None)
            )


class TestInstructions:
    async def test_examples_are_rendered_into_the_instructions(self):
        llm = _llm(_response('{"label": "billing"}'))
        examples = [ClassificationExample(text="my invoice", label="billing")]

        await classify_task(llm, LABELS, examples=examples).arun(
            LLMTaskInput("x", None)
        )

        system = llm.agenerate.await_args.args[0].system_prompt()
        assert "Text: my invoice\nLabel: billing" in system
        assert "billing, support, sales" in system

    async def test_custom_instructions_replace_the_default_wording(self):
        llm = _llm(_response('{"label": "billing"}'))

        await classify_task(llm, LABELS, instructions="Pick one.").arun(
            LLMTaskInput("x", None)
        )

        assert llm.agenerate.await_args.args[0].system_prompt() == "Pick one."

    def test_empty_labels_raise(self):
        with pytest.raises(ValueError, match="labels"):
            classify_task(_llm(), [])
