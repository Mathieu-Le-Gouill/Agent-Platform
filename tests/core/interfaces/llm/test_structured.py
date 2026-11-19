from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.core.errors import RecoveryExhausted
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, ResponseFormat
from agent_platform.core.interfaces.llm.structured import (
    StructuredOutputError,
    generate_structured,
)
from agent_platform.core.schemas.message import AssistantMessage, Prompt
from agent_platform.core.schemas.token import TokenUsage


class _Answer(BaseModel):
    value: int


def _response(text: str | None, tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        message=None if text is None else AssistantMessage(content=text),
        usage=TokenUsage(input_tokens=tokens, output_tokens=tokens),
        model="m",
    )


def _llm(*responses: LLMResponse) -> MagicMock:
    llm = MagicMock()
    llm.agenerate = AsyncMock(side_effect=list(responses))
    return llm


@pytest.fixture
def prompt() -> Prompt:
    return Prompt.build(user="Question")


class TestModelOutput:
    @pytest.mark.asyncio
    async def test_a_valid_answer_is_parsed_and_sent_with_the_schema(self, prompt):
        llm = _llm(_response('{"value": 3}'))

        result = await generate_structured(llm, prompt, _Answer)

        assert result.output == _Answer(value=3)
        assert llm.agenerate.await_count == 1
        sent = llm.agenerate.await_args.args[1]
        assert sent.response_format is ResponseFormat.JSON_SCHEMA
        assert sent.json_schema == _Answer.model_json_schema()

    @pytest.mark.asyncio
    async def test_the_caller_config_is_kept_and_not_mutated(self, prompt):
        llm = _llm(_response('{"value": 3}'))
        config = GenerationConfig(model="x", temperature=0.2)

        await generate_structured(llm, prompt, _Answer, config)

        sent = llm.agenerate.await_args.args[1]
        assert (sent.model, sent.temperature) == ("x", 0.2)
        assert config.response_format is ResponseFormat.TEXT

    @pytest.mark.asyncio
    async def test_an_invalid_answer_is_re_asked_once_with_the_error(self, prompt):
        llm = _llm(_response("nope", tokens=10), _response('{"value": 1}', tokens=7))

        result = await generate_structured(llm, prompt, _Answer)

        assert result.output == _Answer(value=1)
        second_prompt = llm.agenerate.await_args_list[1].args[0]
        assert len(second_prompt.messages) == len(prompt.messages) + 2
        assert second_prompt.messages[-2].text == "nope"
        assert "did not match the required JSON schema" in (
            second_prompt.messages[-1].text
        )

    @pytest.mark.asyncio
    async def test_usage_sums_attempts_and_response_is_the_accepted_one(self, prompt):
        accepted = _response('{"value": 1}', tokens=7)
        llm = _llm(_response("nope", tokens=10), accepted)

        result = await generate_structured(llm, prompt, _Answer)

        assert result.usage == TokenUsage(input_tokens=17, output_tokens=17)
        assert result.response is accepted

    @pytest.mark.asyncio
    async def test_a_second_invalid_answer_raises_with_summed_usage(self, prompt):
        rejected = _response("still bad", tokens=4)
        llm = _llm(_response("nope", tokens=10), rejected)

        with pytest.raises(StructuredOutputError) as exc_info:
            await generate_structured(llm, prompt, _Answer)

        assert isinstance(exc_info.value, RecoveryExhausted)
        assert exc_info.value.last_result is rejected
        assert exc_info.value.usage == TokenUsage(input_tokens=14, output_tokens=14)
        assert llm.agenerate.await_count == 2

    @pytest.mark.asyncio
    async def test_a_missing_message_counts_as_invalid(self, prompt):
        llm = _llm(_response(None), _response('{"value": 2}'))

        result = await generate_structured(llm, prompt, _Answer)

        assert result.output == _Answer(value=2)
        second_prompt = llm.agenerate.await_args_list[1].args[0]
        assert second_prompt.messages[-2].text == ""


class TestTextOutput:
    @pytest.mark.asyncio
    async def test_text_is_returned_as_is_with_text_format(self, prompt):
        llm = _llm(_response("hello"))
        config = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema={}
        )

        result = await generate_structured(llm, prompt, str, config)

        assert result.output == "hello"
        sent = llm.agenerate.await_args.args[1]
        assert sent.response_format is ResponseFormat.TEXT

    @pytest.mark.asyncio
    async def test_no_config_is_passed_through_for_text(self, prompt):
        llm = _llm(_response("hello"))

        await generate_structured(llm, prompt, str)

        assert llm.agenerate.await_args.args[1] is None

    @pytest.mark.asyncio
    async def test_empty_text_is_re_asked_then_raises(self, prompt):
        llm = _llm(_response("  "), _response(None, tokens=3))

        with pytest.raises(StructuredOutputError) as exc_info:
            await generate_structured(llm, prompt, str)

        assert exc_info.value.usage == TokenUsage(input_tokens=13, output_tokens=13)
        assert "empty" in llm.agenerate.await_args_list[1].args[0].messages[-1].text

    @pytest.mark.asyncio
    async def test_empty_text_then_text_succeeds(self, prompt):
        llm = _llm(_response(""), _response("ok"))

        result = await generate_structured(llm, prompt, str)

        assert result.output == "ok"


@pytest.mark.asyncio
async def test_an_unsupported_output_type_raises_type_error(prompt):
    with pytest.raises(TypeError, match="output_type"):
        await generate_structured(_llm(), prompt, int)
