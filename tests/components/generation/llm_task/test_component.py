from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from agent_platform.components.generation.llm_task.component import (
    LLMTask,
    LLMTaskInput,
)
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.interfaces.llm.structured import StructuredResult
from tests.helpers import make_fake_llm_response


class _Answer(BaseModel):
    value: int


def _llm(*texts: str) -> MagicMock:
    llm = MagicMock()
    llm.agenerate = AsyncMock(side_effect=[make_fake_llm_response(t) for t in texts])
    return llm


class TestLLMTask:
    async def test_sends_instructions_as_system_and_text_as_user(self):
        llm = _llm('{"value": 4}')
        task = LLMTask(llm, "Extract the number.", _Answer)

        result = await task.arun(LLMTaskInput("there are 4 cats", None))

        assert isinstance(result, StructuredResult)
        assert result.output == _Answer(value=4)
        prompt = llm.agenerate.await_args.args[0]
        assert prompt.system_prompt() == "Extract the number."
        assert prompt.last_user_message().text == "there are 4 cats"

    async def test_the_call_config_reaches_the_provider(self):
        llm = _llm('{"value": 4}')
        task = LLMTask(llm, "Extract.", _Answer)

        await task.arun(LLMTaskInput("x", GenerationConfig(model="other")))

        sent = llm.agenerate.await_args.args[1]
        assert sent.model == "other"
        assert sent.response_format is ResponseFormat.JSON_SCHEMA

    async def test_a_text_task_returns_the_text(self):
        task = LLMTask(_llm("done"), "Say done.", str)

        result = await task.arun(LLMTaskInput("x", None))

        assert result.output == "done"
