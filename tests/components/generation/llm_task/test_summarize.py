from unittest.mock import AsyncMock, MagicMock

from agent_platform.components.generation.llm_task.component import LLMTaskInput
from agent_platform.components.generation.llm_task.summarize import summarize_task
from tests.helpers import make_fake_llm_response


def _llm(text: str) -> MagicMock:
    llm = MagicMock()
    llm.agenerate = AsyncMock(return_value=make_fake_llm_response(text))
    return llm


class TestSummarizeTask:
    async def test_returns_the_summary_text(self):
        llm = _llm("Short.")

        result = await summarize_task(llm).arun(LLMTaskInput("long text", None))

        assert result.output == "Short."
        prompt = llm.agenerate.await_args.args[0]
        assert prompt.last_user_message().text == "long text"
        assert "words" not in prompt.system_prompt()

    async def test_max_words_goes_into_the_instructions(self):
        llm = _llm("Short.")

        await summarize_task(llm, max_words=20).arun(LLMTaskInput("long text", None))

        assert "20 words" in llm.agenerate.await_args.args[0].system_prompt()
