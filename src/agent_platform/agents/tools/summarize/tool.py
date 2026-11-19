from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.components.generation.llm_task.component import LLMTaskInput
from agent_platform.components.generation.llm_task.summarize import summarize_task
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.structured import StructuredOutputError


class SummarizeInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_words: int | None = Field(
        default=None, ge=1, description="Maximum length of the summary, in words"
    )


class SummarizeTool(Tool):
    name = "summarize"
    description = "Summarize a piece of text concisely."
    input_schema = SummarizeInput
    output_schema = None

    def __init__(self, llm: BaseLLMProvider) -> None:
        self._llm = llm

    async def run(self, **kwargs: Any) -> str:
        validated = SummarizeInput(**kwargs)
        task = summarize_task(self._llm, max_words=validated.max_words)

        async def summarize() -> str:
            try:
                result = await task.arun(LLMTaskInput(validated.text, None))
            except StructuredOutputError as exc:
                # Raised as a ToolError here so `safe_call` passes it through.
                raise ToolError("Summarization returned no content") from exc
            return result.output

        return await safe_call(summarize(), "Summarization failed")
