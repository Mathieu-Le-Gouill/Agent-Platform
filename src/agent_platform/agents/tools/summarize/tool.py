from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.schemas.message import Prompt


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
        system = "Summarize the following text concisely."
        if validated.max_words:
            system += f" Limit the summary to {validated.max_words} words."

        response = await safe_call(
            self._llm.agenerate(Prompt.build(system=system, user=validated.text)),
            "Summarization failed",
        )
        if response.message is None or not response.message.text:
            raise ToolError("Summarization returned no content")
        return response.message.text
