from __future__ import annotations

from agent_platform.components.generation.llm_task.component import LLMTask
from agent_platform.core.interfaces.llm.base import BaseLLMProvider


def summarize_task(
    llm: BaseLLMProvider, *, max_words: int | None = None
) -> LLMTask[str]:
    """Returns a task that summarizes its input text, in at most `max_words` words."""
    instructions = "Summarize the following text concisely."
    if max_words:
        instructions += f" Limit the summary to {max_words} words."
    return LLMTask(llm, instructions, str)
