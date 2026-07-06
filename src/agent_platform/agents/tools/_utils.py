from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Coroutine, Sequence, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from agent_platform.models.chunk import AudioChunk, TextChunk
from agent_platform.models.enums import AudioFormat, DataType

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

T = TypeVar("T")


async def safe_call(
    coro: Coroutine[Any, Any, T],
    error_message: str = "Provider execution failed",
) -> T:
    from agent_platform.agents.tools.base import ToolError

    try:
        return await coro
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{error_message}: {exc}") from exc


def audio_chunk(
    data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        dtype=DataType.FLOAT32,
        format=AudioFormat.UNKNOWN,
    )


def text_chunk(text: str) -> TextChunk:
    return TextChunk(id=uuid4(), text=text, index=0)


def filter_by_confidence(
    chunks: Sequence[TextChunk],
    min_confidence: float = 0.0,
) -> list[TextChunk]:
    if min_confidence <= 0.0:
        return list(chunks)
    return [
        c for c in chunks if (c.metadata.get("confidence") or 1.0) >= min_confidence
    ]


@functools.lru_cache(maxsize=128)
def _model_schema(model: type[BaseModel]) -> dict[str, Any]:
    if model is BaseModel:
        return {"type": "object", "properties": {}}
    return model.model_json_schema()


def tool_to_openai_schema(tool: Tool) -> dict[str, Any]:
    schema = _model_schema(tool.input_schema)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }


def tool_to_anthropic_schema(tool: Tool) -> dict[str, Any]:
    schema = _model_schema(tool.input_schema)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


def tool_schema_description(tool: Tool, provider: str = "openai") -> dict[str, Any]:
    if provider == "openai":
        return tool_to_openai_schema(tool)
    if provider == "anthropic":
        return tool_to_anthropic_schema(tool)
    raise ValueError(f"Unsupported provider: {provider}")
