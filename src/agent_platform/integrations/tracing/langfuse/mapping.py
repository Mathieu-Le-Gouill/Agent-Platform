from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from agent_platform.core.interfaces.tracing.span import AttributeValue
from agent_platform.core.observability import OPERATION_KEY

__all__ = ["ObservationType", "observation_type", "update_fields"]

ObservationType = Literal["generation", "tool", "agent", "guardrail", "chain", "span"]

# The platform's operation names on the left, the Langfuse observation type
# that renders them best on the right. Anything unlisted is a plain span.
_TYPES: dict[str, ObservationType] = {
    "chat": "generation",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "guardrail": "guardrail",
    "pipeline": "chain",
}

_MODEL_KEYS = ("llm.response.model", "llm.request.model")
_USAGE_KEYS = {
    "llm.usage.input_tokens": "input",
    "llm.usage.output_tokens": "output",
}
_COST_KEY = "llm.usage.cost"
_INPUT_KEY = "tool.arguments.content"
_OUTPUT_KEY = "tool.result.content"


def observation_type(attributes: Mapping[str, AttributeValue]) -> ObservationType:
    """The Langfuse observation type for the operation named in `attributes`."""
    operation = attributes.get(OPERATION_KEY)
    if not isinstance(operation, str):
        return "span"
    return _TYPES.get(operation, "span")


def update_fields(
    as_type: ObservationType, attributes: Mapping[str, AttributeValue]
) -> dict[str, Any]:
    """Translate platform attributes into keyword arguments of the SDK's `update()`.

    Model, usage and cost fields only exist on a `generation`: the SDK drops
    them on other types, so there they stay in `metadata` under the platform
    key. Every key with no Langfuse field of its own goes to `metadata`
    unchanged. A usage value is returned as a partial `usage_details`; the
    caller merges it, because the SDK replaces the whole value on each update.
    """
    fields: dict[str, Any] = {}
    metadata: dict[str, AttributeValue] = {}
    models: dict[str, str] = {}
    is_generation = as_type == "generation"

    for key, value in attributes.items():
        if is_generation and key in _MODEL_KEYS and isinstance(value, str):
            models[key] = value
        elif is_generation and key in _USAGE_KEYS and isinstance(value, int):
            fields.setdefault("usage_details", {})[_USAGE_KEYS[key]] = value
        elif is_generation and key == _COST_KEY and isinstance(value, (int, float)):
            fields["cost_details"] = {"total": float(value)}
        elif key == _INPUT_KEY:
            fields["input"] = value
        elif key == _OUTPUT_KEY:
            fields["output"] = value
        else:
            metadata[key] = value

    for key in _MODEL_KEYS:
        if key in models:
            fields["model"] = models[key]
            break
    if metadata:
        fields["metadata"] = metadata
    return fields
