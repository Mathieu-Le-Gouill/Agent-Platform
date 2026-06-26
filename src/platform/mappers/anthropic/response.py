from adapters.llm.response import LLMResponse
from mappers.anthropic.message import from_anthropic as from_anthropic_message
from errors.llm import LLMGenerationError
from models.token import TokenUsage
from adapters.llm.response import FinishReason


def from_anthropic_response(response) -> LLMResponse:
    if response is None:
        raise LLMGenerationError("Empty response returned by Anthropic")

    return LLMResponse(
        message=from_anthropic_message(response),
        usage=from_anthropic_usage(response.usage),
        model=response.model,
        finish_reason=from_anthropic_finish_reason(response.stop_reason),
    )


def from_anthropic_usage(usage) -> TokenUsage:
    if usage is None:
        return TokenUsage.zero()

    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def from_anthropic_finish_reason(reason: str | None) -> FinishReason:
    if reason is None:
        return FinishReason.UNKNOWN

    mapping = {
        "end_turn": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "stop_sequence": FinishReason.STOP_SEQUENCE,
        "tool_use": FinishReason.TOOL_CALL,
    }

    return mapping.get(reason, FinishReason.UNKNOWN)


"""
ANTHROPIC API FINISH REASONS

    "end_turn": the model reached a natural stopping point
    "max_tokens": we exceeded the requested max_tokens or the model's maximum
    "stop_sequence": one of your provided custom stop_sequences was generated
    "tool_use": the model invoked one or more tools
    "pause_turn": we paused a long-running turn. You may provide the response back as-is in a subsequent request to let the model continue.
    "refusal": when streaming classifiers intervene to handle potential policy violations
"""



"""
ANTHROPIC API RESPONSE

{
    "id": "msg_01...",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {
        "type": "text",
        "text": "The capital of France is Paris."
        }
    ],
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8
    }
}   
"""