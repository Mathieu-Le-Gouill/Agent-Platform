from adapters.llm.response import FinishReason, LLMResponse
from mappers.mistral.message import from_mistral as from_mistral_message
from models.token import TokenUsage
from errors.llm import LLMGenerationError


def from_mistral_response(response) -> LLMResponse:
    if not response.choices:
        raise LLMGenerationError(
            "No choices returned by Mistral"
        )

    choice = response.choices[0]

    return LLMResponse(
        message=from_mistral_message(choice.message),
        usage=from_mistral_usage(response.usage),
        model=response.model,
        finish_reason=from_mistral_finish_reason(
            choice.finish_reason
        ),
    )


def from_mistral_usage(usage) -> TokenUsage:
    if usage is None:
        return TokenUsage.zero()

    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
    )


def from_mistral_finish_reason(
    reason: str | None,
) -> FinishReason:

    if reason is None:
        return FinishReason.UNKNOWN

    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "model_length": FinishReason.LENGTH,
        "error": FinishReason.ERROR,
        "tool_calls": FinishReason.TOOL_CALL,
    }

    return mapping.get(
        reason,
        FinishReason.UNKNOWN,
    )


"""
MISTRAL API RESPONSE

{
  "choices": [
    {
        "finish_reason": "stop"|"length"|"model_length"|"error"|"tool_calls",
        "index": 0,
        "message": {
            "content": "Hello! How can I assist you today?",
            "role": "assistant"
        },
        "messages": {}

    }
  ],
  "created": "1702256327",
  "id": "cmpl-e5cc70bb28c444948073e77776eb30ef",
  "model": "mistral-small-latest",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 7,
    "prompt_audio_seconds": 0,
    "num_cached_tokens": 0,
    "prompt_tokens": 13,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "total_tokens": 20
  }
}
"""
