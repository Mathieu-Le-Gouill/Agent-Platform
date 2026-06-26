from adapters.llm.response import FinishReason, LLMResponse
from mappers.openai.message import from_openai as from_openai_message
from models.token import TokenUsage
from errors.llm import LLMGenerationError


def from_openai_response(response) -> LLMResponse:
    if not response.choices:
        raise LLMGenerationError("No choices returned by OpenAI")

    choice = response.choices[0]

    return LLMResponse(
        message=from_openai_message(choice.message),
        usage=from_openai_usage(response.usage),
        model=response.model,
        finish_reason=from_openai_finish_reason(choice.finish_reason),
    )


def from_openai_usage(usage) -> TokenUsage:
    if usage is None:
        return TokenUsage.zero()

    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=(
            usage.output_tokens_details.reasoning_tokens
            if usage.output_tokens_details
            else 0
        ),
    )


def from_openai_finish_reason(reason: str | None) -> FinishReason:
    if reason is None:
        return FinishReason.UNKNOWN

    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALL,
        "function_call": FinishReason.TOOL_CALL,
        "content_filter": FinishReason.CONTENT_FILTER,
        "error": FinishReason.ERROR,
    }

    return mapping.get(reason, FinishReason.UNKNOWN)


"""
OPENAI CHAT COMPLETIONAPI RESPONSE
{
  "id": "chatcmpl-C9EDpkjH60VPPIB86j2zIhiR8kWiC",
  "object": "chat.completion",
  "created": 1756315657,
  "model": "gpt-5.5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Under a blanket of starlight, a sleepy unicorn tiptoed through moonlit meadows, gathering dreams like dew to tuck beneath its silver mane until morning.",
        "refusal": null,
        "annotations": []
      },
      "finish_reason": "stop"
    }
  ],
  ...
}

"""