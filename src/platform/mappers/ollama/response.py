from adapters.llm.response import FinishReason, LLMResponse
from mappers.ollama.message import from_ollama as from_ollama_message
from models.token import TokenUsage
from errors.llm import LLMGenerationError


def from_ollama_response(response) -> LLMResponse:
    if response is None:
        raise LLMGenerationError(
            "Empty response returned by Ollama"
        )

    return LLMResponse(
        message=from_ollama_message(response.message),
        usage=TokenUsage.zero(),
        model=response.model,
        finish_reason=from_ollama_finish_reason(
            response.done_reason
        ),
    )


def from_ollama_finish_reason(
    reason: str | None,
) -> FinishReason:

    if reason is None:
        return FinishReason.UNKNOWN

    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
    }

    return mapping.get(
        reason,
        FinishReason.UNKNOWN,
    )


"""
OLLAMA API RESPONSE
{
  "model": "<string>",
  "created_at": "<string>",
  "response": "<string>",
  "thinking": "<string>",
  "done": true,
  "done_reason": "<string>",
  "total_duration": 123,
  "load_duration": 123,
  "prompt_eval_count": 123,
  "prompt_eval_duration": 123,
  "eval_count": 123,
  "eval_duration": 123,
  "logprobs": [
    {
      "token": "<string>",
      "logprob": 123,
      "bytes": [
        123
      ],
      "top_logprobs": [
        {
          "token": "<string>",
          "logprob": 123,
          "bytes": [
            123
          ]
        }
      ]
    }
  ]
}
"""