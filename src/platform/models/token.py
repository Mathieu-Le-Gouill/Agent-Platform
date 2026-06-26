from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


    @classmethod
    def zero(cls) -> TokenUsage:
        return cls(prompt_tokens=0, completion_tokens=0)