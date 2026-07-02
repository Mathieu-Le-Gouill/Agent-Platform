from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0


    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


    @classmethod
    def zero(cls) -> TokenUsage:
        return cls(input_tokens=0, output_tokens=0, reasoning_tokens=0)