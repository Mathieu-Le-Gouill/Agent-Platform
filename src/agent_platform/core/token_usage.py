from __future__ import annotations

from agent_platform.core.schemas.token import TokenUsage

__all__ = ["TokenUsageAggregator"]


class TokenUsageAggregator:
    def __init__(self) -> None:
        self._totals: dict[str, TokenUsage] = {}

    def record(self, key: str, usage: TokenUsage) -> None:
        self._totals[key] = self._totals.get(key, TokenUsage.zero()) + usage

    def total_for(self, key: str) -> TokenUsage:
        return self._totals.get(key, TokenUsage.zero())

    def grand_total(self) -> TokenUsage:
        total = TokenUsage.zero()
        for usage in self._totals.values():
            total += usage
        return total

    def keys(self) -> list[str]:
        return list(self._totals.keys())
