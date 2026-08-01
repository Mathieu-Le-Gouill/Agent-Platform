from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from agent_platform.core.schemas.token import TokenUsage

__all__ = ["ModelPricing", "CostEstimator", "StaticPricingTable"]


class ModelPricing(BaseModel, frozen=True):
    input_per_1k: float
    output_per_1k: float


class CostEstimator(Protocol):
    def cost_for(self, model: str, usage: TokenUsage) -> float: ...


class StaticPricingTable:
    """`CostEstimator` backed by a fixed `{model: ModelPricing}` table.

    Prices change independently of this codebase's release cycle, so no
    vendor table ships here - callers construct this with whatever prices
    they want (or seed one in `config/container.py`).
    """

    def __init__(self, prices: dict[str, ModelPricing]) -> None:
        self._prices = prices

    def cost_for(self, model: str, usage: TokenUsage) -> float:
        pricing = self._prices.get(model)
        if pricing is None:
            return 0.0  # unknown model: cost silently omitted rather than raising
        return (
            usage.input_tokens / 1000 * pricing.input_per_1k
            + usage.output_tokens / 1000 * pricing.output_per_1k
        )
