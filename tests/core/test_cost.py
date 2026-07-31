from agent_platform.core.cost import ModelPricing, StaticPricingTable
from agent_platform.core.schemas.token import TokenUsage


class TestStaticPricingTable:
    def test_cost_for_known_model(self):
        table = StaticPricingTable(
            {"gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015)}
        )
        usage = TokenUsage(input_tokens=2000, output_tokens=1000)

        cost = table.cost_for("gpt-4o", usage)

        assert cost == 0.005 * 2 + 0.015 * 1

    def test_cost_for_unknown_model_is_zero(self):
        table = StaticPricingTable({})

        cost = table.cost_for(
            "unknown", TokenUsage(input_tokens=1000, output_tokens=1000)
        )

        assert cost == 0.0

    def test_cost_for_zero_usage(self):
        table = StaticPricingTable(
            {"m": ModelPricing(input_per_1k=1.0, output_per_1k=1.0)}
        )

        assert table.cost_for("m", TokenUsage.zero()) == 0.0
