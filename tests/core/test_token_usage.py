from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.token_usage import TokenUsageAggregator


class TestTokenUsageAggregator:
    def test_total_for_unknown_key_is_zero(self):
        aggregator = TokenUsageAggregator()

        assert aggregator.total_for("missing") == TokenUsage.zero()

    def test_record_accumulates_under_a_key(self):
        aggregator = TokenUsageAggregator()

        aggregator.record("conv-1", TokenUsage(input_tokens=10, output_tokens=5))
        aggregator.record("conv-1", TokenUsage(input_tokens=3, output_tokens=1))

        assert aggregator.total_for("conv-1") == TokenUsage(
            input_tokens=13, output_tokens=6
        )

    def test_keys_are_independent(self):
        aggregator = TokenUsageAggregator()

        aggregator.record("conv-1", TokenUsage(input_tokens=10))
        aggregator.record("conv-2", TokenUsage(input_tokens=7))

        assert aggregator.total_for("conv-1") == TokenUsage(input_tokens=10)
        assert aggregator.total_for("conv-2") == TokenUsage(input_tokens=7)

    def test_grand_total_sums_all_keys(self):
        aggregator = TokenUsageAggregator()

        aggregator.record("conv-1", TokenUsage(input_tokens=10, output_tokens=2))
        aggregator.record("conv-2", TokenUsage(input_tokens=7, output_tokens=3))

        assert aggregator.grand_total() == TokenUsage(input_tokens=17, output_tokens=5)

    def test_grand_total_with_no_records_is_zero(self):
        aggregator = TokenUsageAggregator()

        assert aggregator.grand_total() == TokenUsage.zero()

    def test_keys_lists_recorded_keys(self):
        aggregator = TokenUsageAggregator()

        aggregator.record("conv-1", TokenUsage(input_tokens=1))
        aggregator.record("conv-2", TokenUsage(input_tokens=1))

        assert set(aggregator.keys()) == {"conv-1", "conv-2"}
