import pytest

from agent_platform.models.token import TokenUsage


class TestTokenUsage:
    def test_zero_returns_empty(self):
        usage = TokenUsage.zero()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.reasoning_tokens == 0

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=10)
        assert usage.total_tokens == 160

    def test_total_tokens_only_input(self):
        usage = TokenUsage(input_tokens=42)
        assert usage.total_tokens == 42

    def test_total_tokens_only_output(self):
        usage = TokenUsage(output_tokens=77)
        assert usage.total_tokens == 77

    def test_total_tokens_all_zero(self):
        usage = TokenUsage.zero()
        assert usage.total_tokens == 0

    def test_add(self):
        a = TokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
        b = TokenUsage(input_tokens=30, output_tokens=40, reasoning_tokens=15)
        c = a + b
        assert c.input_tokens == 40
        assert c.output_tokens == 60
        assert c.reasoning_tokens == 20

    def test_add_with_zero(self):
        a = TokenUsage(input_tokens=5, output_tokens=5)
        z = TokenUsage.zero()
        r = a + z
        assert r.input_tokens == 5
        assert r.output_tokens == 5

    def test_add_does_not_mutate(self):
        a = TokenUsage(input_tokens=1, output_tokens=2)
        b = TokenUsage(input_tokens=3, output_tokens=4)
        _ = a + b
        assert a.input_tokens == 1
        assert b.input_tokens == 3

    def test_frozen(self):
        usage = TokenUsage(input_tokens=10, output_tokens=20)
        with pytest.raises(ValueError):
            usage.input_tokens = 99

    def test_constructor_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.reasoning_tokens == 0
