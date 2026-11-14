from enum import Enum, IntEnum

import pytest

from agent_platform.core.schemas.enums import FinishReason
from agent_platform.integrations.llm._finish_reason import to_finish_reason


class _NativeReason(Enum):
    STOP = "STOP"


class _ProtoReason(IntEnum):
    """A protobuf-style enum, whose `.value` is an int rather than a name."""

    STOP = 1


class TestToFinishReason:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("stop", FinishReason.STOP),
            ("end_turn", FinishReason.STOP),
            ("eos_token", FinishReason.STOP),
            ("length", FinishReason.LENGTH),
            ("max_tokens", FinishReason.LENGTH),
            ("tool_calls", FinishReason.TOOL_CALL),
            ("tool_use", FinishReason.TOOL_CALL),
            ("content_filter", FinishReason.CONTENT_FILTER),
            ("safety", FinishReason.CONTENT_FILTER),
            ("error", FinishReason.ERROR),
        ],
    )
    def test_maps_vendor_spellings(self, raw, expected):
        assert to_finish_reason(raw) == expected

    def test_is_case_and_whitespace_insensitive(self):
        assert to_finish_reason("  STOP ") == FinishReason.STOP

    def test_reads_enum_values(self):
        assert to_finish_reason(_NativeReason.STOP) == FinishReason.STOP

    def test_a_non_string_enum_value_is_unknown_rather_than_a_guess(self):
        assert to_finish_reason(_ProtoReason.STOP) == FinishReason.UNKNOWN

    def test_missing_reason_is_unknown(self):
        assert to_finish_reason(None) == FinishReason.UNKNOWN

    def test_unrecognized_reason_is_unknown(self):
        assert to_finish_reason("something_new") == FinishReason.UNKNOWN
