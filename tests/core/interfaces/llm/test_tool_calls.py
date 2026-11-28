import pytest

from agent_platform.core.interfaces.llm.response import ToolCallDelta
from agent_platform.core.interfaces.llm.tool_calls import (
    assemble_tool_calls,
    tool_call_from_json,
)


class TestToolCallFromJson:
    def test_object_arguments_are_decoded(self):
        call = tool_call_from_json("c1", "get_weather", '{"location": "Paris"}')

        assert call.id == "c1"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Paris"}
        assert call.arguments_error is None

    def test_empty_arguments_mean_no_arguments(self):
        call = tool_call_from_json("c1", "ping", "")

        assert call.arguments == {}
        assert call.arguments_error is None

    def test_malformed_arguments_json_is_flagged_not_raised(self):
        call = tool_call_from_json("c1", "get_weather", "{not json")

        assert call.arguments == {}
        assert call.arguments_error is not None

    @pytest.mark.parametrize("text", ["[]", "null", '"x"', "3"])
    def test_non_object_arguments_json_is_flagged(self, text):
        call = tool_call_from_json("c1", "get_weather", text)

        assert call.arguments == {}
        assert call.arguments_error == "arguments must be a JSON object"


class TestAssembleToolCalls:
    def test_malformed_arguments_json_is_flagged_not_swallowed(self):
        deltas = [
            ToolCallDelta(
                index=0, id="c1", name="get_weather", arguments_delta="{not json"
            )
        ]

        calls = assemble_tool_calls(deltas)

        assert len(calls) == 1
        assert calls[0].id == "c1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {}
        assert calls[0].arguments_error is not None

    def test_fragments_are_joined_by_index_in_first_seen_order(self):
        deltas = [
            ToolCallDelta(index=1, id="c2", name="b", arguments_delta='{"y"'),
            ToolCallDelta(index=0, id="c1", name="a", arguments_delta='{"x": 1}'),
            ToolCallDelta(index=1, arguments_delta=": 2}"),
        ]

        first, second = assemble_tool_calls(deltas)

        assert (first.id, first.arguments) == ("c2", {"y": 2})
        assert (second.id, second.arguments) == ("c1", {"x": 1})

    def test_valid_and_empty_arguments_have_no_error(self):
        deltas = [
            ToolCallDelta(index=0, id="c1", name="a", arguments_delta='{"x": 1}'),
            ToolCallDelta(index=1, id="c2", name="b"),
        ]

        first, second = assemble_tool_calls(deltas)

        assert first.arguments == {"x": 1}
        assert second.arguments == {}
        assert first.arguments_error is None
        assert second.arguments_error is None
