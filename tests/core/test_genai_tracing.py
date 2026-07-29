from agent_platform.core.genai_tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import traced_span


class TestRecordTokenUsage:
    def test_sets_usage_attributes(self, recorded_spans):
        with traced_span("chat") as span:
            record_token_usage(span, TokenUsage(input_tokens=12, output_tokens=34))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.USAGE_INPUT_TOKENS] == 12
        assert span.attributes[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 34


class TestTracedOperationSpan:
    def test_names_span_after_operation(self, recorded_spans):
        with traced_operation_span("chat"):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "chat"

    def test_sets_operation_name_attribute(self, recorded_spans):
        with traced_operation_span("execute_tool"):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.OPERATION_NAME] == "execute_tool"

    def test_merges_extra_attributes(self, recorded_spans):
        with traced_operation_span("chat", {GenAIAttributes.REQUEST_MODEL: "gpt-4"}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.OPERATION_NAME] == "chat"
        assert span.attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4"
