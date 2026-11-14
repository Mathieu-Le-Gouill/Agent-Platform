from agent_platform.core.observability import (
    OPERATION_KEY,
    AgentRun,
    ComponentRun,
    LLMBatch,
    LLMCall,
    ModelCost,
    ToolRun,
    flatten,
)


class TestLLMCall:
    def test_exports_the_request_and_response_facts(self):
        attributes = flatten(
            LLMCall(
                provider="openai",
                request_model="gpt-4",
                response_id="resp-1",
                input_tokens=3,
                output_tokens=7,
            )
        )

        assert attributes["llm.provider"] == "openai"
        assert attributes["llm.request.model"] == "gpt-4"
        assert attributes["llm.response.id"] == "resp-1"
        assert attributes["llm.usage.input_tokens"] == 3
        assert attributes["llm.usage.output_tokens"] == 7

    def test_omits_usage_that_the_provider_never_reported(self):
        attributes = flatten(LLMCall(provider="openai", request_model="gpt-4"))

        assert "llm.usage.input_tokens" not in attributes
        assert "llm.usage.output_tokens" not in attributes


class TestLLMBatch:
    def test_names_the_operation_after_the_stage(self):
        assert LLMBatch(stage="submit", provider="openai").operation_name == (
            "llm_batch_submit"
        )


class TestModelCost:
    def test_has_no_operation_of_its_own(self):
        attributes = flatten(ModelCost(agent_name="bot", model="gpt-4", cost=0.5))

        assert OPERATION_KEY not in attributes

    def test_carries_the_cost_onto_the_enclosing_span(self):
        attributes = flatten(ModelCost(agent_name="bot", model="gpt-4", cost=0.5))

        assert attributes["llm.usage.cost"] == 0.5
        assert attributes["agent.name"] == "bot"


class TestToolRun:
    def test_arguments_and_results_are_payloads(self):
        attributes = flatten(ToolRun(tool_name="search", arguments={"q": "secret"}))

        assert attributes["tool.arguments.size"] > 0
        assert "tool.arguments" not in attributes


class TestAgentRun:
    def test_exports_the_loop_facts(self):
        attributes = flatten(
            AgentRun(agent_name="bot", conversation_id="c-1", iterations=3)
        )

        assert attributes["conversation.id"] == "c-1"
        assert attributes["agent.iterations"] == 3


class TestComponentRun:
    def test_is_named_after_the_component_operation(self):
        assert flatten(ComponentRun(name="embed"))[OPERATION_KEY] == "embed"
