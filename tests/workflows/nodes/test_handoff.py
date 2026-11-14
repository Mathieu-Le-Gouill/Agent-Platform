import pytest

from agent_platform.agents.agent import Agent
from agent_platform.core.observability import WorkflowNode, observe
from agent_platform.core.schemas.message import UserMessage
from agent_platform.workflows.nodes.handoff import handoff_node
from agent_platform.workflows.state import HandoffState
from tests.helpers import make_fake_llm_response


class TestHandoffNode:
    @pytest.mark.asyncio
    async def test_runs_agent_and_stamps_active_agent(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="handled")
        agent = Agent(name="specialist", llm=mock_llm)
        node = handoff_node(agent)

        state = HandoffState(messages=[UserMessage(content="hi")])
        new_state = await node(state)

        assert new_state.active_agent == "specialist"
        assert new_state.messages[-1].text == "handled"

    @pytest.mark.asyncio
    async def test_second_handoff_overwrites_active_agent(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="ok")
        agent_a = Agent(name="agent_a", llm=mock_llm)
        agent_b = Agent(name="agent_b", llm=mock_llm)

        state = HandoffState(messages=[UserMessage(content="hi")])
        state = await handoff_node(agent_a)(state)
        state = await handoff_node(agent_b)(state)

        assert state.active_agent == "agent_b"

    @pytest.mark.asyncio
    async def test_records_the_agent_it_handed_to(self, mock_llm, recorded_spans):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="handled")
        node = handoff_node(Agent(name="specialist", llm=mock_llm))

        with observe(WorkflowNode(node="delegate")):
            await node(HandoffState(messages=[UserMessage(content="hi")]))

        node_span = next(
            s for s in recorded_spans.get_finished_spans() if s.name == "workflow_node"
        )
        assert node_span.attributes["workflow.handoff.to"] == "specialist"
