import pytest

from agent_platform.agents.agent import Agent
from agent_platform.core.schemas.message import UserMessage
from agent_platform.workflows.nodes.agent_node import agent_node
from agent_platform.workflows.state import MessagesState
from tests.helpers import make_fake_llm_response


class TestAgentNode:
    @pytest.mark.asyncio
    async def test_runs_agent_and_appends_turn(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hello there")
        agent = Agent(name="assistant", llm=mock_llm)
        node = agent_node(agent)

        state = MessagesState(messages=[UserMessage(content="hi")])
        new_state = await node(state)

        assert len(new_state.messages) == 2
        assert new_state.messages[-1].text == "Hello there"
        # Original state is untouched (state flows by value).
        assert len(state.messages) == 1

    @pytest.mark.asyncio
    async def test_max_iterations_is_configurable(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="ok")
        agent = Agent(name="assistant", llm=mock_llm)
        node = agent_node(agent, max_iterations=1)

        state = MessagesState(messages=[UserMessage(content="hi")])
        new_state = await node(state)

        assert new_state.messages[-1].text == "ok"
