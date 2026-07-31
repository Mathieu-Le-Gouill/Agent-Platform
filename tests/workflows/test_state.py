from agent_platform.core.schemas.message import UserMessage
from agent_platform.workflows.state import (
    HandoffState,
    MessagesState,
    WorkflowCheckpoint,
)


class TestMessagesState:
    def test_defaults_to_empty_messages(self):
        state = MessagesState()
        assert state.messages == []

    def test_model_copy_replaces_messages(self):
        state = MessagesState()
        new_messages = [UserMessage(content="hi")]
        updated = state.model_copy(update={"messages": new_messages})
        assert updated.messages == new_messages
        assert state.messages == []


class TestHandoffState:
    def test_defaults_to_no_active_agent(self):
        state = HandoffState()
        assert state.active_agent == ""
        assert state.messages == []


class TestWorkflowCheckpoint:
    def test_holds_next_node_and_state(self):
        state = MessagesState(messages=[UserMessage(content="hi")])
        checkpoint = WorkflowCheckpoint(next_node="step_2", state=state)
        assert checkpoint.next_node == "step_2"
        assert checkpoint.state.messages == state.messages
