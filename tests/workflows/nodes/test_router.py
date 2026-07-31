from agent_platform.workflows.nodes.router import FALSE, TRUE, binary_router
from agent_platform.workflows.state import WorkflowState


class FlagState(WorkflowState):
    flag: bool = False


class TestBinaryRouter:
    def test_returns_true_key_when_predicate_is_true(self):
        router = binary_router(lambda s: s.flag)
        assert router(FlagState(flag=True)) == TRUE

    def test_returns_false_key_when_predicate_is_false(self):
        router = binary_router(lambda s: s.flag)
        assert router(FlagState(flag=False)) == FALSE
