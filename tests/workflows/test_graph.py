import pytest

from agent_platform.core.persistence import InMemoryCheckpointer
from agent_platform.workflows.errors import (
    WorkflowExecutionError,
    WorkflowValidationError,
)
from agent_platform.workflows.graph import END, WorkflowGraph
from agent_platform.workflows.nodes.router import FALSE, TRUE, binary_router
from agent_platform.workflows.state import WorkflowCheckpoint, WorkflowState


class CounterState(WorkflowState):
    count: int = 0
    log: tuple[str, ...] = ()


async def _increment(state: CounterState) -> CounterState:
    return state.model_copy(
        update={"count": state.count + 1, "log": (*state.log, "inc")}
    )


async def _double(state: CounterState) -> CounterState:
    return state.model_copy(
        update={"count": state.count * 2, "log": (*state.log, "double")}
    )


def _build_linear_graph() -> WorkflowGraph[CounterState]:
    graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
    graph.add_node("increment", _increment)
    graph.add_node("double", _double)
    graph.set_entry_point("increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double", END)
    return graph


class TestCompileValidation:
    def test_missing_entry_point_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        with pytest.raises(WorkflowValidationError, match="set_entry_point"):
            graph.compile()

    def test_entry_point_not_a_node_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("missing")
        with pytest.raises(WorkflowValidationError, match="not a registered node"):
            graph.compile()

    def test_reserved_end_name_rejected(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        with pytest.raises(WorkflowValidationError, match="reserved"):
            graph.add_node(END, _increment)

    def test_duplicate_node_name_rejected(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        with pytest.raises(WorkflowValidationError, match="already registered"):
            graph.add_node("increment", _increment)

    def test_edge_source_not_a_node_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("increment")
        graph.add_edge("increment", END)
        graph.add_edge("phantom", END)
        with pytest.raises(WorkflowValidationError, match="edge source"):
            graph.compile()

    def test_edge_target_not_a_node_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("increment")
        graph.add_edge("increment", "nowhere")
        with pytest.raises(WorkflowValidationError, match="edge target"):
            graph.compile()

    def test_node_with_no_outgoing_edge_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("increment")
        with pytest.raises(WorkflowValidationError, match="no outgoing edge"):
            graph.compile()

    def test_unreachable_node_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.add_node("double", _double)
        graph.set_entry_point("increment")
        graph.add_edge("increment", END)
        graph.add_edge("double", END)
        with pytest.raises(WorkflowValidationError, match="unreachable"):
            graph.compile()

    def test_valid_linear_graph_compiles(self):
        _build_linear_graph().compile()


class TestArun:
    @pytest.mark.asyncio
    async def test_yields_state_after_each_node(self):
        workflow = _build_linear_graph().compile()
        states = [state async for state in workflow.arun(CounterState())]
        assert [s.log for s in states] == [("inc",), ("inc", "double")]
        assert states[-1].count == 2

    @pytest.mark.asyncio
    async def test_conditional_edges_route_by_router_key(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.add_node("double", _double)
        graph.set_entry_point("increment")
        graph.add_conditional_edges(
            "increment",
            binary_router(lambda s: s.count > 0),
            {TRUE: "double", FALSE: END},
        )
        graph.add_edge("double", END)
        workflow = graph.compile()

        states = [state async for state in workflow.arun(CounterState())]

        assert states[-1].count == 2

    @pytest.mark.asyncio
    async def test_router_returning_unknown_key_raises(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("increment")
        graph.add_conditional_edges("increment", lambda s: "nope", {"yes": END})
        workflow = graph.compile()

        with pytest.raises(WorkflowExecutionError, match="not one of"):
            async for _ in workflow.arun(CounterState()):
                pass

    @pytest.mark.asyncio
    async def test_max_steps_guards_against_a_non_converging_loop(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.set_entry_point("increment")
        graph.add_edge("increment", "increment")
        workflow = graph.compile(max_steps=3)

        with pytest.raises(WorkflowExecutionError, match="max_steps"):
            async for _ in workflow.arun(CounterState()):
                pass

    @pytest.mark.asyncio
    async def test_checkpointer_without_run_id_raises(self):
        workflow = _build_linear_graph().compile()
        checkpointer = InMemoryCheckpointer()
        with pytest.raises(WorkflowExecutionError, match="run_id"):
            async for _ in workflow.arun(CounterState(), checkpointer=checkpointer):
                pass


class TestCheckpointingAndResume:
    @pytest.mark.asyncio
    async def test_saves_a_checkpoint_after_every_node(self):
        workflow = _build_linear_graph().compile()
        checkpointer = InMemoryCheckpointer()

        async for _ in workflow.arun(
            CounterState(), checkpointer=checkpointer, run_id="run-1"
        ):
            pass

        checkpoint = await checkpointer.load("run-1")
        assert checkpoint is not None
        assert checkpoint.next_node == END
        assert checkpoint.state.count == 2

    @pytest.mark.asyncio
    async def test_resume_continues_from_the_next_node(self):
        graph: WorkflowGraph[CounterState] = WorkflowGraph(CounterState)
        graph.add_node("increment", _increment)
        graph.add_node("double", _double)
        graph.set_entry_point("increment")
        graph.add_edge("increment", "double")
        graph.add_edge("double", END)
        workflow = graph.compile()
        checkpointer = InMemoryCheckpointer()

        # Simulate a run interrupted right after "increment".
        checkpoint_state = CounterState(count=1, log=("inc",))
        await checkpointer.save(
            "run-2", WorkflowCheckpoint(next_node="double", state=checkpoint_state)
        )

        states = [state async for state in workflow.resume("run-2", checkpointer)]

        assert len(states) == 1
        assert states[0].count == 2
        assert states[0].log == ("inc", "double")

    @pytest.mark.asyncio
    async def test_resume_with_no_checkpoint_raises(self):
        workflow = _build_linear_graph().compile()
        checkpointer = InMemoryCheckpointer()
        with pytest.raises(WorkflowExecutionError, match="no checkpoint"):
            async for _ in workflow.resume("missing-run", checkpointer):
                pass
