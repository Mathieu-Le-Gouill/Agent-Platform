from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Generic, TypeVar

from agent_platform.core.persistence import Checkpointer
from agent_platform.core.tracing import traced_span
from agent_platform.workflows.errors import (
    WorkflowExecutionError,
    WorkflowValidationError,
)
from agent_platform.workflows.state import WorkflowCheckpoint, WorkflowState

__all__ = ["END", "NodeFn", "Router", "WorkflowGraph", "CompiledWorkflow"]

StateT = TypeVar("StateT", bound=WorkflowState)

#: Sentinel edge target marking the end of a run. Not a real node name, so it
#: can never collide with one (`WorkflowGraph.add_node` rejects it).
END = "__end__"

NodeFn = Callable[[StateT], Awaitable[StateT]]
Router = Callable[[StateT], str]


class WorkflowGraph(Generic[StateT]):
    """Builder for a typed-state DAG: register nodes, wire edges (static or
    conditional), then `compile()` into a `CompiledWorkflow` that validates
    the graph once up front instead of failing mid-run.

    Deliberately dependency-free (no LangGraph/LangChain): a node is just an
    `async def(state: StateT) -> StateT`, so anything already shaped that way
    -- an `agent_node`, a `tool_node`, a bare function -- composes without an
    adapter layer beyond what `workflows/nodes/` already provides.
    """

    def __init__(self, state_cls: type[StateT]) -> None:
        self._state_cls = state_cls
        self._nodes: dict[str, NodeFn[StateT]] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, tuple[Router[StateT], dict[str, str]]] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, fn: NodeFn[StateT]) -> WorkflowGraph[StateT]:
        if name == END:
            raise WorkflowValidationError(
                f"'{END}' is reserved and not a valid node name"
            )
        if name in self._nodes:
            raise WorkflowValidationError(f"node '{name}' is already registered")
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> WorkflowGraph[StateT]:
        self._entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> WorkflowGraph[StateT]:
        """Unconditional transition: after `from_node` runs, go to `to_node`
        (or `END`)."""
        self._edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self, from_node: str, router: Router[StateT], path_map: dict[str, str]
    ) -> WorkflowGraph[StateT]:
        """After `from_node` runs, call `router(state)` and look its return
        value up in `path_map` to find the next node (or `END`)."""
        self._conditional_edges[from_node] = (router, path_map)
        return self

    def compile(self, *, max_steps: int = 100) -> CompiledWorkflow[StateT]:
        entry_point = self._entry_point
        if entry_point is None:
            raise WorkflowValidationError("call set_entry_point() before compile()")
        if entry_point not in self._nodes:
            raise WorkflowValidationError(
                f"entry point '{entry_point}' is not a registered node"
            )

        for source in (*self._edges, *self._conditional_edges):
            if source not in self._nodes:
                raise WorkflowValidationError(
                    f"edge source '{source}' is not a registered node"
                )

        targets = list(self._edges.values())
        for _router, path_map in self._conditional_edges.values():
            targets.extend(path_map.values())
        for target in targets:
            if target != END and target not in self._nodes:
                raise WorkflowValidationError(
                    f"edge target '{target}' is not a registered node"
                )

        for name in self._nodes:
            if name not in self._edges and name not in self._conditional_edges:
                raise WorkflowValidationError(
                    f"node '{name}' has no outgoing edge; "
                    "add one to another node or to END"
                )

        unreachable = self._nodes.keys() - self._reachable_nodes(entry_point)
        if unreachable:
            raise WorkflowValidationError(
                f"unreachable from entry point '{entry_point}': {sorted(unreachable)}"
            )

        return CompiledWorkflow(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entry_point=entry_point,
            max_steps=max_steps,
        )

    def _reachable_nodes(self, entry_point: str) -> set[str]:
        visited: set[str] = set()
        queue = [entry_point]
        while queue:
            current = queue.pop()
            if current == END or current in visited:
                continue
            visited.add(current)
            if current in self._edges:
                queue.append(self._edges[current])
            if current in self._conditional_edges:
                _router, path_map = self._conditional_edges[current]
                queue.extend(path_map.values())
        return visited


class CompiledWorkflow(Generic[StateT]):
    """A validated, runnable graph. Only `WorkflowGraph.compile()` constructs
    one, so by the time `arun`/`resume` execute, the graph shape is already
    known-good."""

    def __init__(
        self,
        *,
        nodes: dict[str, NodeFn[StateT]],
        edges: dict[str, str],
        conditional_edges: dict[str, tuple[Router[StateT], dict[str, str]]],
        entry_point: str,
        max_steps: int = 100,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._max_steps = max_steps

    def _next_node(self, current: str, state: StateT) -> str:
        if current in self._conditional_edges:
            router, path_map = self._conditional_edges[current]
            key = router(state)
            if key not in path_map:
                raise WorkflowExecutionError(
                    f"router at node '{current}' returned '{key}', "
                    f"not one of {sorted(path_map)}"
                )
            return path_map[key]
        return self._edges[current]

    async def arun(
        self,
        initial_state: StateT,
        *,
        checkpointer: Checkpointer[WorkflowCheckpoint[StateT]] | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[StateT]:
        """Run from the entry point, yielding the state after every node so a
        caller can stream/observe intermediate progress rather than only the
        final result.

        If `checkpointer` is given, `run_id` must be too: the state (and
        which node runs next) is saved after each step, keyed by `run_id`, so
        `resume(run_id, checkpointer)` can continue an interrupted run.
        """
        if checkpointer is not None and run_id is None:
            raise WorkflowExecutionError("run_id is required when checkpointer is set")

        async for state in self._run_from(
            self._entry_point, initial_state, checkpointer, run_id
        ):
            yield state

    async def resume(
        self,
        run_id: str,
        checkpointer: Checkpointer[WorkflowCheckpoint[StateT]],
    ) -> AsyncIterator[StateT]:
        """Reload the last checkpoint saved under `run_id` and continue the
        run from the node it was about to enter."""
        checkpoint = await checkpointer.load(run_id)
        if checkpoint is None:
            raise WorkflowExecutionError(f"no checkpoint found for run_id '{run_id}'")

        async for state in self._run_from(
            checkpoint.next_node, checkpoint.state, checkpointer, run_id
        ):
            yield state

    async def _run_from(
        self,
        start_node: str,
        state: StateT,
        checkpointer: Checkpointer[WorkflowCheckpoint[StateT]] | None,
        run_id: str | None,
    ) -> AsyncIterator[StateT]:
        current = start_node
        steps = 0
        while True:
            steps += 1
            if steps > self._max_steps:
                raise WorkflowExecutionError(
                    f"workflow exceeded max_steps ({self._max_steps}); "
                    "a conditional edge may be looping without converging"
                )

            node_fn = self._nodes[current]
            with traced_span("workflow_node", {"workflow.node": current}):
                state = await node_fn(state)
            next_node = self._next_node(current, state)

            if checkpointer is not None:
                assert run_id is not None
                await checkpointer.save(
                    run_id, WorkflowCheckpoint(next_node=next_node, state=state)
                )

            yield state

            if next_node == END:
                return
            current = next_node
