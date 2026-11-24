from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from agent_platform.core.observability import WorkflowNode, observe
from agent_platform.core.persistence import Checkpointer
from agent_platform.workflows.errors import (
    WorkflowExecutionError,
    WorkflowValidationError,
)
from agent_platform.workflows.state import WorkflowCheckpoint, WorkflowState

__all__ = [
    "END",
    "NodeFn",
    "Router",
    "StepContext",
    "WorkflowGraph",
    "CompiledWorkflow",
    "bind_step",
    "current_step",
]

StateT = TypeVar("StateT", bound=WorkflowState)


@dataclass
class StepContext:
    """Where a running node sits in its run, for naming the tool calls it makes.

    `run_id`, `node` and `step` are the same on a first run and on `resume`,
    and the call counter restarts with each node run, so a node that is re-run
    after a crash builds the same call ids in the same order. `branch` is set
    inside a `fan_out`, so concurrent branches calling the same tool get
    different ids. Each branch has its own context, and so its own counter.
    """

    run_id: str | None
    node: str
    step: int
    branch: str | None = None
    _calls: int = field(default=0, repr=False)

    def next_tool_call_id(self) -> str | None:
        """Return the id for the next tool call of this node run.

        `None` when the run has no `run_id`: the caller then picks a random id.
        """
        if self.run_id is None:
            return None
        self._calls += 1
        parts = [self.run_id, self.node, str(self.step)]
        if self.branch is not None:
            parts.append(self.branch)
        parts.append(str(self._calls))
        return ":".join(parts)

    def for_branch(self, name: str) -> StepContext:
        """Return a fresh context for one `fan_out` branch, named after `name`
        (appended to the enclosing branch when fan-outs nest)."""
        branch = name if self.branch is None else f"{self.branch}:{name}"
        return StepContext(self.run_id, self.node, self.step, branch)


_current_step: ContextVar[StepContext | None] = ContextVar(
    "current_workflow_step", default=None
)


def current_step() -> StepContext | None:
    """Return the context of the node being run, or `None` outside a run."""
    return _current_step.get()


@contextmanager
def bind_step(context: StepContext) -> Iterator[None]:
    """Make `context` the current step for the duration of the block."""
    token = _current_step.set(context)
    try:
        yield
    finally:
        _current_step.reset(token)


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

    def _next_node(self, current: str, state: StateT) -> tuple[str, str | None]:
        """The node to run after `current`, and the route key a router picked
        to get there (`None` for a static edge, which had no decision to make)."""
        if current in self._conditional_edges:
            router, path_map = self._conditional_edges[current]
            key = router(state)
            if key not in path_map:
                raise WorkflowExecutionError(
                    f"router at node '{current}' returned '{key}', "
                    f"not one of {sorted(path_map)}"
                )
            return path_map[key], key
        return self._edges[current], None

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

        `run_id` must be unique per run: it seeds the idempotency key of each
        `tool_node` call (see `StepContext`), so two runs sharing one would
        share keys and a tool would treat the second run's call as a repeat.
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

        if checkpoint.next_node == END:
            return
        if checkpoint.next_node not in self._nodes:
            raise WorkflowExecutionError(
                f"checkpoint for run_id '{run_id}' points to unknown node "
                f"'{checkpoint.next_node}'"
            )

        async for state in self._run_from(
            checkpoint.next_node,
            checkpoint.state,
            checkpointer,
            run_id,
            steps=checkpoint.steps,
        ):
            yield state

    async def _run_from(
        self,
        start_node: str,
        state: StateT,
        checkpointer: Checkpointer[WorkflowCheckpoint[StateT]] | None,
        run_id: str | None,
        *,
        steps: int = 0,
    ) -> AsyncIterator[StateT]:
        current = start_node
        while True:
            steps += 1
            if steps > self._max_steps:
                raise WorkflowExecutionError(
                    f"workflow exceeded max_steps ({self._max_steps}); "
                    "a conditional edge may be looping without converging"
                )

            node_fn = self._nodes[current]
            with observe(WorkflowNode(node=current, step=steps)) as node:
                # Bound only around the node call, so the consumer of this
                # generator never sees it between two nodes.
                with bind_step(StepContext(run_id, current, steps)):
                    state = await node_fn(state)
                # Resolved inside the span so the trace shows where the run
                # went next, and which branch a router chose to send it there.
                next_node, route = self._next_node(current, state)
                node.next_node = next_node
                node.route = route

            if checkpointer is not None:
                assert run_id is not None
                await checkpointer.save(
                    run_id,
                    WorkflowCheckpoint(next_node=next_node, state=state, steps=steps),
                )

            yield state

            if next_node == END:
                return
            current = next_node
