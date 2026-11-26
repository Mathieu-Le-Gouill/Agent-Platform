# Workflows: Multi-Agent Orchestration

## Design

`workflows/` sits above `agents/` in the layer stack: `core` → `integrations`
→ `components` → `pipelines` → `agents` → `workflows` → `evals` (see
`agent_platform/README.md`'s layer map). It answers what none of the layers
below it can: running more than one `Agent`/`Tool` as a single typed-state
graph, with conditional branching, handoffs, and resumability, instead of one
`ConversationAgent` per process or a hand-written linear `pipelines/` flow.

The engine is an in-house DAG executor, not LangGraph: this repo's providers,
tools, and agent loop are all built as dependency-free ABCs/primitives with
vendor SDKs isolated to `integrations/`, and pulling in a framework-specific
graph abstraction here would break that pattern for the one layer meant to
compose everything below it. A node is just `async def(state) -> state`, so
anything already shaped that way composes without an adapter beyond what
`nodes/` provides.

## Directory Layout

```
workflows/
├── errors.py   # WorkflowError, WorkflowValidationError, WorkflowExecutionError
├── state.py    # WorkflowState (base), MessagesState, HandoffState, WorkflowCheckpoint
├── graph.py    # WorkflowGraph (builder), CompiledWorkflow (runtime), END sentinel
└── nodes/
    ├── agent_node.py  # agent_node(agent) -> NodeFn, adapts an Agent's think-act loop
    ├── tool_node.py   # tool_node(tool, input_fn, output_fn) -> NodeFn, deterministic step
    ├── router.py      # binary_router(predicate) -> Router, yes/no conditional edges
    └── handoff.py     # handoff_node(agent) -> NodeFn, multi-agent delegation
```

## Implemented

| Component | File | What It Does |
|---|---|---|
| `WorkflowState` | `state.py` | Base Pydantic model for a workflow's typed state; state flows through the graph by value (`model_copy(update={...})` per node) |
| `MessagesState` | `state.py` | `WorkflowState` + `messages: list[Message]`, the shape `agent_node`/`handoff_node` require |
| `HandoffState` | `state.py` | `MessagesState` + `active_agent: str`, stamped by `handoff_node` after each turn |
| `WorkflowCheckpoint[StateT]` | `state.py` | What gets persisted per step: `next_node` + `state` + `steps` (nodes executed so far, default `0` so older checkpoints still load), so a resumed run knows where to continue, not just what was last computed, and `max_steps` keeps counting across resumes |
| `WorkflowGraph[StateT]` | `graph.py` | Builder: `add_node`, `set_entry_point`, `add_edge` (unconditional), `add_conditional_edges(node, router, path_map)`. `compile(max_steps=100)` validates the whole graph once (unregistered entry point/edge endpoints, an unreachable node, a node with no outgoing edge) instead of failing mid-run, and returns a `CompiledWorkflow`; `max_steps` bounds a run the same way `AgentExecutor.max_iterations` bounds the think-act loop, so a conditional edge that never converges raises `WorkflowExecutionError` instead of looping forever |
| `CompiledWorkflow[StateT]` | `graph.py` | `arun(initial_state, *, checkpointer=None, run_id=None)` runs from the entry point, yielding the state after every node (streamable, observable progress, not just the final result); each node executes inside an `observe(WorkflowNode(...))` carrying the node name and step, plus the `next_node` it resolved and (on a conditional edge) the `route` a router returned, so a trace shows where a run went next and why. `run_id` must be unique per run: it seeds the idempotency key of every `tool_node` call. Each node call runs in a step context `(run_id, node, step)` with its own call counter; inside `fan_out` each branch gets its own context with the branch name appended (nested branches joined by `:`), and branch names are not checked for uniqueness. `resume(run_id, checkpointer)` reloads the last `WorkflowCheckpoint` and continues from the node it was about to enter, counting steps from `checkpoint.steps`; resuming a finished run (`next_node == END`) yields nothing, and a checkpoint naming a node the graph lacks raises `WorkflowExecutionError` |
| `agent_node(agent, *, max_iterations=10)` | `nodes/agent_node.py` | Runs `agent`'s think-act loop (via a private `AgentExecutor`) over `state.messages`, returns updated state. Pure adapter: a hardened `Agent` (guardrails, `response_schema`, tool-call retry) behaves identically driven directly or through a node |
| `tool_node(tool, *, input_fn, output_fn)` | `nodes/tool_node.py` | Deterministic (no LLM) step: `input_fn` extracts the tool's kwargs from state, `output_fn` folds the `ToolMessage` back in. Runs through a private single-tool `ToolRegistry` so argument validation/timeout/error-wrapping match `Agent.act()`, rather than calling `tool.run()` directly. The call id, which the tool reads as `current_tool_call().idempotency_key`, is `run_id:node:step[:branch]:n` (`n` counts the node run's calls), the same on `resume`, so a tool can skip work a crashed run already did; without a `run_id` it is a random `uuid4`. A tool the model calls inside an `agent_node` gets no such stable key: the model's call ids change when a resumed node asks again |
| `binary_router(predicate)` | `nodes/router.py` | Wraps a boolean predicate into a `Router` for the common yes/no conditional edge (`TRUE`/`FALSE` path-map keys) |
| `handoff_node(agent, *, max_iterations=10)` | `nodes/handoff.py` | Built directly on `agent_node`: hands control to a different agent, sharing `state.messages`, and stamps `state.active_agent` afterward, plus `workflow.handoff.to` onto the enclosing node span so a trace names the receiving agent |

### Error Types (`workflows/errors.py`)

```
WorkflowError
├── WorkflowValidationError  (raised by WorkflowGraph.compile() on a malformed graph)
└── WorkflowExecutionError   (raised at run time: unresolved router key,
                               checkpointer used without run_id, missing checkpoint on resume)
```

## Building a Workflow

```python
from agent_platform.workflows import END, MessagesState, WorkflowGraph
from agent_platform.workflows.nodes import agent_node, binary_router

class TriageState(MessagesState):
    escalate: bool = False

graph: WorkflowGraph[TriageState] = WorkflowGraph(TriageState)
graph.add_node("first_line", agent_node(first_line_agent))
graph.add_node("specialist", agent_node(specialist_agent))
graph.set_entry_point("first_line")
graph.add_conditional_edges(
    "first_line",
    binary_router(lambda s: s.escalate),
    {"true": "specialist", "false": END},
)
graph.add_edge("specialist", END)

workflow = graph.compile()
async for state in workflow.arun(TriageState(messages=[UserMessage(content="...")])):
    ...  # observe/stream intermediate progress
```

Add resumability by passing `checkpointer=InMemoryCheckpointer()` (or any
`Checkpointer[WorkflowCheckpoint[TriageState]]`, see `core/persistence.py`)
and a `run_id` to `arun()`; an interrupted run continues via
`workflow.resume(run_id, checkpointer)`.

## What's Left to Build

- No parallel fan-out/fan-in (running multiple nodes concurrently from one
  state and merging results back); today's graph is a strict DAG walked one
  node at a time.
