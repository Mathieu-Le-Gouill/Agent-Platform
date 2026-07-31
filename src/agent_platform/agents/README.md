# Agents: Status & Roadmap

The agents layer is the top-level composition: LLM reasoning, tool calls, and conversation state.

```
Agent = LLM (reasoning) + Tools (capabilities) + Executor (loop control)
```

Each agent skill should be:
- **Self-contained**, clear input/output contract
- **Reusable**, composable into larger workflows
- **Language-driven**, LLM for planning and reasoning
- **Observable**, errors surfaced via `AgentError` hierarchy

## Implemented

### Tool Abstraction (`agents/tools/`)

Each tool lives in its own `tools/<name>/tool.py`, one directory per tool, matching the convention `integrations/<domain>/<provider>/` and `components/<name>/component.py` already use (see `components/README.md`'s "Directory Layout" for the rule). Shared plumbing (`base.py`, `registry.py`, `errors.py`, `safe_execution.py`) stays at the `tools/` root since it isn't a single tool.

| Component | File | What It Does |
|---|---|---|
| `Tool` Protocol | `tools/base.py` | `name`, `description`, `input_schema: type[BaseModel]`, `output_schema`, `async run(**kwargs) -> Any`; validates at subclass definition time |
| `ToolRegistry` | `tools/registry.py` | Register/get/remove/iterate; `resolve_call()` validates `call.arguments` against the tool's `input_schema` (raising `ToolCallValidationError`) before dispatching, then calls `tool.run(**call.arguments)` with the original arguments unchanged; a tool that leaves `input_schema` at its `Tool` protocol default (bare `BaseModel`, uninstantiable) opts out of this pre-validation; `call_and_wrap()` returns a `ToolMessage` (captures errors as `is_error=True`) |
| `TranscribeTool` | `tools/transcribe/tool.py` | Wraps `BaseSpeechToText`, audio → transcript |
| `SearchTool` | `tools/search/tool.py` | Embeds query → vector store search → ranked chunks |
| `OCRTool` | `tools/ocr/tool.py` | Wraps `BaseOCRProvider`, image → extracted text |
| `safe_call()` | `tools/safe_execution.py` | Error-wrapping helper for provider calls inside tools |

### Agent Runtime (`agents/`)

| Component | File | What It Does |
|---|---|---|
| `Agent` | `agent.py` | Holds `system_prompt`, `tool_registry`, `llm`; `think()` → `AssistantMessage`; `act()` → `list[ToolMessage]` (runs tool calls concurrently via `asyncio.gather`, order-preserving); `step()` = think + act |
| `AgentExecutor` | `executor.py` | think-act loop with `max_iterations` guard; `run(user_input)` and `run_with_messages(messages)` |
| `ConversationAgent` | `conversation.py` | Extends `Agent` with persistent `_history`, `chat()`, pluggable `context_strategy: ContextStrategy | None` for history trimming |
| `ContextStrategy` | `context.py` | Protocol: `trim(history) -> history`; `TurnCountStrategy(max_turns)` is the built-in implementation, `ConversationAgent(max_history_turns=N)` is sugar for `context_strategy=TurnCountStrategy(N)` (passing both raises `ValueError`) |

### Error Types (`agents/errors.py`)

```
AgentError
├── AgentThinkError    (LLM generation failed)
├── AgentActError      (tool execution failed at agent level)
└── AgentMaxIterations (loop exceeded max_iterations)
```

Tool-level errors live in `tools/errors.py`:
```
ToolError
├── ToolNotFoundError
├── ToolRegistrationError
└── ToolCallValidationError  (raised by ToolRegistry.resolve_call on schema mismatch)
```

## What's Left to Build

All planned tools (`OCRTool`, `SearchTool`, `TranscribeTool`, `TranslateTool`, `SummarizeTool`, `ClassifyTool`, `GenerateImageTool`) are implemented under `agents/tools/`.

## Roadmap: closing the gap with modern agent platforms

The provider/integration layer (13 domains, clean ABCs, DI) is close to parity
with libraries like Pydantic-AI or LiteLLM. The gap is above that layer: a
single fixed agent loop, no multi-agent orchestration, and no evals. This
section tracks that gap as a living roadmap, phased by leverage (each phase
should be usable on its own, not blocked on the next one). Update it in the
same change that closes or reshapes an item, same rule as the rest of this
README.

### Phase 1 — Evals (done)

A new top-level `evals/` package now provides `EvalCase`/`EvalDataset`/
`EvalRunner`/`Scorer`/`EvalReport`, an `agent-platform-eval` CLI entrypoint,
and a golden `ConversationAgent`/`AgentExecutor` tool-selection dataset,
wired into CI as an optional `evals` job. See `evals/README.md` for the
package's design and how to add scorers/targets/datasets.

### Phase 2 — Orchestration (`workflows/`, currently an empty stub)

The only agent shape today is one `ConversationAgent` per process. No
supervisor/sub-agent delegation, handoffs, fan-out/fan-in, or conditional
branching exists; `pipelines/` only gives fixed hand-written linear flows.

- Decide the engine before writing code: LangGraph (already implied by the
  stub's name) trades a new heavy dependency for speed-to-market; a lightweight
  in-house DAG executor (typed state schema, async node callables, conditional
  edges) stays consistent with this repo's "no monolithic dependency, inject
  everything" style but is more to build and maintain. This is a call for the
  user to make before Phase 2 starts.
- `workflows/state.py` (typed state base), `workflows/graph.py`
  (`WorkflowGraph`, node/edge registration, `arun()`), `workflows/nodes/`
  (`agent_node` wrapping an `Agent`, `tool_node`, conditional router).
- Multi-agent handoff: a node type that hands control to a different
  `Agent`/`ConversationAgent`, passing shared typed state.
- Pluggable checkpointing (in-memory default, injectable backend) so a
  workflow run can be interrupted and resumed, this also covers part of the
  Phase 3 "no persistence" gap.

### Phase 3 — Harness engineering hardening

Makes the existing single-agent loop production-grade; independent of Phase 2
and can proceed in parallel.

- ~~`Agent.act()` currently awaits tool calls sequentially~~ **done**: `act()`
  now runs independent tool calls concurrently via `asyncio.gather`
  (order-preserving).
- ~~`with_retry` retries any exception regardless of `PlatformError.retryable`~~
  **done**: `core/retry.py`'s `with_retry()` now re-raises immediately on a
  `PlatformError` with `retryable=False`. Note this is currently a no-op for
  every existing `integrations/` call site, since they all wrap `with_retry()`
  *inside* an outer `@error_logged(re_raise=ProviderError)`, so `with_retry`
  only ever sees the raw pre-translation SDK exception there; it activates
  wherever a future call site wraps something that raises `PlatformError`
  directly (a tool, the agent loop).
- Context management: `ConversationAgent` only truncates by turn count
  (`max_history_turns`). ~~Add token-aware trimming/summarization behind a
  pluggable strategy~~ **partially done**: the pluggable seam now exists
  (`ContextStrategy` protocol in `context.py`, `TurnCountStrategy` is the only
  built-in implementation so far). Still to add: a token-budget-aware
  strategy and a summarizing strategy (token counting is model-specific, so
  this can't be one fixed algorithm).
- Streaming LLM text generation: `Agent.think()` has no streaming variant
  today (only tool-call streaming via `act_stream()` exists); add one and wire
  it into `/chat` as SSE.
- ~~Malformed tool-call recovery~~ **prerequisite done**: `ToolRegistry.resolve_call`
  now validates `call.arguments` against `tool.input_schema` before invoking
  `run()`, raising `ToolCallValidationError` (a `ToolError` subclass) on
  mismatch instead of failing deep inside the tool. Still to add: have
  `AgentExecutor` catch that error and feed it back to the model as a
  corrective turn once, instead of surfacing it straight as a failed
  `ToolMessage`.

### Phase 4 — Production hygiene / observability polish

Lower urgency than Phases 1-3; these round out tracing and resilience once the
above land.

- Attach `conversation_id`/`session_id` as a span attribute in
  `core/tracing.py` so a multi-turn session's spans correlate without an
  external join.
- Rate limiter and circuit breaker utilities alongside `with_retry` in
  `core/errors.py`, opt-in per provider (today only retry+backoff exists, no
  fallback/failover chain across providers).
- Cost/token attribution rollup: aggregate `TokenUsage` per conversation,
  exposed via a hook or in `/chat` response metadata.
- Guardrails as an extension point (pre/post-generation validators for input
  moderation, output schema/PII checks) on `Agent`, not a baked-in policy.

## Infrastructure notes

- **Streaming**, `Agent.think()` does not yet support streaming responses
  (tool-call streaming via `act_stream()` already exists) — tracked as part
  of Phase 3 above.

## Adding a New Tool

1. Create `agents/tools/<name>.py`
2. Define an `input_schema` class (Pydantic `BaseModel`)
3. Implement the `Tool` protocol: set `name`, `description`, `input_schema`; implement `async run(**kwargs) -> Any`
4. Wrap provider calls with `safe_call()` from `safe_execution.py` so errors surface as `ToolError`
5. Register with `ToolRegistry` at construction time

```python
class MyTool:
    name = "my_tool"
    description = "Does something useful."
    input_schema = MyInput

    def __init__(self, backend: BaseMyProvider) -> None:
        self._backend = backend

    async def run(self, **kwargs: Any) -> Any:
        input = MyInput(**kwargs)
        return await safe_call(self._backend.do_thing, input.value)
```
