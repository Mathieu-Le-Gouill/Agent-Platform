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

Each tool lives in its own `tools/<name>/tool.py`, one directory per tool, matching the convention `integrations/<domain>/<provider>/` and `components/<stage>/<name>/component.py` already use (see `components/README.md`'s "Directory Layout" for the rule). Shared plumbing (`base.py`, `registry.py`, `errors.py`, `safe_execution.py`) stays at the `tools/` root since it isn't a single tool.

| Component | File | What It Does |
|---|---|---|
| `Tool` Protocol | `tools/base.py` | `name`, `description`, `input_schema: type[BaseModel]`, `output_schema`, `async run(**kwargs) -> Any`; validates at subclass definition time |
| `ToolRegistry` | `tools/registry.py` | Register/get/remove/iterate; `register(tool, *, timeout=None, max_concurrency=None)` optionally guards a tool with a per-call timeout (`run()` only, raises `ToolTimeoutError`) and/or a concurrency cap (`asyncio.Semaphore`, enforced on both `run()` and `astream()`), both opt-in and independent; `discover_entry_points(group="agent_platform.tools")` registers every `Tool` an installed package exposes via `importlib.metadata` entry points (a zero-arg factory per entry point); `resolve_call()` validates `call.arguments` against the tool's `input_schema` (raising `ToolCallValidationError`) before dispatching, then calls `tool.run(**call.arguments)` with the original arguments unchanged; a tool that leaves `input_schema` at its `Tool` protocol default (bare `BaseModel`, uninstantiable) opts out of this pre-validation; `call_and_wrap()` returns a `ToolMessage` (captures errors as `is_error=True`); its content is `""` for `None`, a `str` as is, and JSON for anything else (pydantic models dumped, every `bytes` value replaced by `<N bytes>`, `str()` for what JSON cannot encode); `ToolRegistry(max_result_chars=20_000)` cuts longer content and appends a truncation marker, `None` keeps results whole (`workflows/nodes/tool_node.py` does) |
| `TranscribeTool` | `tools/transcribe/tool.py` | Wraps `BaseSpeechToText`, audio → transcript |
| `SearchTool` | `tools/search/tool.py` | Embeds query → vector store search → ranked chunks |
| `OCRTool` | `tools/ocr/tool.py` | Wraps `BaseOCRProvider`, image → extracted text. `run()` only reads sources its `OCRSourcePolicy` allows (`allowed_dir` for files, resolved so `..` and symlinks cannot escape; `allowed_hosts` for `http`/`https` URLs); the default policy denies everything and a denied source raises `ToolSourceNotAllowedError` before the backend is called. `to_blocks()` is not checked: the application calls it, not the model |
| `MCPToolAdapter` / `discover_mcp_tools()` | `tools/mcp/adapter.py`, `tools/mcp/discovery.py` | Wraps one MCP server tool (`core/interfaces/mcp/base.py`'s `BaseMCPClient`) as a `Tool`, building `input_schema` dynamically from the tool's raw JSON Schema; `discover_mcp_tools(client)` lists every tool a connected client's server exposes and returns one adapter per tool, ready for `ToolRegistry.register()`. No MCP-transport code lives here, see `integrations/mcp/` |
| `safe_call()` | `tools/safe_execution.py` | Error-wrapping helper for provider calls inside tools |

### Agent Runtime (`agents/`)

| Component | File | What It Does |
|---|---|---|
| `Agent` | `agent.py` | Holds `system_prompt`, `tool_registry`, `llm`, optional `guardrails: list[Guardrail]` and `response_schema: type[BaseModel]`; `think()` → `AssistantMessage` (runs guardrails around generation, then validates `response_schema` with one corrective retry via `validation.retry_once_on_invalid` when there are no tool calls); `think_stream()` → `AsyncIterator[StreamChunk]` (same tool-list/config setup as `think()`, raw: no guardrail/schema wrapping); `think_from_stream()` → `(AssistantMessage, list[str])` (consumes `think_stream()` under the same guardrails and `response_schema` retry as `think()`, and returns the deltas of the accepted attempt, or the final text alone when a guardrail short-circuited or rewrote it); `act()` → `list[ToolMessage]` (runs tool calls concurrently via `asyncio.gather`, order-preserving); `act_stream()` propagates a tool call's validation-error tag onto the `ToolMessage` it reconstructs from stream chunks; `step()` = think + act |
| `AgentExecutor` | `executor.py` | think-act loop with `max_iterations` guard; `run(user_input)` and `run_with_messages(messages)`; each round is wrapped in `validation.retry_once_on_invalid` so a `ToolCallValidationError` (tagged via `tools.registry.is_tool_validation_error`) triggers one corrective retry before giving up; `run_streaming(user_input)` calls `Agent.think_from_stream()` (so guardrails and `response_schema` apply as on `run()`) and yields its text deltas interleaved with `act_stream()` tool events; each attempt of the corrective retry buffers its own events and only the kept attempt's are yielded. Fragmented `ToolCallDelta`s are reassembled into `ToolCall`s by `_assemble_tool_calls()` in `agent.py` |
| `ConversationAgent` | `conversation.py` | Extends `Agent` with persistent `_history`, `chat()`, pluggable `context_strategy: ContextStrategy | None` for history trimming, optional `checkpointer: Checkpointer[list[Message]] | None` (saves history after every `chat()` turn); `ConversationAgent.resume(conversation_id, checkpointer, ...)` classmethod reconstructs an agent's history from a checkpointer |
| `ContextStrategy` | `context.py` | Protocol: `trim(history) -> history`. `TurnCountStrategy(max_turns)` is the original built-in implementation, `ConversationAgent(max_history_turns=N)` is sugar for `context_strategy=TurnCountStrategy(N)` (passing both raises `ValueError`). `TokenBudgetStrategy(max_tokens, count_tokens)` drops oldest whole turns until under budget, given an injected token-counting callable (always keeps at least the most recent turn). `SummarizingStrategy(llm, max_turns=...)` behaves like `TurnCountStrategy` but folds dropped turns into a `SystemMessage` summary via the injected `BaseLLMProvider`'s synchronous `generate()` (not `agenerate()`), so `trim()` stays synchronous and the `ContextStrategy` protocol/`ConversationAgent` need no changes. |
| `Guardrail` | `guardrails.py` | `Guardrail = Middleware[GuardrailContext]` (Phase 1's `core/middleware.py` primitive, parameterized over `GuardrailContext(messages)`); `Agent(guardrails=[...])` runs them through a `MiddlewarePipeline` wrapping generation (`before` can short-circuit with a canned `AssistantMessage`, `after` can inspect/reject the real one). `OutputNotEmptyGuardrail` is the shipped example: raises `AgentGuardrailError` if the final answer has no text and no tool calls. Policy is opt-in, not baked in. Guardrails also run on `run_streaming()` (through `think_from_stream()`), so a blocked answer raises before any text is yielded. |
| `retry_once_on_invalid` | `validation.py` | Shared "run, check, corrective-retry-once, else give up" helper (`attempt`/`check`/`correct` callables) used by both malformed tool-call recovery (`AgentExecutor`) and `response_schema` validation (`Agent.think()`); raises `AgentRecoveryExhausted` (carrying the failing `last_result`) if the retried attempt is still invalid. |

### Error Types (`agents/errors.py`)

```
AgentError
├── AgentThinkError        (LLM generation failed)
├── AgentActError          (tool execution failed at agent level)
├── AgentMaxIterations     (loop exceeded max_iterations)
├── AgentRecoveryExhausted (one corrective retry still failed; carries `last_result`)
└── AgentGuardrailError    (raised by a concrete Guardrail, e.g. OutputNotEmptyGuardrail)
```

Tool-level errors live in `tools/errors.py`:
```
ToolError
├── ToolNotFoundError
├── ToolRegistrationError
├── ToolCallValidationError  (raised by ToolRegistry.resolve_call on schema mismatch;
│                             tagged onto the resulting ToolMessage's metadata, see
│                             tools.registry.is_tool_validation_error())
└── ToolTimeoutError         (raised by ToolRegistry.resolve_call when a tool's
                              per-call timeout, set via register(timeout=...), elapses)
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

### Phase 2 — Orchestration (`workflows/`) (done)

The only agent shape before this phase was one `ConversationAgent` per
process; `pipelines/` only gave fixed hand-written linear flows. `workflows/`
now provides a typed-state DAG: `WorkflowGraph`/`CompiledWorkflow`
(`graph.py`), `WorkflowState`/`MessagesState`/`HandoffState`/
`WorkflowCheckpoint` (`state.py`), and `agent_node`/`tool_node`/
`binary_router`/`handoff_node` (`nodes/`). See `workflows/README.md` for the
full design.

- Engine decision: an in-house DAG executor, not LangGraph, consistent with
  every other layer's "dependency-free ABCs, vendor SDKs isolated to
  `integrations/`" pattern (the unused `langgraph`/`langchain` base
  dependencies this decision left dangling have been removed from
  `pyproject.toml`).
- Multi-agent handoff: `nodes/handoff.py::handoff_node` hands control to a
  different `Agent`, sharing `state.messages`, and stamps `state.active_agent`.
- Pluggable checkpointing: `CompiledWorkflow.arun(checkpointer=..., run_id=...)`
  saves a `WorkflowCheckpoint` (state + next node) after every step;
  `resume(run_id, checkpointer)` continues an interrupted run. Built on Phase
  1's `Checkpointer[StateT]`, the same primitive `ConversationAgent` uses.

### Phase 3 — Harness engineering hardening (done)

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
- ~~Context management: `ConversationAgent` only truncates by turn count~~
  **done**: `TokenBudgetStrategy` (token-budget-aware trimming, injected
  token counter) and `SummarizingStrategy` (folds dropped turns into an
  LLM-generated summary via `BaseLLMProvider.generate()`) join
  `TurnCountStrategy` in `context.py`, all implementing `ContextStrategy`.
- ~~Streaming LLM text generation: `Agent.think()` has no streaming variant~~
  **done**: `Agent.think_stream()` wraps `BaseLLMProvider.stream()` the same
  way `think()` wraps `agenerate()`; `AgentExecutor.run_streaming()` uses
  `Agent.think_from_stream()` on top of it, so the assistant's own text is
  emitted (not just tool-call deltas) after the guardrails and the
  `response_schema` check, one model call at a time rather than token by token
  while the model is still answering, and `api/app.py` exposes it as SSE via
  `POST /chat/stream`.
- ~~Malformed tool-call recovery~~ **done**: `AgentExecutor` catches a
  `ToolCallValidationError`-tagged `ToolMessage` (via
  `tools.registry.is_tool_validation_error`) and retries `think()`+`act()`
  once with corrective feedback appended to the conversation, via the shared
  `validation.retry_once_on_invalid` helper, before giving up and returning
  the error like any other failed tool call. `run_streaming()` applies the
  same recovery.
- New this phase, not originally scoped but built on the same primitives:
  `response_schema` on `Agent` (structured final-answer validation, reusing
  `retry_once_on_invalid`), `Guardrail`s (`guardrails.py`, built on Phase 1's
  `MiddlewarePipeline`), and `ConversationAgent` checkpointing/`resume()`
  (built on Phase 1's `Checkpointer[StateT]`). These pull forward two Phase 4
  items below (guardrails, part of persistence); Phase 4 is updated to match.

### Phase 3b - Tooling ecosystem (done)

Independent of Phases 2/4: closes the "every tool is hand-registered Python"
gap without changing the `Tool` protocol itself, so anything already
implementing `Tool` (including a future `workflows/` node) keeps working
unmodified.

- Per-tool resilience: `ToolRegistry.register(tool, *, timeout=None,
  max_concurrency=None)`, enforced in `call_and_wrap`/`call_and_stream`
  (`asyncio.wait_for` for `timeout`, an `asyncio.Semaphore` for
  `max_concurrency`), both opt-in, reusing `core/resilience.py`'s spirit
  (guarding a call site's traffic) without pulling `CircuitBreaker`/
  `RateLimiter` themselves in, a single slow/runaway tool is a simpler
  failure mode than the sustained-traffic one those guard against.
- Generic plugin discovery: `ToolRegistry.discover_entry_points(group=
  "agent_platform.tools")` loads `Tool` factories any installed package
  exposes via `importlib.metadata` entry points, no registry code change
  needed to pick up a new third-party tool package.
- MCP interop, following the existing provider-inversion pattern (a tool
  wraps a `core/interfaces/<domain>` ABC, never a concrete `integrations/`
  class directly, see `TranscribeTool`/`BaseSpeechToText`):
  - `core/schemas/mcp.py::MCPToolSpec` (name/description/raw JSON Schema)
    and `core/interfaces/mcp/base.py::BaseMCPClient` (the only domain ABC
    that owns connection lifecycle, `connect()`/`aclose()`, since an MCP
    server is a stateful session, not a stateless per-call provider).
  - `integrations/mcp/stdio/provider.py::StdioMCPClient` is the only file
    that imports the third-party `mcp` SDK (optional `tools-mcp` extra),
    per `integrations/README.md`'s "only integrations/ imports third-party
    packages" rule.
  - `agents/tools/mcp/adapter.py::MCPToolAdapter` and
    `agents/tools/mcp/discovery.py::discover_mcp_tools()` depend only on
    `BaseMCPClient`, so they import fine even without the `mcp` SDK
    installed; only constructing a concrete `StdioMCPClient` needs the extra.
  - Settings-driven wiring: `Settings.mcp_stdio_servers` (`{name: [command,
    *args]}`, the same shape as Claude Desktop/VS Code's `mcpServers` config)
    plus `config/container.py::build_agent_async()` - a separate async
    entrypoint from the sync `build_agent()`, since connecting to an MCP
    server is an async handshake and every other provider `build_agent`
    resolves is synchronous. `api/app.py`'s lifespan calls it and closes
    each returned client on shutdown; non-API callers (evals, scripts, most
    tests) keep using the plain sync `build_agent()` and simply get no MCP
    tools unless they opt into the async variant.

### Phase 4 — Production hygiene / observability polish (done)

Lower urgency than Phases 1-3; these round out tracing and resilience once the
above landed.

- ~~Attach `conversation_id`/`session_id` as a span attribute~~ **done**:
  `AgentExecutor.run`/`run_with_messages`/`run_streaming` take an optional
  `conversation_id` keyword and set it on the `AgentRun` observation when
  given (omitted, not set to `None`, otherwise);
  `ConversationAgent.chat()` and `api/app.py`'s `/chat`, `/chat/stream` routes
  always pass theirs, so a multi-turn session's spans correlate without an
  external join.
- ~~Rate limiter and circuit breaker utilities wired into an LLM call site~~
  **done, for `CircuitBreaker`**: `core/interfaces/llm/fallback.py`'s
  `FallbackLLMProvider` tries an ordered list of `BaseLLMProvider`s, each
  behind its own `CircuitBreaker`, falling through to the next on failure - a
  `BaseLLMProvider` itself, so `Agent(llm=...)` doesn't need to know it's
  talking to more than one vendor. `config/container.py::build_agent` wires
  it in when `Settings.fallback_llm_models` is non-empty; empty (the default)
  skips the wrapper entirely. `RateLimiter` remains unwired, no call site
  needed it yet.
- ~~Cost/token attribution rollup~~ **done**: `Agent` takes an optional
  `token_usage_aggregator`/`usage_key`, records `TokenUsage` after every
  `_generate()` call and every `think_stream()` chunk that carries one, and
  exposes the running total via `Agent.token_usage`. `ConversationAgent` uses
  its own `conversation_id` as the key; `config/container.py::build_agent`
  wires one in by default, and `api/app.py`'s `/chat` route returns it as
  `ChatResponse.usage`.
- ~~Guardrails as an extension point~~ **done**: see Phase 3 above
  (`agents/guardrails.py`).
- ~~`ConversationAgent` session persistence~~ **done**: see Phase 3 above
  (`checkpointer` param, `ConversationAgent.resume()`).

## Infrastructure notes

- **Observability**, the agent layer emits three kinds of span, all through
  `core/observability`'s typed observations (no backend name anywhere
  here, and a cheap no-op when no backend is installed):
  - `invoke_agent` (`executor.py`) wraps the whole think-act loop. `_LoopTrace`
    collects what makes a loop pathological rather than merely slow:
    `agent.iterations` against `.max_iterations`, the
    `.exit_reason` (`stop`, `max_iterations`, `recovery_exhausted`, or the
    exception's type name), whether tool-call recovery fired
    (`.recovery_attempted`, notified by `validation.py`'s `on_retry` callback),
    and how many tool calls the model requested versus how many actually ran.
  - `execute_tool` (`tools/registry.py`), a `ToolRun` recording the tool's
    name, call id, kind and source, which is what makes an MCP tool
    distinguishable from a local one in a trace (the MCP adapter declares
    `ToolKind.REMOTE`/`mcp`, a plain `Tool` defaults to `ToolKind.FUNCTION`).
    Arguments and results contribute their size and a short digest always, and
    their content only under the opt-in content-capture switch (see
    `core/README.md`); streaming tools also report their chunk count.
  - `guardrail` (`guardrails.py`), a `GuardrailCheck` per `before`/`after`
    stage, via `TracedGuardrail`. `Agent` wraps every guardrail it is given, so any
    third-party object satisfying the `Guardrail` protocol gets a span
    without implementing anything extra, and a blocked request stops being an
    action with no trace evidence.

  Cost is attached in `agent.py::_record_usage`: the estimator prices each
  `TokenUsage` and `emit(ModelCost(...))` stamps it onto the enclosing span
  (the provider's own `chat` span is already closed by then). When several
  model calls share one `invoke_agent` span the attribute reports the latest.
- **Streaming**, `Agent.think_from_stream()` and `AgentExecutor.run_streaming()`
  emit the assistant's own text, not just tool-call deltas, once per accepted
  model call (guardrails and the schema check run first); `/chat/stream`
  exposes this over SSE. Note `/chat/stream` is single-turn (built directly on
  `AgentExecutor`, not `ConversationAgent`): it does not read from or append
  to a `ConversationAgent`'s persistent history the way `/chat` does.

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
