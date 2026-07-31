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
| `Agent` | `agent.py` | Holds `system_prompt`, `tool_registry`, `llm`, optional `guardrails: list[Guardrail]` and `response_schema: type[BaseModel]`; `think()` → `AssistantMessage` (runs guardrails around generation, then validates `response_schema` with one corrective retry via `validation.retry_once_on_invalid` when there are no tool calls); `think_stream()` → `AsyncIterator[StreamChunk]` (same tool-list/config setup as `think()`, no guardrail/schema wrapping); `act()` → `list[ToolMessage]` (runs tool calls concurrently via `asyncio.gather`, order-preserving); `act_stream()` propagates a tool call's validation-error tag onto the `ToolMessage` it reconstructs from stream chunks; `step()` = think + act |
| `AgentExecutor` | `executor.py` | think-act loop with `max_iterations` guard; `run(user_input)` and `run_with_messages(messages)`; each round is wrapped in `validation.retry_once_on_invalid` so a `ToolCallValidationError` (tagged via `tools.registry.is_tool_validation_error`) triggers one corrective retry before giving up; `run_streaming(user_input)` streams `Agent.think_stream()` text chunks interleaved with `act_stream()` tool events, reassembling fragmented `ToolCallDelta`s into `ToolCall`s via the internal `_assemble_tool_calls()` |
| `ConversationAgent` | `conversation.py` | Extends `Agent` with persistent `_history`, `chat()`, pluggable `context_strategy: ContextStrategy | None` for history trimming, optional `checkpointer: Checkpointer[list[Message]] | None` (saves history after every `chat()` turn); `ConversationAgent.resume(conversation_id, checkpointer, ...)` classmethod reconstructs an agent's history from a checkpointer |
| `ContextStrategy` | `context.py` | Protocol: `trim(history) -> history`. `TurnCountStrategy(max_turns)` is the original built-in implementation, `ConversationAgent(max_history_turns=N)` is sugar for `context_strategy=TurnCountStrategy(N)` (passing both raises `ValueError`). `TokenBudgetStrategy(max_tokens, count_tokens)` drops oldest whole turns until under budget, given an injected token-counting callable (always keeps at least the most recent turn). `SummarizingStrategy(llm, max_turns=...)` behaves like `TurnCountStrategy` but folds dropped turns into a `SystemMessage` summary via the injected `BaseLLMProvider`'s synchronous `generate()` (not `agenerate()`), so `trim()` stays synchronous and the `ContextStrategy` protocol/`ConversationAgent` need no changes. |
| `Guardrail` | `guardrails.py` | `Guardrail = Middleware[GuardrailContext]` (Phase 1's `core/middleware.py` primitive, parameterized over `GuardrailContext(messages)`); `Agent(guardrails=[...])` runs them through a `MiddlewarePipeline` wrapping generation (`before` can short-circuit with a canned `AssistantMessage`, `after` can inspect/reject the real one). `OutputNotEmptyGuardrail` is the shipped example: raises `AgentGuardrailError` if the final answer has no text and no tool calls. Policy is opt-in, not baked in. |
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
└── ToolCallValidationError  (raised by ToolRegistry.resolve_call on schema mismatch;
                              tagged onto the resulting ToolMessage's metadata, see
                              tools.registry.is_tool_validation_error())
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
  way `think()` wraps `agenerate()`; `AgentExecutor.run_streaming()` uses it
  so the assistant's own text streams token-by-token (not just tool-call
  deltas), and `api/app.py` exposes it as SSE via `POST /chat/stream`.
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

### Phase 4 — Production hygiene / observability polish

Lower urgency than Phases 1-3; these round out tracing and resilience once the
above land.

- Attach `conversation_id`/`session_id` as a span attribute in
  `core/tracing.py` so a multi-turn session's spans correlate without an
  external join.
- Rate limiter and circuit breaker utilities alongside `with_retry`: `core/resilience.py`'s
  `CircuitBreaker`/`RateLimiter` exist (Phase 1) but are not yet wired into
  any LLM provider call site, opt-in per provider (today only retry+backoff
  is wired, no fallback/failover chain across providers).
- Cost/token attribution rollup: `core/token_usage.py`'s `TokenUsageAggregator`
  exists (Phase 1) but nothing yet calls `record()` per conversation or
  exposes totals via a hook or in `/chat` response metadata.
- ~~Guardrails as an extension point~~ **done**: see Phase 3 above
  (`agents/guardrails.py`).
- ~~`ConversationAgent` session persistence~~ **done**: see Phase 3 above
  (`checkpointer` param, `ConversationAgent.resume()`).

## Infrastructure notes

- **Streaming**, `Agent.think_stream()` and `AgentExecutor.run_streaming()`
  now stream the assistant's own text, not just tool-call deltas; `/chat/stream`
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
