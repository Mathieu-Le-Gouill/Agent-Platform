# Agents: Single-Agent Composition

The agents layer is single-agent composition: LLM reasoning, tool calls, and conversation state. Running several agents together is `workflows/`, the layer above.

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
| `AgentExecutor` | `executor.py` | think-act loop with `max_iterations` guard; `run(user_input)` and `run_with_messages(messages)`; each round is wrapped in `core.retry.retry_once_on_invalid` so a `ToolCallValidationError` (tagged via `tools.registry.is_tool_validation_error`) triggers one corrective retry before giving up; `run_streaming(user_input)` calls `Agent.think_from_stream()` (so guardrails and `response_schema` apply as on `run()`) and yields its text deltas interleaved with `act_stream()` tool events; each attempt of the corrective retry buffers its own events and only the kept attempt's are yielded. Fragmented `ToolCallDelta`s are reassembled into `ToolCall`s by `_assemble_tool_calls()` in `agent.py` |
| `ConversationAgent` | `conversation.py` | Extends `Agent` with persistent `_history`, `chat()`, pluggable `context_strategy: ContextStrategy | None` for history trimming, optional `checkpointer: Checkpointer[list[Message]] | None` (saves history after every `chat()` turn); `ConversationAgent.resume(conversation_id, checkpointer, ...)` classmethod reconstructs an agent's history from a checkpointer |
| `ContextStrategy` | `context.py` | Protocol: `trim(history) -> history`. `TurnCountStrategy(max_turns)` is the original built-in implementation, `ConversationAgent(max_history_turns=N)` is sugar for `context_strategy=TurnCountStrategy(N)` (passing both raises `ValueError`). `TokenBudgetStrategy(max_tokens, count_tokens)` drops oldest whole turns until under budget, given an injected token-counting callable (always keeps at least the most recent turn). `SummarizingStrategy(llm, max_turns=...)` behaves like `TurnCountStrategy` but folds dropped turns into a `SystemMessage` summary via the injected `BaseLLMProvider`'s synchronous `generate()` (not `agenerate()`), so `trim()` stays synchronous and the `ContextStrategy` protocol/`ConversationAgent` need no changes. |
| `Guardrail` | `guardrails.py` | `Guardrail = Middleware[GuardrailContext]` (the `core/middleware.py` primitive, parameterized over `GuardrailContext(messages)`); `Agent(guardrails=[...])` runs them through a `MiddlewarePipeline` wrapping generation (`before` can short-circuit with a canned `AssistantMessage`, `after` can inspect/reject the real one). `OutputNotEmptyGuardrail` is the shipped example: raises `AgentGuardrailError` if the final answer has no text and no tool calls. Policy is opt-in, not baked in. Guardrails also run on `run_streaming()` (through `think_from_stream()`), so a blocked answer raises before any text is yielded. |
| `retry_once_on_invalid` | `core/retry.py` | Shared "run, check, corrective-retry-once, else give up" helper (`attempt`/`check`/`correct` callables) used by malformed tool-call recovery (`AgentExecutor`), `response_schema` validation (`Agent.think()`) and `core/interfaces/llm/structured.py`; raises `RecoveryExhausted` (`core/errors.py`, carrying the failing `last_result`) if the retried attempt is still invalid. |

`Agent` builds its `response_schema` config, validation and correction message with `schema_config`, `schema_error` and `schema_correction` from `core/interfaces/llm/structured.py`. Its `model` defaults to `""`: the provider's bound model then applies.

### Error Types (`agents/errors.py`)

```
AgentError
├── AgentThinkError        (LLM call failed; carries the provider error's code/retryable/retry_after, or code "llm_error")
├── AgentActError          (tool execution failed at agent level)
├── AgentMaxIterations     (loop exceeded max_iterations; code "agent_max_iterations")
├── AgentRecoveryExhausted (the corrective retry of a tool call or a response schema still failed; code "agent_recovery_exhausted"; `last_message` holds the invalid answer. Not a `core.errors.RecoveryExhausted`, so the executor's own `except RecoveryExhausted` never catches it)
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

## Design Decisions

- **Tools wrap a `core/interfaces` ABC, never a concrete `integrations/` class** (`TranscribeTool` wraps `BaseSpeechToText`). MCP follows suit: `tools/mcp/` depends only on `BaseMCPClient`, so it imports without the `tools-mcp` extra; only `integrations/mcp/stdio/` imports the `mcp` SDK.
- **MCP wiring is async.** Servers come from `Settings.mcp_stdio_servers` (`{name: [command, *args]}`). Connecting is an async handshake, so `config/container.py::build_agent_async()` sits beside the sync `build_agent()`: `api/app.py`'s lifespan calls it and closes each client on shutdown; other callers (evals, scripts, tests) use `build_agent()` and get no MCP tools.
- **Per-tool `timeout`/`max_concurrency`, not `CircuitBreaker`/`RateLimiter`.** One slow or runaway tool is a per-call failure; those two guard a call site's sustained traffic. `RateLimiter` has no call site yet.
- **LLM fallback is opt-in:** `build_agent` wraps the LLM in `FallbackLLMProvider` (see `core/README.md`) only when `Settings.fallback_llm_models` is non-empty.
- **Token usage:** `ConversationAgent.with_conversation()` gives each fork its own `TokenUsageAggregator` (see `core/README.md`); `/chat` returns the conversation's total as `ChatResponse.usage`.

## Infrastructure notes

- **Observability**, the agent layer emits three kinds of span, all through
  `core/observability`'s typed observations (no backend name anywhere
  here, and a cheap no-op when no backend is installed):
  - `invoke_agent` (`executor.py`) wraps the whole think-act loop. `_LoopTrace`
    collects what makes a loop pathological rather than merely slow:
    `agent.iterations` against `.max_iterations`, the
    `.exit_reason` (`stop`, `max_iterations`, `recovery_exhausted`, or the
    exception's type name), whether tool-call recovery fired
    (`.recovery_attempted`, notified by `retry_once_on_invalid`'s `on_retry` callback),
    and how many tool calls the model requested versus how many actually ran.
  - `execute_tool` (`tools/registry.py`), a `ToolRun` recording the tool's
    name, call id, kind and source, which is what makes an MCP tool
    distinguishable from a local one in a trace (the MCP adapter declares
    `ToolKind.REMOTE`/`mcp`, a plain `Tool` defaults to `ToolKind.FUNCTION`).
    Arguments and results contribute their size always, and a short digest
    and their content only under the opt-in content-capture switch (see
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

  `AgentExecutor.run`/`run_with_messages`/`run_streaming` take an optional
  `conversation_id` and set it on the `AgentRun` observation;
  `ConversationAgent.chat()` and both API chat routes always pass one, so a
  session's spans correlate without an external join.

## Idempotent Tools and Partial History

A tool that has a side effect (sends an email, writes a record, starts a billed job) must be idempotent on `current_tool_call().idempotency_key`. `ToolRegistry` sets `ToolCallContext(tool_name, idempotency_key)` (`tools/call_context.py`, exported from `agents.tools`) around `tool.run` and around each advance of `tool.astream`; `current_tool_call()` returns it, or `None` outside a registry call.

```python
async def run(self, **kwargs: Any) -> Any:
    ctx = current_tool_call()
    key = ctx.idempotency_key if ctx else None
    return await safe_call(self._backend.send(MyInput(**kwargs), idempotency_key=key))
```

- Inside an agent the key is the model's call id. In the corrective retry the model issues new ids, so: a call re-emitted unchanged (same name, same canonical-JSON arguments) after it failed runs again under the failed call's key, so a tool can recognise a side effect it may have started; a call that succeeded is not run again (see below); a changed or new call gets its own id. Each earlier key is used once. If the provider reuses the failed call's id for another call in the retry, the re-emitted call runs under its own id instead, so no two calls share a key. Across a workflow resume of an `agent_node` keys still change: the model is asked again and issues new call ids.
- Inside a workflow `tool_node` the key is `run_id:node:step[:branch]:n` and survives `resume` (see `workflows/README.md`). `run_id` must be unique per run, or two runs share keys.
- Corrective retry: when a malformed tool call triggers the one corrective retry, a retried call with the same name and canonical-JSON arguments as a call that succeeded (`is_error=False`) in the failed attempt reuses that result under its new call id instead of running again. Each stored result is reused at most once; tool messages keep the assistant message's call order; `run_streaming` emits one final chunk for a reused call.
- Partial history: `AgentExecutor.run_in_place(messages, *, conversation_id=None)` appends the turn's messages to the given list as they happen (`run_with_messages` still works on a copy). When `ConversationAgent.chat` fails, history keeps the completed tool calls and their results, drops a trailing assistant message whose tool calls have no results, saves that history to the checkpointer (a save failure is logged, not raised) and re-raises the original error. On cancellation it only drops the dangling message and re-raises. A caller that resent the same message after a failed `chat` and relied on a clean history now sees the earlier user message and tool results; the next user message follows a tool message.
- `safe_call`/`safe_stream` set `ToolError.retryable` from a `PlatformError` cause, else `False`.

## Adding a New Tool

1. Create `agents/tools/<name>/tool.py`
2. Define an `input_schema` class (Pydantic `BaseModel`)
3. Implement the `Tool` protocol: set `name`, `description`, `input_schema`; implement `async run(**kwargs) -> Any`
4. Wrap provider calls with `safe_call()` from `safe_execution.py` so errors surface as `ToolError`
5. Register with `ToolRegistry` at construction time
6. If the tool has a side effect, make it idempotent on `current_tool_call().idempotency_key` (see above)

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
