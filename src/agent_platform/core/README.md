# Core: Library-Independent Foundation

## Design

`core/` is the **near-zero-dependency foundation** of the platform. It defines everything that does not depend on an external library, pure Python ABCs, Pydantic schemas, enums, and error types. That now includes observability: `interfaces/tracing/` declares the `BaseTracer`/`Span` contract plus a working no-op default, while every real backend (Langfuse today) lives in `integrations/tracing/`, so no telemetry SDK is a runtime dependency of this layer. No integration, component, pipeline, or agent ever imports from outside `core/` for its base abstractions.

## Directory Layout

```
core/
├── base.py           # Entity (UUID mixin), Timestamped
├── config.py         # ProviderConfig (base class every interfaces/<domain>/config.py extends), ModelConfig (ProviderConfig subclass adding `model: str`, for model-backed domains), RequestOptions (ProviderConfig subclass adding `timeout`/`max_retries`, mixed in by network-bound domains' configs)
├── errors.py         # PlatformError hierarchy (ProviderError, ConfigError, LLMError, AgentError, …)
├── retry.py          # retry_async()/retry_sync() per-call retry and the with_retry() decorator: exponential backoff, honors PlatformError.retryable
├── resilience.py     # CircuitBreaker (open/half-open/closed state machine), RateLimiter (token bucket)
├── middleware.py     # Middleware protocol + MiddlewarePipeline: generic pre/post hooks around a typed operation
├── persistence.py    # Checkpointer[StateT] protocol + InMemoryCheckpointer
├── token_usage.py    # TokenUsageAggregator: accumulates TokenUsage per string key
├── credentials.py    # Credentials (secrets only, e.g. api_key), ClientOptions (base_url/timeout/max_retries)
├── observability/    # What the platform observed, in its own vocabulary (the contract lives in interfaces/tracing/)
│   ├── facts.py      #   Observation base + @Fact-annotated fields, flatten() -> span attributes
│   ├── observations.py #  The vocabulary itself: LLMCall, LLMBatch, ModelCost, ToolRun, AgentRun, GuardrailCheck, WorkflowNode, Handoff, PipelineRun, Retrieval, ComponentRun
│   ├── runtime.py    #   observe(observation) (one span), emit(observation) (facts on the enclosing span), current(kind)
│   ├── tracer.py     #   Tracer registry: set_tracer()/get_tracer()/reset_tracer(), current_span()
│   ├── content.py    #   Opt-in payload capture: capture_content(), payload_facts()
│   └── decorators.py #   @traced(Observation, **facts): wrap any sync or async function in one
├── similarity.py     # Vector similarity math helpers (compute_similarity, similarity_bounds), used by components/ and evals/
├── interfaces/       # ABCs for every capability (the "contract" layer)
│   ├── llm/
│   ├── embeddings/
│   ├── vad/
│   ├── ocr/
│   ├── vector_store/
│   ├── reranking/
│   ├── chunking/
│   ├── speech/
│   ├── translation/
│   ├── clustering/
│   ├── classification/
│   ├── loader/
│   ├── image_generation/
│   ├── dataset/
│   ├── tracing/      # BaseTracer + Span protocol + NoOpTracer default (see below),
│   │                   #   the one domain whose "no provider installed" case is a
│   │                   #   usable implementation rather than an error
│   └── mcp/          # BaseMCPClient: connect()/aclose() + list_tools()/call_tool(),
│                       #   the one domain ABC that owns connection lifecycle (an MCP
│                       #   server is a stateful session, not a stateless per-call provider)
└── schemas/          # Pydantic v2 data structures (shared across all layers)
    ├── document.py
    ├── chunk.py
    ├── embedding.py
    ├── score.py
    ├── token.py
    ├── span.py
    ├── cluster.py
    ├── message.py
    ├── conversation.py
    ├── mcp.py        # MCPToolSpec: one MCP server tool's name/description/raw JSON Schema
    └── enums.py
```

### `interfaces/`: The Contract Layer

Each subdirectory follows a fixed pattern (example: `interfaces/llm/`):

```
interfaces/<domain>/
├── base.py           # ABC (e.g. BaseLLMProvider) with abstract async methods
├── config.py         # Pydantic config model for the domain
└── response.py       # Pydantic response model(s)
```

`interfaces/llm/` has one file beyond that fixed pattern: `fallback.py`'s `FallbackLLMProvider` (+ `FallbackEntry`), a `BaseLLMProvider` implementation that tries an ordered list of other `BaseLLMProvider`s, one `CircuitBreaker` per entry, falling through to the next on every failure, though only a retryable one counts against that entry's circuit (see `resilience.py` below). It belongs here rather than in `integrations/` because it composes the ABC generically and imports no vendor SDK, same rule as everything else in `interfaces/`.

An interface ABC **never imports from `integrations/`**, **never imports from `components/`**, and **never imports from `pipelines/`**. It may only use `core/schemas/`, `core/config.py`, `core/errors.py`, and `core/credentials.py`. Every domain's `config.py` defines a `<Domain>Config` that extends `core/config.py`'s `ProviderConfig`, the shared base for all provider configs. Domains whose providers are parameterized by a model id (llm, embeddings, reranking, speech, image_generation) extend `ModelConfig` instead, which adds `model: str` on top of `ProviderConfig`; domains without a model concept (loader, chunking, clustering, vad, ocr, translation, classification, vector_store, dataset) extend `ProviderConfig` directly.

**Where a field belongs:** put it on the shared `<Domain>Config` only if almost every provider in the domain actually uses it. If just one or two do, it belongs on that provider's own `config.py` instead (e.g. `max_samples` lives on `SileroVadConfig`, not `VADConfig`, since only Silero needs it). Fields no provider uses get deleted, not kept around. One exception: a field a cross-cutting caller sets regardless of backend (`OCRConfig.min_confidence`) can stay on the shared config even if a specific provider ignores it.

**Defaults across providers:** don't hardcode one vendor's default onto the shared config, since other vendors often default differently for the same knob. Use `None` for "unset" and forward the field conditionally (`if config.x is not None`), so an untouched field lets each provider's own default apply (see `GenerationConfig.temperature`).

**Required parameters never live on config:** `config` is always optional (`config: ConfigT | None = None`) on every ABC method, so a caller can always omit it and get provider defaults. That means any value a call cannot proceed without (a path, a query string, the text to classify, the items to embed, …) must be its own positional/keyword parameter on the method, never a `Config` field, even a required one, since a required config field would force every caller to construct a config just to make an otherwise-optional-looking argument work (`BaseDatasetProvider.load(self, record_type, path, config=None)` puts `path` directly on the signature rather than inside `DatasetConfig`, for exactly this reason). Concretely: no `<Domain>Config` field is ever declared without a default; every field has a `None` or concrete default so the config stays fully optional as a whole.

### `schemas/`: Shared Data Structures

All Pydantic v2 models live here. They are the currency passed between layers, a component returns a schema type, a pipeline accepts one, a tool wraps one.

### `errors.py`: Error Hierarchy

```
PlatformError
├── ProviderError
├── ConfigError
├── NotFoundError
├── ValidationError
├── MissingCredentialError
├── LLMError
│   ├── LLMGenerationError
│   ├── LLMTimeoutError
│   └── LLMRateLimitError
├── AgentError
│   ├── AgentThinkError
│   ├── AgentActError
│   └── AgentMaxIterations
└── ToolError
```

Every error carries `code`, `retryable`, and `context` fields. `retry.py` holds the retry loop in two per-call functions, `retry_async(attempt_once, max_attempts=...)` and `retry_sync(...)` (exponential backoff with jitter, `retry_on` configurable; the sync one sleeps with `time.sleep`), and the `with_retry()` decorator delegates to the async one. They check this flag: if a caught exception is a `PlatformError` with `retryable=False`, it re-raises immediately instead of burning through retry attempts. For most `@with_retry()` call sites in `integrations/`, which sit *inside* an outer `@error_logged(re_raise=ProviderError)`, the decorator only sees the raw underlying SDK exception, so the flag-check is a no-op there. The `llm` providers are the exception: `NativeLLMProvider.generate`/`agenerate` classify each vendor error as temporary or not inside one attempt (`ProviderError(retryable=...)`) and run the attempts through `retry_sync`/`retry_async`, so a bad request fails after one call and `retryable` is honored.

### `interfaces/tracing/`: The Backend Contract

Observability is split like any other domain: `core` owns the contract, `integrations/tracing/` owns the backends.

`interfaces/tracing/` holds the generic layer, with no tracing SDK imported anywhere in it:
- `span.py`: the `Span` protocol (`set_attribute()`, `record_error()`, `trace_id`, `span_id`) and `AttributeValue`, the narrow set of value types an attribute may hold. It is a `Protocol` rather than an ABC because implementations adapt a backend's own span object instead of subclassing ours. `record_error(exc, record_exception=False)` is for call sites that catch and swallow an error (rather than re-raise it) but still need the span tagged as failed, e.g. tool calls in `agents/tools/registry.py`. `trace_id`/`span_id` exist so a caller can correlate a trace with data submitted to a backend out of band, e.g. attaching an eval score or user feedback to a Langfuse/LangSmith trace after the fact.
- `base.py`: `BaseTracer`, a single `start_span(name, attributes)` returning a context manager. The seam is deliberately at the tracer level, not the span-context level: an implementation delegates parent/child nesting and cross-task context propagation to its backend rather than reimplementing them.
- `noop.py`: `NoOpTracer`/`NoOpSpan`, the default. This is what lets every call site use `observe()` unconditionally with no "is tracing enabled" branch and no tracing extra installed.

Backend selection and the Langfuse wiring live in `integrations/tracing/` and `integrations/README.md`.

### `observability/`: Typed Observations

`core` describes **what the platform observed, in the platform's own words**. It never spells a backend's field names: `integrations/tracing/langfuse/mapping.py` is the single place the platform's vocabulary is translated into Langfuse's. That is what lets a backend change its fields without a single call site changing.

The unit is an `Observation`: a mutable dataclass whose fields are annotated with `Fact`.

```python
@dataclass
class ToolRun(Observation):
    operation = "execute_tool"

    tool_name: Annotated[str, Fact("tool.name")]
    call_id: Annotated[str | None, Fact("tool.call.id")] = None
    arguments: Annotated[Any | None, Fact("tool.arguments", payload=True)] = None
```

One object therefore declares everything a backend needs, and `facts.py::flatten()` derives all of it:
- the **span attributes**, from every non-`None` fact;
- the **payload policy**, from the facts marked `payload=True`, which expand into `{key}.size` and `{key}.digest` always and `{key}.content` only when capture is on.

Adding a fact is one annotated field rather than an edit in several places.

`observations.py` holds the vocabulary itself, and is the only file to read to know what the platform reports: `LLMCall`, `LLMBatch`, `ModelCost`, `ToolRun`, `AgentRun`, `GuardrailCheck`, `WorkflowNode`, `Handoff`, `PipelineRun`, `Retrieval`, `ComponentRun`. A layer with a concept `core` has no business knowing about declares its own locally instead, which is what `evals/runner.py` does with `_EvalCaseRun`.

### `observability/runtime.py`: `observe`, `emit`, `current`

`observe(observation)` runs a unit of work: it opens one span named after `observation.operation_name` and tags a failure (raised or swallowed via `record_error`) with `error.type`. Duration is the span's own, so nothing here times the block.

Observations are mutable on purpose, because most of what is worth recording is only known once the work is done. `observe` yields the observation back, records what was known when the span opened, and flushes whatever changed when it closes:

```python
with observe(LLMCall(provider="openai", request_model=model)) as call:
    response = await client.chat(...)
    call.response_model = response.model
    call.input_tokens = response.usage.input_tokens
```

`emit(observation)` is the event counterpart: facts are stamped on whatever span is already open, but there is no duration, because there is nothing to time. It is for a step inside someone else's unit of work: what a retrieval found (`Retrieval`), who a workflow node handed off to (`Handoff`), what a finished model call cost (`ModelCost`, priced only after the call it describes has ended).

`current(ObservationType)` returns the innermost open observation of that type, so code nested under someone else's `observe` can annotate it without that object being threaded through every signature in between.

`tracer.py` is the registry behind both: `set_tracer()` installs the process-wide backend once at startup (`api/app.py`, via `config.container.build_tracer()`) and `reset_tracer()` restores the no-op. Nothing here selects or knows about a backend.

### `observability/content.py`: Opt-In Payload Capture

Prompts, tool arguments and tool results are the most useful thing in a trace and the most likely to contain personal data, so they are **off by default**. What is always exported is the shape of a payload, never its content: `payload_facts(value)` returns its serialized size in bytes and a 16-character sha256 digest, which is enough to spot an oversized argument or to tell two identical calls apart without a backend ever seeing the text.

`capture_content()` decides whether the content itself is exported, reading `AGENT_PLATFORM_CAPTURE_CONTENT` per call, so toggling it needs no restart. It is the only switch: a backend never decides capture itself. It also covers errors: without it, a span's error carries the exception type only, never its message or stack trace.

### `observability/decorators.py`: `@traced`

`traced(ObservationType, **facts)` wraps any function, sync or async, in `observe(ObservationType(**facts))`. It exists so that instrumenting a whole unit of work is one line and no indentation change, which is what makes it practical to trace every pipeline entry point (`pipelines/`) and every `Component.arun` (`components/base.py` applies it automatically in `__init_subclass__`). A fresh observation is built per call, since an observation is mutable; anything only known at runtime is set through `current(...)` inside the body. It preserves the wrapped function's identity through `functools.wraps`, so it composes with the rest of the decorator stack.


### `resilience.py`: Circuit Breaker & Rate Limiter

Sits alongside `retry.py` rather than replacing it: `with_retry()` retries a single failing call, `CircuitBreaker` and `RateLimiter` guard a call site's overall traffic. `CircuitBreaker(failure_threshold, reset_timeout)` wraps an async callable via `call(func, *args, **kwargs)`; it tracks consecutive failures, opens (raising `ProviderError(retryable=True)` without invoking `func`) once `failure_threshold` is hit, and moves to half-open after `reset_timeout` seconds elapse, a single success there closes it again, a failure reopens it. A `PlatformError` with `retryable=False` (a bad request, a revoked key) is re-raised without counting as a failure, since it says nothing about the service being down. `RateLimiter(rate, burst=None)` is a token-bucket limiter (`burst` defaults to `rate`, i.e. one second of headroom); it raises `ConfigError` when `rate <= 0` or the capacity (`burst`, else `rate`) is below 1, since a bucket that cannot hold one token would block every `acquire()` forever; `await acquire(tokens=1.0)` blocks until enough tokens have replenished. Both are generic over any async callable, no `agents/tools/` concept involved. `record_success()`/`record_failure()`/`record_error(exc)` on `CircuitBreaker` are public (not just reachable through `call()`; `record_error` applies the non-retryable rule before counting) so a call site that can't route through an awaited wrapper, a sync call, or one step of a streaming response, can still report the outcome; `core/interfaces/llm/fallback.py::FallbackLLMProvider` is the first such consumer. `CircuitBreaker` is wired into that per-provider LLM fallback chain; `RateLimiter` remains available but unused by any call site so far.

### `middleware.py`: Generic Pre/Post Hooks

`Middleware[CtxT, ResultT]` is a `Protocol` with `async before(ctx: CtxT) -> ResultT | None` and `async after(ctx: CtxT, result: ResultT) -> ResultT`; a non-`None` return from `before` short-circuits the pipeline (the operation and every `after` are skipped, that value becomes the result). `CtxT` and `ResultT` are separate type parameters since the context an operation reads from and the value it produces are often different types (e.g. `agents/guardrails.py`'s `GuardrailContext` in, `AssistantMessage` out); collapsing them into one parameter would make `before`'s short-circuit return type wrong for any such consumer. `MiddlewarePipeline[CtxT, ResultT]` composes a list of them: `run(ctx, operation)` calls each `before` in order, runs `operation(ctx)` if none short-circuited, then calls each `after` in *reverse* order so the first middleware wraps outermost, matching the usual decorator/onion model. No `Agent`/`Tool` concept, reused later for agent guardrails and tool-call approval hooks.

### `persistence.py`: Checkpointing

`Checkpointer[StateT]` is a `Protocol` with `async save(key, state) -> None` and `async load(key) -> StateT | None`. `InMemoryCheckpointer[StateT]` is the default implementation, a plain dict keyed by `key`. Generic over `StateT`, so it works for a `ConversationAgent` session or a future `workflows/` run state alike; a real backend (Redis, a DB table) implements the same protocol.

### `token_usage.py`: Token Accounting

`TokenUsageAggregator` accumulates `TokenUsage` (see `schemas/token.py`) under an arbitrary string key via `record(key, usage)`, using `TokenUsage.__add__`. `total_for(key)` returns that key's running total (`TokenUsage.zero()` if unseen), `grand_total()` sums across every key, `keys()` lists recorded keys. No opinion on what the key means, a conversation id, a session id, a workflow-run id all work the same way. Wired into `agents/agent.py::Agent` (optional `token_usage_aggregator`/`usage_key` constructor params, recorded after every `_generate()`/`think_stream()` call) and exposed via `Agent.token_usage`; `ConversationAgent` passes its own `conversation_id` as the key, so `agent.token_usage` is the running total for that conversation. `config/container.py::build_agent` wires one in by default; `api/app.py`'s `/chat` route returns it as `ChatResponse.usage`.

## How to Extend

### Add a new interface (new capability)

```
core/interfaces/<new_domain>/
├── __init__.py
├── base.py           # class Base<NewDomain>Provider(ABC): …
├── config.py         # class <NewDomain>Config(BaseModel): …
└── response.py       # class <NewDomain>Response(BaseModel): …
```

1. Define the ABC in `base.py` with `@abstractmethod` async methods
2. Define the config model in `config.py`
3. Define response models in `response.py`
4. Export from `core/interfaces/<domain>/__init__.py`
5. If the domain defines a new shared data type, add it to `core/schemas/`

### Add a new schema

1. Add a new file in `core/schemas/` (or extend an existing one)
2. Use `pydantic.BaseModel`
3. Export from `core/schemas/__init__.py`

### Add a new error

1. Add the class in `core/errors.py`
2. Add to `__all__` in the same file
3. Place it in the correct branch of the `PlatformError` hierarchy

### Add a new enum

1. Add to `core/schemas/enums.py`
2. Export from `core/schemas/__init__.py`

## Constraints

- **No imports from `integrations/`, `components/`, `pipelines/`, or `agents/`**
- **No external library imports** (no langchain, no openai, no opentelemetry, etc.), only `pydantic`, `abc`, `typing`, `uuid`, `datetime`
- ABCs use `Generic` TypeVars for type safety where appropriate
