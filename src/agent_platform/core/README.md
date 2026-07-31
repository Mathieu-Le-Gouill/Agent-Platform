# Core: Library-Independent Foundation

## Design

`core/` is the **near-zero-dependency foundation** of the platform. It defines everything that does not depend on an external library, pure Python ABCs, Pydantic schemas, enums, and error types, plus `tracing.py`/`genai_tracing.py`, whose one exception is depending on the lightweight `opentelemetry-api` (always installed; a no-op unless the optional `tracing` extra and an OTLP endpoint are configured). No integration, component, pipeline, or agent ever imports from outside `core/` for its base abstractions.

## Directory Layout

```
core/
├── base.py           # Entity (UUID mixin), Timestamped
├── config.py         # ProviderConfig (base class every interfaces/<domain>/config.py extends), ModelConfig (ProviderConfig subclass adding `model: str`, for model-backed domains), RequestOptions (ProviderConfig subclass adding `timeout`/`max_retries`, mixed in by network-bound domains' configs)
├── errors.py         # PlatformError hierarchy (ProviderError, ConfigError, LLMError, AgentError, …)
├── retry.py          # with_retry() decorator: exponential backoff, honors PlatformError.retryable
├── resilience.py     # CircuitBreaker (open/half-open/closed state machine), RateLimiter (token bucket)
├── middleware.py     # Middleware protocol + MiddlewarePipeline: generic pre/post hooks around a typed operation
├── persistence.py    # Checkpointer[StateT] protocol + InMemoryCheckpointer
├── token_usage.py    # TokenUsageAggregator: accumulates TokenUsage per string key
├── credentials.py    # Credentials (secrets only, e.g. api_key), ClientOptions (base_url/timeout/max_retries)
├── tracing.py        # TracingBackend, TracingConfig, configure_tracing(), traced_span(), mark_span_error()
├── genai_tracing.py  # GenAIAttributes, traced_operation_span(), record_token_usage()
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

`interfaces/llm/` has one file beyond that fixed pattern: `fallback.py`'s `FallbackLLMProvider` (+ `FallbackEntry`), a `BaseLLMProvider` implementation that tries an ordered list of other `BaseLLMProvider`s, one `CircuitBreaker` per entry, falling through to the next on failure (see `resilience.py` below). It belongs here rather than in `integrations/` because it composes the ABC generically and imports no vendor SDK, same rule as everything else in `interfaces/`.

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

Every error carries `code`, `retryable`, and `context` fields. `retry.py`'s `with_retry()` decorator (exponential backoff with jitter, `max_attempts`/`retry_on` configurable) checks this flag: if a caught exception is a `PlatformError` with `retryable=False`, it re-raises immediately instead of burning through retry attempts. Since every current `@with_retry()` call site in `integrations/` sits *inside* an outer `@error_logged(re_raise=ProviderError)` (decorator order: `error_logged` above `with_retry`), `with_retry` there only ever sees the raw underlying SDK exception, never the translated `PlatformError`, so this flag-check is a no-op for those call sites today; it takes effect wherever `with_retry` wraps a call that itself raises `PlatformError` directly (e.g. a tool or agent-loop retry).

### `tracing.py`: Vendor-Agnostic Observability

Contains zero vendor-specific code and zero GenAI-specific code: it only knows OpenTelemetry primitives, `TracingBackend` is `AUTO` (default, exports over OTLP whenever `OTEL_EXPORTER_OTLP_ENDPOINT`/`_TRACES_ENDPOINT` is set, otherwise a safe no-op), `NONE` (force-disabled), or `CONSOLE` (stdout, for local debugging). `configure_tracing()` is called once at process startup (`api/app.py`) to install a `TracerProvider`; it defers to any provider already installed by something else (`opentelemetry-instrument`, an APM agent, the host application) instead of overwriting it, and accepts an `exporter=` override for backends with no standard OTLP path.

`traced_span(name, attributes=None)` is the generic context manager every span-producing call site is built on: it records latency as span duration automatically and, on a propagating exception, calls `mark_span_error(span, exc)`. `mark_span_error(span, exc, *, record_exception=True)` sets the OTel error-status attributes by hand; pass `record_exception=False` for call sites that catch and swallow an error (rather than re-raise it) but still need the span tagged as failed, e.g. tool calls in `agents/tools/registry.py`.

Vendor endpoint/auth recipes (LangSmith, Langfuse, ...) live in `integrations/README.md` as env var snippets, not in this module, any OTLP-speaking backend works via the same `AUTO` path.

### `genai_tracing.py`: GenAI Semantic Conventions

Paired with `tracing.py`, this is where GenAI-specific knowledge lives. `traced_operation_span(operation, attributes=None)` (a thin wrapper over `traced_span()`) is what every instrumented call site uses, it names the span after `operation` and sets `gen_ai.operation.name` to match. `GenAIAttributes` centralizes the `gen_ai.*` semantic-convention attribute keys so every call site (and any future one, embeddings, reranking, ...) spells them identically; `record_token_usage(span, usage)` records token counts the same way everywhere. Currently wired into the agent loop (`agents/executor.py`), tool calls (`agents/tools/registry.py`), and LLM calls (each `integrations/llm/<provider>/provider.py`). `GenAIAttributes.CONVERSATION_ID` is set on `agents/executor.py`'s `invoke_agent` span whenever a caller passes a `conversation_id` (`AgentExecutor.run`/`run_with_messages`/`run_streaming`'s optional keyword; `ConversationAgent.chat()` and `api/app.py`'s `/chat`, `/chat/stream` routes always do), so a multi-turn session's spans correlate without an external join; omitted entirely (not set to `None`) for callers that don't have one, e.g. a bare `Agent` driven directly.

### `resilience.py`: Circuit Breaker & Rate Limiter

Sits alongside `retry.py` rather than replacing it: `with_retry()` retries a single failing call, `CircuitBreaker` and `RateLimiter` guard a call site's overall traffic. `CircuitBreaker(failure_threshold, reset_timeout)` wraps an async callable via `call(func, *args, **kwargs)`; it tracks consecutive failures, opens (raising `ProviderError(retryable=True)` without invoking `func`) once `failure_threshold` is hit, and moves to half-open after `reset_timeout` seconds elapse, a single success there closes it again, a failure reopens it. `RateLimiter(rate, burst=None)` is a token-bucket limiter (`burst` defaults to `rate`, i.e. one second of headroom); `await acquire(tokens=1.0)` blocks until enough tokens have replenished. Both are generic over any async callable, no `agents/tools/` concept involved. `record_success()`/`record_failure()` on `CircuitBreaker` are public (not just reachable through `call()`) so a call site that can't route through an awaited wrapper, a sync call, or one step of a streaming response, can still report the outcome; `core/interfaces/llm/fallback.py::FallbackLLMProvider` is the first such consumer. `CircuitBreaker` is wired into that per-provider LLM fallback chain; `RateLimiter` remains available but unused by any call site so far.

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
- **No external library imports** (no langchain, no openai, no numpy, etc.), only `pydantic`, `abc`, `typing`, `uuid`, `datetime`, and `opentelemetry-api` (used only by `tracing.py`/`genai_tracing.py`)
- ABCs use `Generic` TypeVars for type safety where appropriate
