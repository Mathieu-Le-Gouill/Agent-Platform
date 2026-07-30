# Core: Library-Independent Foundation

## Design

`core/` is the **near-zero-dependency foundation** of the platform. It defines everything that does not depend on an external library, pure Python ABCs, Pydantic schemas, enums, and error types, plus `tracing.py`/`genai_tracing.py`, whose one exception is depending on the lightweight `opentelemetry-api` (always installed; a no-op unless the optional `tracing` extra and an OTLP endpoint are configured). No integration, component, pipeline, or agent ever imports from outside `core/` for its base abstractions.

## Directory Layout

```
core/
├── base.py           # Entity (UUID mixin), Timestamped
├── config.py         # ProviderConfig (base class every interfaces/<domain>/config.py extends), ModelConfig (ProviderConfig subclass adding `model: str`, for model-backed domains), RequestOptions (ProviderConfig subclass adding `timeout`/`max_retries`, mixed in by network-bound domains' configs)
├── errors.py         # PlatformError hierarchy (ProviderError, ConfigError, LLMError, AgentError, …)
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
│   └── dataset/
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

Every error carries `code`, `retryable`, and `context` fields.

### `tracing.py`: Vendor-Agnostic Observability

Contains zero vendor-specific code and zero GenAI-specific code: it only knows OpenTelemetry primitives, `TracingBackend` is `AUTO` (default, exports over OTLP whenever `OTEL_EXPORTER_OTLP_ENDPOINT`/`_TRACES_ENDPOINT` is set, otherwise a safe no-op), `NONE` (force-disabled), or `CONSOLE` (stdout, for local debugging). `configure_tracing()` is called once at process startup (`api/app.py`) to install a `TracerProvider`; it defers to any provider already installed by something else (`opentelemetry-instrument`, an APM agent, the host application) instead of overwriting it, and accepts an `exporter=` override for backends with no standard OTLP path.

`traced_span(name, attributes=None)` is the generic context manager every span-producing call site is built on: it records latency as span duration automatically and, on a propagating exception, calls `mark_span_error(span, exc)`. `mark_span_error(span, exc, *, record_exception=True)` sets the OTel error-status attributes by hand; pass `record_exception=False` for call sites that catch and swallow an error (rather than re-raise it) but still need the span tagged as failed, e.g. tool calls in `agents/tools/registry.py`.

Vendor endpoint/auth recipes (LangSmith, Langfuse, ...) live in `integrations/README.md` as env var snippets, not in this module, any OTLP-speaking backend works via the same `AUTO` path.

### `genai_tracing.py`: GenAI Semantic Conventions

Paired with `tracing.py`, this is where GenAI-specific knowledge lives. `traced_operation_span(operation, attributes=None)` (a thin wrapper over `traced_span()`) is what every instrumented call site uses, it names the span after `operation` and sets `gen_ai.operation.name` to match. `GenAIAttributes` centralizes the `gen_ai.*` semantic-convention attribute keys so every call site (and any future one, embeddings, reranking, ...) spells them identically; `record_token_usage(span, usage)` records token counts the same way everywhere. Currently wired into the agent loop (`agents/executor.py`), tool calls (`agents/tools/registry.py`), and LLM calls (each `integrations/llm/<provider>/provider.py`).

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
