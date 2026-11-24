# agent_platform: Codebase Architecture

## Layers

```
┌──────────────────────────────────────────────────┐
│  evals/      ← regression testing (EvalRunner,    │
│                 EvalDataset, Scorer, evals/cli.py) │
├──────────────────────────────────────────────────┤
│  agents/         ← composition (Agent, Executor,  │
│                     ConversationAgent, Tool ABC,  │
│                     ToolRegistry, 7 tools + MCP)   │
│  api/        ← FastAPI entrypoint (build_agent)   │
│  workflows/  ← orchestration (WorkflowGraph, DAG,  │
│                agent/tool/handoff nodes)          │
├──────────────────────────────────────────────────┤
│  pipelines/         ← orchestration               │
│  (ingestion, speech_translation, rag)             │
├──────────────────────────────────────────────────┤
│  components/        ← reusable processing units   │
├──────────────────────────────────────────────────┤
│  integrations/      ← service adapters            │
├──────────────────────────────────────────────────┤
│  core/  audio/  utils/ ← foundation               │
└──────────────────────────────────────────────────┘
```

## Layer Details

### `core/`: Foundation

Near-zero external dependencies. Everything here is pure Python, `pydantic`, `abc`, `typing`, `uuid`, `datetime`, with no exceptions: observability is defined here only as the `BaseTracer`/`Span` contract plus its no-op default, and every real backend (Langfuse today) lives in `integrations/tracing/`.

| Path | Contents |
|---|---|
| `core/base.py` | `Entity` (UUID mixin), `Timestamped` |
| `core/config.py` | `ProviderConfig`, the base class every `core/interfaces/<domain>/config.py` extends; `ModelConfig`, a `ProviderConfig` subclass adding `model: str` for model-backed domains (llm, embeddings, reranking, speech, image_generation); `RequestOptions`, a `ProviderConfig` subclass adding `timeout`/`max_retries`, mixed in by network-bound domains' configs alongside `ModelConfig` (currently `llm`, `embeddings`, `moderation`) |
| `core/errors.py` | `PlatformError` hierarchy, `ProviderError`, `ConfigError`, `NotFoundError`, `ValidationError`, `MissingCredentialError`, `RecoveryExhausted`, `AgentError`, `ToolError`. `PlatformError` carries `code`, `retryable`, `retry_after`, `context`. `retry.py`'s `with_retry(translate=)` turns each failed attempt into a `PlatformError` before the retry decision; non-LLM providers pass `integrations/sdk_errors.py::translate_sdk_error`. `LLMError` (under `ProviderError`) lives in `interfaces/llm/errors.py` |
| `core/credentials.py` | `Credentials` (secrets only, e.g. `api_key`), `ClientOptions` (non-secret construction-time transport settings: `base_url`, `timeout`, `max_retries`) |
| `core/interfaces/tracing/` | Backend-agnostic tracing contract: `BaseTracer`, the `Span` protocol (`set_attribute()`, `record_error()`, `trace_id`, `span_id`), and the `NoOpTracer`/`NoOpSpan` default |
| `core/observability/` | What call sites use: typed `Observation`s describing what the platform observed in its own vocabulary (`observations.py`: `LLMCall`, `ToolRun`, `AgentRun`, `GuardrailCheck`, `WorkflowNode`, `PipelineRun`, ...), the `Fact` annotation and `flatten()` that derive span attributes and payload policy from them (`facts.py`), the `observe()`/`emit()`/`current()` runtime, the tracer registry, opt-in payload capture, and `@traced`. No backend name appears here: `integrations/tracing/langfuse/mapping.py` maps the platform's vocabulary onto Langfuse fields |
| `core/interfaces/<domain>/` | ABCs for every capability (llm, embeddings, vad, ocr, vector_store, reranking, chunking, speech, translation, clustering, classification, loader, image_generation, dataset) |
| `core/schemas/` | Shared Pydantic v2 data models used across all layers |

#### `core/schemas/`: Shared Data Structures

All models are `pydantic.BaseModel`. Frozen where appropriate.

| File | Contents |
|---|---|
| `document.py` | `Document` hierarchy, `TextDocument`, `ImageDocument`, `AudioDocument`, `VideoDocument` |
| `chunk.py` | `Chunk` hierarchy, `TextChunk`, `AudioChunk`, `VideoChunk` |
| `message.py` | Chat messages, `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage`, `ToolCall`, `ToolResult`, `Prompt` |
| `conversation.py` | `Utterance` (speaker, text, timestamps, confidence), `Transcript` |
| `embedding.py` | `Embedding` with vector, norm, dimension helpers |
| `score.py` | `Score` with kind, bounds, normalization |
| `token.py` | `TokenUsage` with addition, total |
| `span.py` | `TimeSpan`, `SampleSpan` |
| `cluster.py` | `Cluster` with label, items, centroid |
| `enums.py` | `MediaType`, `DocumentFormat`, `ImageFormat`, `AudioFormat`, `VideoFormat`, `DataType`, `Language`, `FileFormat` |

### `integrations/`: Service Adapters

Each domain follows:

```
integrations/<domain>/
├── __init__.py
├── langchain_base.py    # Optional: LangChain intermediate abstract
└── <provider>/
    ├── __init__.py
    ├── config.py        # Provider-specific Pydantic config
    └── <provider>.py    # Concrete class implementing core/interfaces ABC
```

`loader/` uses `strategies/` instead of plain `<provider>/` subdirectories because each strategy handles a different media type rather than a different vendor.

Each domain's `integrations/<domain>/__init__.py` re-exports its provider classes lazily via a module-level `__getattr__` (PEP 562) over a `_PROVIDERS: dict[class_name, module_path]` map, so callers get a flat import, `from agent_platform.integrations.llm import OpenAILLM`, without eagerly importing every provider's SDK (each class is only imported on first access). Add a new provider to a domain by adding one entry to that domain's `_PROVIDERS` map.

### `components/`: Composable Units

`Component[InputT, OutputT]` ABC with `async def arun(input) -> OutputT`. `generation/llm_task/` holds `LLMTask[T]`, one prompt plus an output type that runs on any LLM provider, with `classify_task` and `summarize_task` as presets. See `components/README.md`.

### `pipelines/`: Orchestration Flows

Multi-step operations that chain components together. See `pipelines/README.md`.

### `audio/`: DSP Utilities

- `io.py`, AudioDocument/Chunk construction, tensor/numpy/base64 conversions
- `dsp.py`, resampling, waveform chunking, full audio processing pipeline

### `config/`: Settings and DI

- `settings.py`, `Settings` (`pydantic-settings`, env prefix `AGENT_PLATFORM_`), `get_settings()` (`lru_cache`'d singleton; a malformed setting raises `ConfigError` naming the field and rule, never the value). Default provider selection for each of the three wired domains is a single `"<provider>:<model>"` string (`default_llm_model`, `default_image_model`, `default_audio_model`), parsed by `model_string.py::parse_model_string`
- `model_string.py`, `parse_model_string(value)`, splits a `"<provider>:<model>"` string into `(provider, model)`; kept separate from `core/config.py`'s `ProviderConfig`/`ModelConfig` pydantic schemas since it's DI-wiring logic specific to `Settings`' model-string convention, not a schema those classes share
- `container.py`, `build_provider(domain_module, provider_name)`, a generic factory that instantiates any `integrations.<domain>` provider. `provider_name` is resolved against that domain module's `PROVIDER_ALIASES` map first (a short slug, e.g. `"openai"` -> `"OpenAILLM"`), falling back to the literal exported class name (`getattr(domain_module, provider_name)()`) so an explicit class name still works; raises `ConfigError` if neither resolves. It reads no separate provider registry, the target domain's own `_PROVIDERS` + `PROVIDER_ALIASES` maps (see `integrations/` above) are already the source of truth for valid names
- `container.py`, `build_provider_from_model_string(domain_module, model_string)`, combines `parse_model_string` + `build_provider`, returning `(provider_instance, model)`
- `container.py`, `build_agent(settings)`, builds a `ConversationAgent` whose LLM, `GenerateImageTool`, and `TranscribeTool` (registered on its `ToolRegistry`) are all resolved this way from `settings.default_llm_model`/`default_image_model`/`default_audio_model`

### `api/`: FastAPI Entrypoint

`app.py` builds a template `ConversationAgent` via `config.build_agent_async()` behind `/chat`, `/chat/stream` and `/health`, using `config.Settings`/`get_settings()` for configuration and `config.setup_logging()` + `core.observability.set_tracer(config.build_tracer(settings))` at startup. That agent already carries image-generation and transcription tools (see `config/` above), not chat only. `create_app()` builds the app; there is no module-level `app`, and `main()` (the `agent-platform-api` script) calls it.

- **Auth** (`auth.py`): `/chat` and `/chat/stream` require `Authorization: Bearer <key>` with a key from `AGENT_PLATFORM_API_KEYS` (a JSON list); anything else is 401. `/health` stays open. The app refuses to start (`ConfigError`) with no keys unless `AGENT_PLATFORM_API_AUTH_DISABLED=true`; then every caller is one anonymous owner, with one shared conversation cap, and the `/docs`, `/redoc` and `/openapi.json` routes stay on (they are off when auth is enabled). A caller's key id is the SHA-256 digest of its key; the raw key is never stored or logged.
- **Conversations** (`conversations.py`): `/chat` starts a conversation when `conversation_id` is absent and returns its id in `ChatResponse.conversation_id`; send it back to continue. Each conversation is a `ConversationAgent.with_conversation()` fork of the template, private to the key that created it: another key's id answers 404, the same as an unknown id. Turns on one conversation run one after the other. Each key keeps at most `AGENT_PLATFORM_API_MAX_CONVERSATIONS_PER_KEY` (default 100); one more evicts its least recently used. Storage is in memory: lost on restart and not shared between uvicorn workers.
- **Errors** (`errors.py`): a failure answers `{"error": <code>, "message": <fixed text>}`, never the exception's own text (it can echo the prompt). `llm_rate_limited` is 429 (with `Retry-After` when the provider gave one), `llm_context_window` 422, `llm_timeout` 504, `provider_unavailable` 503, `provider_bad_request` and `agent_recovery_exhausted` 502, `agent_max_iterations` 500. Any other `PlatformError` or exception is 500 `internal_error`. A 5xx logs one traceback, a lower status one warning line. `/chat/stream` has already sent its 200 status, so a failure ends the stream with `data: {"type": "error", "error": <code>, "message": <text>}` and no usage event; a client disconnect sends nothing.
- **`/chat/stream`** is stateless: it ignores `conversation_id` and runs on a throwaway fork, so its usage event covers that turn only.

## Extension Patterns

### Add a new integration provider
```
integrations/<domain>/<new_provider>/
├── __init__.py
├── config.py
└── <new_provider>.py   # implements core/interfaces/<domain>/base.py ABC
```

### Add a new LLM provider (native SDK)
```python
class MyLLM(BaseLLMProvider[MyGenerationConfig]):
    def _client(self, config: MyGenerationConfig) -> SomeVendorClient:
        return SomeVendorClient(api_key=..., **_to_native_params(config))
```

`llm`, `embeddings`, `vector_store`, and `reranking` providers each call their
vendor's native SDK or REST endpoint directly, no shared LangChain wrapper.
Only `chunking` still has an optional `langchain_base.py`, wrapping
`langchain-text-splitters` (see `integrations/README.md`).

### Add a new integration domain
1. Create the ABC in `core/interfaces/<domain>/` (see `core/README.md`)
2. Create `integrations/<domain>/<first_provider>/` and implement the ABC

### Wire a new provider domain into `Settings`/DI

Once a domain has real consumers beyond direct constructor injection (the
pattern `agents/tools/*` use today), expose it through `config/` the same way
`llm`/`image_generation`/`speech_to_text` already are: a `PROVIDER_ALIASES`
map on the domain module, plus a `"<provider>:<model>"` `Settings` field:

```python
# integrations/vector_store/__init__.py
PROVIDER_ALIASES: dict[str, str] = {"chroma": "ChromaStore", "qdrant": "QdrantStore"}

# config/settings.py
default_vector_store_model: str = "chroma:default"

# config/container.py
def build_vector_store(settings: Settings) -> BaseVectorStore:
    import agent_platform.integrations.vector_store as vector_store_module
    provider, _model = build_provider_from_model_string(
        vector_store_module, settings.default_vector_store_model
    )
    return provider
```

`build_provider`/`build_provider_from_model_string` already work for any
domain module that follows the `_PROVIDERS: dict[class_name, module_path]` +
`PROVIDER_ALIASES: dict[slug, class_name]` + lazy `__getattr__` convention,
so this is the only code needed, no new registry, no per-domain dict.

## Known Issues

| Issue | Location | Status |
|---|---|---|
| No parallel fan-out/fan-in (a strict DAG walked one node at a time) | `workflows/graph.py` | Tracked in `workflows/README.md` |
