# agent_platform: Codebase Architecture

## Layers

```
┌──────────────────────────────────────────────────┐
│  agents/         ← composition (Agent, Executor,  │
│                     ConversationAgent, Tool ABC,  │
│                     ToolRegistry, 3 tools)        │
│  api/        ← FastAPI entrypoint (build_agent)   │
│  workflows/  ← stub                                │
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

Near-zero external dependencies. Everything here is pure Python, `pydantic`, `abc`, `typing`, `uuid`, `datetime`, with one exception: `core/tracing.py` depends on `opentelemetry-api` (always installed, lightweight; a no-op unless the optional `tracing` extra + an OTLP endpoint are configured).

| Path | Contents |
|---|---|
| `core/base.py` | `Entity` (UUID mixin), `Timestamped` |
| `core/config.py` | `ProviderConfig`, the base class every `core/interfaces/<domain>/config.py` extends |
| `core/errors.py` | `PlatformError` hierarchy, `ProviderError`, `ConfigError`, `NotFoundError`, `ValidationError`, `MissingCredentialError`, `LLMError`, `AgentError`, `ToolError` |
| `core/credentials.py` | `BaseCredentials`, `ProviderCredentials` |
| `core/tracing.py` | Vendor-agnostic OpenTelemetry tracing: `TracingBackend`, `TracingConfig`, `configure_tracing()`, `traced_span()`/`traced_operation_span()`, `record_token_usage()`, `GenAIAttributes` |
| `core/interfaces/<domain>/` | ABCs for every capability (llm, embeddings, vad, ocr, vector_store, reranking, chunking, speech, translation, clustering, classification, loader, image_generation) |
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

`Component[InputT, OutputT]` ABC with `async def arun(input) -> OutputT`. See `components/README.md`.

### `pipelines/`: Orchestration Flows

Multi-step operations that chain components together. See `pipelines/README.md`.

### `audio/`: DSP Utilities

- `io.py`, AudioDocument/Chunk construction, tensor/numpy/base64 conversions
- `dsp.py`, resampling, waveform chunking, full audio processing pipeline

### `config/`: Settings and DI

- `settings.py`, `Settings` (`pydantic-settings`, env prefix `AGENT_PLATFORM_`), `get_settings()` (`lru_cache`'d singleton)
- `container.py`, `build_provider(domain_module, provider_name)`, a generic factory that instantiates any `integrations.<domain>` provider by its exported class name (`getattr(domain_module, provider_name)()`), raising `ConfigError` if the name isn't found. It reads no separate provider registry, the target domain's own `_PROVIDERS` map (see `integrations/` above) is already the source of truth for valid names, so a `Settings` field for a domain just stores that class name directly (e.g. `llm_provider: str = "OpenAILLM"`)
- `container.py`, `build_agent(settings)`, the only concrete wiring today: builds a `ConversationAgent` from `settings.llm_provider` via `build_provider`

### `api/`: FastAPI Entrypoint

`app.py` wires one `ConversationAgent` via `config.build_agent()` behind `/chat` and `/health`, using `config.Settings`/`get_settings()` for configuration and `config.setup_logging()` + `core.tracing.configure_tracing()` at startup.

## Extension Patterns

### Add a new integration provider
```
integrations/<domain>/<new_provider>/
├── __init__.py
├── config.py
└── <new_provider>.py   # implements core/interfaces/<domain>/base.py ABC
```

### Add a new LLM provider (LangChain-based)
```python
class MyLLM(LangChainLLMProvider):
    def _client(self, model, config):
        return SomeLangChainModel(model, **_to_langchain_some(config))
```

### Add a new integration domain
1. Create the ABC in `core/interfaces/<domain>/` (see `core/README.md`)
2. Create `integrations/<domain>/<first_provider>/` and implement the ABC

### Wire a new provider domain into `Settings`/DI

Once a domain has real consumers beyond direct constructor injection (the
pattern `agents/tools/*` use today), expose it through `config/` instead of
adding a bespoke factory function per domain:

```python
# config/settings.py
vector_store_provider: str = "ChromaStore"   # class name from integrations.vector_store

# config/container.py
def build_vector_store(settings: Settings) -> BaseVectorStore:
    import agent_platform.integrations.vector_store as vector_store_module
    return build_provider(vector_store_module, settings.vector_store_provider)
```

`build_provider` already works for any domain module that follows the
`_PROVIDERS: dict[class_name, module_path]` + lazy `__getattr__` convention,
so this is the only code needed, no new registry, no per-domain dict.

## Known Issues

| Issue | Location | Status |
|---|---|---|
| `speech_translation.py` is pseudocode, `run()` raises `NotImplementedError` | `pipelines/speech_translation.py` | Open |
| Minimal RAG pipelines | `rag/ingest.py`, `rag/query.py` missing chunk/embed/rerank/generate steps | Open |
| `classification` has no `integrations/` layer, only interface response models and component-level use | `core/interfaces/classification/`, `integrations/classification/` (missing) | Open |
| `classification-transformers` extra declared but unused | `pyproject.toml` | Open |
| `uv run --extra chunking-pdf`/`--extra all` fails (uv resolves an old `unstructured`→`numba` pin incompatible with Python >=3.10); `pip install -e ".[chunking-pdf]"` works | `pyproject.toml` | Open |
| Empty stub | `workflows/` | By design |
