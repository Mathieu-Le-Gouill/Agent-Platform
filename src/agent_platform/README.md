# agent_platform: Codebase Architecture

## Layers

```
┌──────────────────────────────────────────────────┐
│  evals/      ← regression testing (EvalRunner,    │
│                 EvalDataset, Scorer, evals/cli.py) │
├──────────────────────────────────────────────────┤
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

Near-zero external dependencies. Everything here is pure Python, `pydantic`, `abc`, `typing`, `uuid`, `datetime`, with one exception: `core/tracing.py`/`core/genai_tracing.py` depend on `opentelemetry-api` (always installed, lightweight; a no-op unless the optional `tracing` extra + an OTLP endpoint are configured).

| Path | Contents |
|---|---|
| `core/base.py` | `Entity` (UUID mixin), `Timestamped` |
| `core/config.py` | `ProviderConfig`, the base class every `core/interfaces/<domain>/config.py` extends; `ModelConfig`, a `ProviderConfig` subclass adding `model: str` for model-backed domains (llm, embeddings, reranking, speech, image_generation) |
| `core/errors.py` | `PlatformError` hierarchy, `ProviderError`, `ConfigError`, `NotFoundError`, `ValidationError`, `MissingCredentialError`, `LLMError`, `AgentError`, `ToolError` |
| `core/credentials.py` | `Credentials` (secrets only, e.g. `api_key`), `ClientOptions` (non-secret construction-time transport settings: `base_url`, `timeout`, `max_retries`) |
| `core/tracing.py` | Vendor-agnostic OpenTelemetry tracing: `TracingBackend`, `TracingConfig`, `configure_tracing()`, `traced_span()`, `mark_span_error()` |
| `core/genai_tracing.py` | GenAI semantic-convention layer on top of `core/tracing.py`: `traced_operation_span()`, `record_token_usage()`, `GenAIAttributes` |
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

`Component[InputT, OutputT]` ABC with `async def arun(input) -> OutputT`. See `components/README.md`.

### `pipelines/`: Orchestration Flows

Multi-step operations that chain components together. See `pipelines/README.md`.

### `audio/`: DSP Utilities

- `io.py`, AudioDocument/Chunk construction, tensor/numpy/base64 conversions
- `dsp.py`, resampling, waveform chunking, full audio processing pipeline

### `config/`: Settings and DI

- `settings.py`, `Settings` (`pydantic-settings`, env prefix `AGENT_PLATFORM_`), `get_settings()` (`lru_cache`'d singleton). Default provider selection for each of the three wired domains is a single `"<provider>:<model>"` string (`default_llm_model`, `default_image_model`, `default_audio_model`), the convention LangChain's `init_chat_model`/Pydantic AI use, parsed by `model_string.py::parse_model_string`
- `model_string.py`, `parse_model_string(value)`, splits a `"<provider>:<model>"` string into `(provider, model)`; kept separate from `core/config.py`'s `ProviderConfig`/`ModelConfig` pydantic schemas since it's DI-wiring logic specific to `Settings`' model-string convention, not a schema those classes share
- `container.py`, `build_provider(domain_module, provider_name)`, a generic factory that instantiates any `integrations.<domain>` provider. `provider_name` is resolved against that domain module's `PROVIDER_ALIASES` map first (a short slug, e.g. `"openai"` -> `"OpenAILLM"`), falling back to the literal exported class name (`getattr(domain_module, provider_name)()`) so an explicit class name still works; raises `ConfigError` if neither resolves. It reads no separate provider registry, the target domain's own `_PROVIDERS` + `PROVIDER_ALIASES` maps (see `integrations/` above) are already the source of truth for valid names
- `container.py`, `build_provider_from_model_string(domain_module, model_string)`, combines `parse_model_string` + `build_provider`, returning `(provider_instance, model)`
- `container.py`, `build_agent(settings)`, builds a `ConversationAgent` whose LLM, `GenerateImageTool`, and `TranscribeTool` (registered on its `ToolRegistry`) are all resolved this way from `settings.default_llm_model`/`default_image_model`/`default_audio_model`

### `api/`: FastAPI Entrypoint

`app.py` wires one `ConversationAgent` via `config.build_agent()` behind `/chat` and `/health`, using `config.Settings`/`get_settings()` for configuration and `config.setup_logging()` + `core.tracing.configure_tracing()` at startup. That agent already carries image-generation and transcription tools (see `config/` above), not chat only.

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
| Empty stub | `workflows/` | By design |
