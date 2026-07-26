# Integrations: Self-Contained Provider Implementations

## Design

`integrations/` is where **external libraries meet the platform**. Each subdirectory implements one `core/interfaces` ABC by wrapping a specific external library or API. Integrations are the only layer that imports third-party packages (`openai`, `langchain-*`, `whisperx`, `chromadb`, `boto3`, …).

Every integration is **fully self-contained**:
- It vendors its own dependencies (declared in `pyproject.toml`)
- It manages its own credentials via `integrations/credentials.py` and `core/credentials.py`
- It encapsulates all conversion logic between the external library's types and the platform's `core/schemas/` types
- It never depends on another integration

## Directory Layout

```
integrations/
├── credentials.py         # Every provider's credential model, one file
├── <domain>/
│   ├── __init__.py
│   ├── langchain_base.py  # Optional: LangChain intermediate abstract
│   ├── _base.py           # Optional: native (non-LangChain) shared base, see below
│   └── <provider>/
│       ├── __init__.py
│       ├── config.py      # Provider-specific Pydantic config
│       └── <provider>.py  # Concrete class implementing core/interfaces ABC
└── …
```

**Exception:** `loader/` uses `strategies/` instead of `<provider>/` because each strategy handles a different media type (text, image, audio, video) rather than a different vendor; per-media-type ABCs and configs live in `core/interfaces/loader/{text,image,audio,video}/`, concrete loaders in `loader/strategies/`. `loader/composite/` holds `AutoLoader` (dispatches to the right strategy by file extension) and `MultiLoader` (loads heterogeneous sources via `AutoLoader`), both re-exported through `loader/__init__.py` like any other provider.

## Credentials

Provider credentials all live in the single file `integrations/credentials.py`. Each is a frozen Pydantic model extending `ProviderCredentials` from `core/credentials.py`. Providers that don't need real credentials (local loaders, local chunkers, FAISS/Chroma, local rerankers/VAD/STT, …) simply don't take a `credentials` constructor argument at all, there is no placeholder credentials type for them.

```python
# integrations/credentials.py
class OpenAICredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr
    organization: str | None = None
```

## The Provider File

Every provider file contains one class that subclasses the domain's ABC from `core/interfaces/`. The ABC itself does not take credentials, a provider that needs them accepts and stores `self._credentials` directly in its own `__init__`. Example:

```python
class OpenAILLMProvider(BaseLLMProvider[OpenAIGenerationConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = credentials if credentials is not None else OpenAICredentials()
    async def agenerate(self, prompt, model, config, tools) -> LLMResponse: ...
    def generate(self, prompt, model, config, tools) -> LLMResponse: ...
    async def stream(self, prompt, model, config) -> AsyncIterator[StreamChunk]: ...
```

## Current Domains

| Domain | Providers | ABC |
|---|---|---|
| `llm/` | openai, anthropic, mistral, ollama, huggingface (each calls its native SDK directly, no LangChain; share `_base.py::NativeLLMProvider`) | `BaseLLMProvider` |
| `embeddings/` | openai, mistral, ollama, huggingface (each calls its native SDK directly, no LangChain; share `_base.py::NativeEmbeddingProvider`) | `BaseEmbeddingProvider` |
| `vad/` | silero, webrtc, pvcobra, ten | `BaseVAD` |
| `ocr/` | tesseract, google_vision, aws_textract, mistral | `BaseOCR` |
| `vector_store/` | chroma, qdrant, pinecone, weaviate, faiss (each calls its native SDK directly, no LangChain) | `BaseVectorStore` |
| `reranking/` | cohere, jina, huggingface, flashrank, voyage | `BaseReranker` |
| `chunking/` | recursive, markdown, html, latex, pdf | `BaseChunker` |
| `speech_to_text/` | whisperx, faster_whisper, deepgram, openai | `BaseSpeechToText` |
| `translation/` | deepl, google_translate, azure | `BaseTranslator` |
| `clustering/` | hdbscan, kmeans, gmm | `ClusteringAlgorithm` |
| `loader/` | strategies: pil (image), pyav (video/audio), soundfile (audio), unstructured (text/composite documents) | `BaseMediaLoader` |
| `image_generation/` | dalle, midjourney, stable_diffusion | `BaseImageGenerator` |
| `classification/` | transformers (zero-shot) | `BaseClassificationProvider` |

## How to Extend

### Add a new provider to an existing domain

```
integrations/<domain>/<new_provider>/
├── __init__.py
├── config.py              # Pydantic model for provider-specific settings
└── <new_provider>.py      # class implements the domain's ABC
```

If the provider needs credentials, add its model to `integrations/credentials.py`.

Steps:
1. Add the credential model to `integrations/credentials.py` if needed
2. Create `integrations/<domain>/<new_provider>/`
3. Define config model (extend domain config or `BaseModel`)
4. Implement the domain's ABC, all abstract methods
5. Inline all conversion logic inside the provider file
6. Use `asyncio.to_thread()` for blocking library calls
7. Translate all failures to `ProviderError` from `core/errors.py`

### Add a new domain

1. First, create the ABC in `core/interfaces/<new_domain>/` (see `core/README.md`)
2. Create `integrations/<new_domain>/`
3. Add the first provider under `integrations/<new_domain>/<first_provider>/`

## Integration Conventions

- All public methods are `async`
- Synchronous library calls are wrapped with `asyncio.to_thread()`
- Provider-specific config is a frozen Pydantic model
- Errors are translated to `core/errors.py` types (`ProviderError`)
- No integration imports from another integration
- No integration imports from `components/`, `pipelines/`, or `agents/`
- `langchain_base.py` (or its native counterpart, `_base.py`) is only used when multiple providers within the *same domain* share a common wrapper pattern, it is optional, and never crosses domain boundaries (an integration never depends on another integration, per the rule above). `reranking` still has a shared `langchain_base.py`. The `llm` and `embeddings` domains have a native, LangChain-free equivalent instead: `integrations/llm/_base.py::NativeLLMProvider` and `integrations/embeddings/_base.py::NativeEmbeddingProvider` factor out the tracing/retry/token-usage (llm) or `TextChunk`/query → `EmbeddingResponse` (embeddings) plumbing that's otherwise identical across every provider in that domain, leaving each provider file with only its vendor-specific client construction, request/response mapping, and (for `llm`) `stream()`. Providers across *different* domains that happen to share a vendor (e.g. `llm/openai` and `embeddings/openai`) still duplicate their client-construction snippet on purpose, that duplication is the intentional cost of each domain staying self-contained, not an oversight. `vector_store` has no shared base: `add`/`delete`/`search`/`search_with_scores` bodies differ too much per vendor's filter DSL for a template to pay for itself
- Each domain's `__init__.py` lazily re-exports its provider classes via a module-level `__getattr__` (PEP 562) over a `_PROVIDERS: dict[class_name, module_path]` map, so callers import `from agent_platform.integrations.<domain> import <ProviderClass>` instead of the nested `<domain>.<provider>.<provider>` path, without eagerly importing every provider's SDK. Add a provider by adding one entry to that map, no separate factory/registry layer
- Provider SDKs are not in the base install, each provider's package(s) are declared as a `<domain>-<provider>` extra in `pyproject.toml`. When adding a new provider, add its package(s) as a new extra and append it to that domain's bundle extra and to `all`
- Providers that make outbound network calls (OCR, translation, speech-to-text, DALL-E/Midjourney image generation, every `llm`/`embeddings`/`vector_store` provider, and the shared `langchain_base.py` for `reranking`) wrap their call in `@error_logged(re_raise=ProviderError, message=...)` outer + `@with_retry()` inner from `core/errors.py`, retry happens first, and if every attempt fails the final exception is logged and translated to `ProviderError`. The `reranking` shared base applies this uniformly to all providers built on it (including the locally-running `huggingface` cross-encoder), since one shared `rerank()` method serves them all. Purely local/offline compute with its own bespoke method (`tesseract`, `whisperx`, `vad`, `clustering`, `chunking`, `flashrank`, `stable_diffusion`, `classification`, and the local `faiss` vector store) is exempt from both, retrying a deterministic local failure wastes CPU/GPU time instead of recovering from it
- Tracing has zero vendor-specific code: `core/tracing.py` only knows OpenTelemetry primitives. `core/tracing.traced_operation_span(operation, **attributes)` (a thin wrapper over `traced_span()` that names the span after `operation` and sets `gen_ai.operation.name` to match) is used unconditionally throughout the codebase (agent loop in `agents/executor.py`, tool calls in `agents/tools/registry.py`, LLM calls in each `integrations/llm/<provider>/provider.py`) to record one uniform set of `gen_ai.*`-attributed spans (keys centralized in `core/tracing.GenAIAttributes`; token usage recorded via `core/tracing.record_token_usage()`), with latency captured automatically as span duration. Without the `agent_platform[tracing]` extra installed, or with nothing configured, these calls are cheap no-ops via OpenTelemetry's default no-op tracer.
  - `configure_tracing()` (called once at process startup, e.g. `api/app.py`) resolves `AGENT_PLATFORM_TRACING` (`auto` | `none` | `console`, default `auto`) into a tracer provider. `auto` exports over OTLP whenever the standard `OTEL_EXPORTER_OTLP_ENDPOINT` (or `_TRACES_ENDPOINT`) env var is set, and is a safe no-op otherwise, so no separate "enable tracing" flag exists beyond pointing that one standard env var at a backend. `OTEL_EXPORTER_OTLP_HEADERS` carries auth; `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` switches transport (needs the `agent_platform[tracing-grpc]` extra), default is `http/protobuf`. `none` force-disables even if the endpoint env var happens to be set; `console` prints spans to stdout for local dev, no network or credentials needed
  - Any OTLP-speaking backend works this way (Honeycomb, Grafana Tempo/Cloud, Datadog, New Relic, SigNoz, LangSmith, Langfuse, a self-hosted OTel Collector fanning out to Jaeger/Zipkin/X-Ray/multiple destinations at once, ...); the vendor-specific endpoint + header recipes live in each vendor's own docs, not in this codebase, e.g.:
    ```bash
    # LangSmith
    OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
    OTEL_EXPORTER_OTLP_HEADERS=x-api-key=<key>,Langsmith-Project=<project>

    # Langfuse
    OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(public_key:secret_key)>
    ```
  - if a tracer provider is already installed by something else (the `opentelemetry-instrument` wrapper, an APM agent, or the host application embedding agent_platform), `configure_tracing()` defers to it instead of overwriting it; `configure_tracing(exporter=...)` also accepts a pre-built exporter directly for backends with no standard OTLP path
