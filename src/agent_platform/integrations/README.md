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
│   ├── mappers.py         # Optional: domain-level shared conversion functions, when
│   │                       #   multiple providers (or langchain_base.py) share them
│   └── <provider>/
│       ├── __init__.py
│       ├── config.py      # Provider-specific Pydantic config
│       ├── mappers.py     # Optional: free functions converting vendor types <-> core
│       │                   #   schemas, when the provider has more than a couple of them
│       └── <provider>.py  # Concrete class implementing core/interfaces ABC
└── …
```

### Provider and mapper files

Every provider file contains one class that owns client construction, retry/tracing
orchestration, and the domain ABC's public methods. When that class also carries
non-trivial vendor-type ↔ core-schema conversion logic (message/request building,
response parsing), those conversions live as free functions in a sibling `mappers.py`
rather than inline in the provider file, so the provider file stays focused on
orchestration. `mappers.py` holds only pure functions (no `self`, no network/client
calls) and imports the same things the provider file would (vendor SDK types for
annotations, `core/schemas`, `core/interfaces`); it never imports its sibling
`<provider>.py`, since the provider file imports from it. This mirrors how the domain
already separates cross-provider shared conversions into `ocr/sources.py`,
`speech_to_text/language.py`, and `reranking/scoring.py`, just at per-provider granularity
when the conversions are provider-specific rather than shared. Providers with only thin,
inline dict-literal conversions (a couple of lines, no free functions) don't get a
`mappers.py`, adding one would be ceremony without removing real duplication.

**Exception:** `loader/` uses `strategies/` instead of `<provider>/` because each strategy handles a different media type (text, image, audio, video) rather than a different vendor; per-media-type ABCs and configs live in `core/interfaces/loader/{text,image,audio,video}/`, concrete loaders in `loader/strategies/`. `loader/composite/` holds `AutoLoader` (dispatches to the right strategy by file extension) and `MultiLoader` (loads heterogeneous sources via `AutoLoader`), both re-exported through `loader/__init__.py` like any other provider.

## Credentials and client options

Provider credentials all live in the single file `integrations/credentials.py`. Each is a frozen Pydantic model extending `Credentials` from `core/credentials.py` and holds secrets only (`api_key` and friends). Providers that don't need real secrets (local loaders, local chunkers, FAISS/Chroma, local rerankers/VAD/STT, …) simply don't take a `credentials` constructor argument at all, there is no placeholder credentials type for them.

```python
# integrations/credentials.py
class OpenAICredentials(Credentials, frozen=True):
    api_key: SecretStr
    organization: str | None = None
```

Non-secret, construction-time transport settings (`base_url`, `timeout`, `max_retries`) are a separate, single `ClientOptions` type (`core/credentials.py`), not subclassed per provider since these fields mean the same thing for every vendor. Only providers that are actually network-bound and read these fields (currently `llm`, `embeddings`, `moderation`) accept a `client_options` constructor argument; `timeout`/`max_retries` can still be overridden per call via the domain's own config (`resolve_timeout`/`resolve_max_retries` prefer the per-call value, falling back to `client_options`).

## The Provider File

Every provider file contains one class that subclasses the domain's ABC from `core/interfaces/`. The ABC itself does not take credentials, a provider that needs them accepts and stores `self._credentials` (and, for network-bound providers, `self._client_options`) directly in its own `__init__`. Example:

```python
class OpenAILLMProvider(BaseLLMProvider[OpenAIGenerationConfig]):
    def __init__(
        self,
        credentials: OpenAICredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)
        self._client_options = resolve_client_options(client_options)
    async def agenerate(self, prompt, model, config, tools) -> LLMResponse: ...
    def generate(self, prompt, model, config, tools) -> LLMResponse: ...
    async def stream(self, prompt, model, config) -> AsyncIterator[StreamChunk]: ...
```

## Current Domains

| Domain | Providers | ABC |
|---|---|---|
| `llm/` | openai, anthropic, mistral, ollama, huggingface, google (each calls its native SDK directly, no LangChain; share `_base.py::NativeLLMProvider`). `openai`/`anthropic` also implement `core/interfaces/llm/batch.py::BaseBatchLLMProvider` for async batch inference (submit/poll/fetch), a separate mixin since Mistral/Ollama/HuggingFace/Google have no first-class batch API to implement it against. `stream()` across all 5 providers accepts a `tools` param and can yield `ToolCallDelta` fragments (`core/interfaces/llm/response.py`); OpenAI/Anthropic stream real per-fragment tool-call deltas natively, the rest emit one whole delta per call | `BaseLLMProvider` |
| `embeddings/` | openai, mistral, ollama, huggingface, google (each calls its native SDK directly, no LangChain; share `_base.py::NativeEmbeddingProvider`); `bm25` implements the separate `core/interfaces/embeddings/sparse.py::BaseSparseEmbeddingProvider` ABC (local feature-hashing term-frequency, no vendor call, no corpus-fit step) | `BaseEmbeddingProvider` (`bm25`: `BaseSparseEmbeddingProvider`) |
| `vad/` | silero, webrtc, pvcobra, ten | `BaseVAD` |
| `ocr/` | tesseract, google_vision, aws_textract, mistral | `BaseOCR` |
| `vector_store/` | chroma, qdrant, pinecone, weaviate, faiss (each calls its native SDK directly, no LangChain). `qdrant` additionally implements `add_hybrid`/`search_hybrid` (dense+sparse, fused server-side via `models.FusionQuery(fusion=Fusion.RRF)` over named `"dense"`/`"sparse"` vectors, paired with `embeddings/bm25` for the sparse side); Pinecone hybrid support is a deliberate non-goal for now since Pinecone's hybrid model expects a client-side-fitted sparse encoder, a different (stateful) shape than the stateless per-text embedding pattern used everywhere else in this codebase | `BaseVectorStore` |
| `reranking/` | cohere, jina, voyage (each calls its native SDK or REST endpoint directly, no LangChain; share `_base.py::NativeReranker`), huggingface, flashrank (local, no shared base, see below) | `BaseReranker` |
| `chunking/` | recursive, markdown, html, latex, pdf | `BaseChunker` |
| `speech_to_text/` | whisperx, faster_whisper, deepgram, openai (whisperx/faster_whisper/openai share `_base.py::buffered_stream()` for client-side stream windowing; deepgram streams natively instead) | `BaseSpeechToText` |
| `translation/` | deepl, google_translate, azure (share `_base.py::NativeTranslator`) | `BaseTranslator` |
| `clustering/` | hdbscan, kmeans, gmm | `ClusteringAlgorithm` |
| `loader/` | strategies: pil (image), pyav (video/audio), soundfile (audio), unstructured (text/composite documents) | `BaseMediaLoader` |
| `image_generation/` | dalle, midjourney, stable_diffusion | `BaseImageGenerator` |
| `classification/` | transformers (zero-shot) | `BaseClassificationProvider` |
| `moderation/` | openai (`omni-moderation-latest`, network-bound, shares `_base.py::NativeModerationProvider` with the same dual-client shape as `reranking/_base.py::NativeReranker`), local (offline keyword/regex rule set, no credentials) | `BaseModerationProvider` |
| `dataset/` | huggingface (wraps the `datasets` library's `load_dataset`; `map`/`filter`/`batch` on the returned splits delegate straight to the underlying `datasets.Dataset`'s own methods, so its Arrow storage, disk-cached `map()` fingerprinting, and streaming/`IterableDataset` support carry through unchanged, see `huggingface/mappers.py::resolve_splits` for native vs. ratio-based train/test/eval split resolution) | `BaseDatasetProvider` |

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
5. Put conversion logic in a sibling `mappers.py` if there's more than a couple of
   one-line conversions, otherwise inline it in the provider file
6. Use `asyncio.to_thread()` for blocking library calls
7. Translate all failures to `ProviderError` from `core/errors.py`

### Add a new domain

1. First, create the ABC in `core/interfaces/<new_domain>/` (see `core/README.md`)
2. Create `integrations/<new_domain>/`
3. Add the first provider under `integrations/<new_domain>/<first_provider>/`

## Integration Conventions

- Every provider method that does real work (a network call or blocking local compute) ships in a sync/async pair: the sync method keeps the plain name (`rerank`, `translate`, `generate`, `embed_document`, `detect`, `moderate`), the async twin is the same name with an `a` prefix (`arerank`, `atranslate`, `agenerate`, `aembed_document`, `adetect`, `amoderate`). This mirrors how modern provider SDKs (OpenAI, Anthropic, Cohere, Voyage, Qdrant, …) ship distinct sync and async clients rather than forcing every caller onto one or the other. Streaming methods (`stream()`) are the one exception and stay async-only, since a sync streaming generator has no equivalent caller pattern in this codebase; `llm`'s `stream()` additionally takes an optional `tools` param and can yield `ToolCallDelta` fragments on `StreamChunk`, at whatever granularity each vendor's streaming API actually provides. `llm`'s `submit_batch`/`get_batch_status`/`fetch_batch_results` (from `BaseBatchLLMProvider`, implemented by `openai`/`anthropic` only) are async-only by nature, batch jobs have no synchronous equivalent. Domains where every provider is purely local/offline compute with no vendor sync/async split to mirror (`chunking`, `clustering`, `classification`, `loader`) are not held to this pairing; `embeddings/bm25`'s `embed_sparse` (from the separate `BaseSparseEmbeddingProvider` ABC) follows this same local-compute exemption
- Synchronous library calls are wrapped with `asyncio.to_thread()` (either per call, or once in a domain's `_base.py` when the async method is templated as `asyncio.to_thread(self.<sync_method>, ...)`, see `translation/_base.py::NativeTranslator`)
- Provider-specific config is a frozen Pydantic model, and stays fully optional: every field has a default, so `config=None` (the ABC's default) always works. A provider never introduces a required config field to smuggle in a value the ABC's method signature should carry directly instead (see `core/README.md`'s "Required parameters never live on config")
- Errors are translated to `core/errors.py` types (`ProviderError`)
- No integration imports from another integration
- No integration imports from `components/`, `pipelines/`, or `agents/`
- `langchain_base.py` (or its native counterpart, `_base.py`) is only used when multiple providers within the *same domain* share a common wrapper pattern, it is optional, and never crosses domain boundaries (an integration never depends on another integration, per the rule above). `chunking` is the only domain left with a `langchain_base.py`, wrapping `langchain-text-splitters` directly, since that's a genuinely useful splitting library, not a thin pass-through causing gaps. Every other domain's `_base.py` calls vendor SDKs or REST endpoints natively; `reranking` was the last domain to drop LangChain as an intermediate (its wrapper wasn't "close to a whole cover", the per-provider config files carried multiple `# NOTE: library gap` comments documenting fields the LangChain wrapper couldn't forward, and `huggingface` had already been forced to reimplement LangChain's own `CrossEncoderReranker` from scratch after an upstream breaking move). `integrations/llm/_base.py::NativeLLMProvider` and `integrations/embeddings/_base.py::NativeEmbeddingProvider` factor out the tracing/retry/token-usage (llm) or `TextChunk`/query → `EmbeddingResponse` (embeddings) plumbing that's otherwise identical across every provider in that domain, leaving each provider file with only its vendor-specific client construction, request/response mapping, and (for `llm`) `stream()`. `integrations/reranking/_base.py::NativeReranker` factors the same shape for the network-bound reranking providers (`cohere`, `jina`, `voyage`): empty-input short-circuit, retry/error-translation, and result → `TextChunk` mapping, leaving each provider only its client construction and request/result field access. `huggingface` and `flashrank` (local/offline reranking) don't use it, same reasoning as the local-compute exemption below. `integrations/moderation/_base.py::NativeModerationProvider` follows the same dual-client shape as `NativeReranker` (minus the batch/index-mapping machinery, since moderation is one-text-in-one-result-out): `openai` uses it since its moderation endpoint is a real network call on both `OpenAI`/`AsyncOpenAI` clients; `local` (offline keyword/regex rules) doesn't, same local-compute exemption as `huggingface`/`flashrank` reranking. `integrations/translation/_base.py::NativeTranslator` factors the sync/async split itself: none of `deepl`/`google_translate`/`azure`'s SDKs ship a native async client, so `atranslate` is templated once as `asyncio.to_thread(self.translate, ...)` with retry/error-translation applied, and each provider only owns client construction and the blocking `_invoke()`. Providers across *different* domains that happen to share a vendor (e.g. `llm/openai` and `embeddings/openai`) still duplicate their client-construction snippet on purpose, that duplication is the intentional cost of each domain staying self-contained, not an oversight. `vector_store` has no shared base: `add`/`delete`/`search`/`search_with_scores` bodies differ too much per vendor's filter DSL for a template to pay for itself; `add_hybrid`/`search_hybrid` (dense+sparse) are concrete methods on `BaseVectorStore` that default to `NotImplementedError`, following the same optional-override shape rather than a templated base, since only `qdrant` implements them today. `speech_to_text` has a narrower shared helper instead of a base class: `integrations/speech_to_text/_base.py::buffered_stream()` is a free async generator (not a `BaseSpeechToText` subclass, since `_default_config()`/credentials/client construction differ too much across its providers to templatize) that factors the client-side windowing loop (accumulate frames until `config.min_duration_ms` is reached, transcribe the window, flush the remainder) shared verbatim by `whisperx`, `faster_whisper`, and `openai`'s `stream()`. `deepgram` does not use it: its `stream()` speaks a real low-latency socket protocol rather than batching client-side, so forcing it onto the same helper would be templating code that isn't actually identical, the same reasoning that keeps `vector_store` base-free. `ocr` and `image_generation` also have no shared base: each ABC in those domains has a single method (no sync/async or batch/stream pair to collapse) and vendor response shapes differ enough (e.g. OCR bounding boxes present in some providers, absent in others) that a template would cost more abstraction than the duplication it removes
- Each domain's `__init__.py` lazily re-exports its provider classes via a module-level `__getattr__` (PEP 562) over a `_PROVIDERS: dict[class_name, module_path]` map, so callers import `from agent_platform.integrations.<domain> import <ProviderClass>` instead of the nested `<domain>.<provider>.<provider>` path, without eagerly importing every provider's SDK. Add a provider by adding one entry to that map, no separate factory/registry layer
- Provider SDKs are not in the base install, each provider's package(s) are declared as a `<domain>-<provider>` extra in `pyproject.toml`. When adding a new provider, add its package(s) as a new extra and append it to that domain's bundle extra and to `all`
- Providers that make outbound network calls (OCR, translation, speech-to-text, DALL-E/Midjourney image generation, every `llm`/`embeddings`/`vector_store` provider, `reranking`'s `cohere`/`jina`/`voyage` via `NativeReranker`, and `moderation`'s `openai` via `NativeModerationProvider`) wrap their async call in `@error_logged(re_raise=ProviderError, message=...)` outer (`core/errors.py`) + `@with_retry()` inner (`core/retry.py`), retry happens first, and if every attempt fails the final exception is logged and translated to `ProviderError`; the sync twin of a sync/async pair is not wrapped this way (matches `llm`/`embeddings`, which established the pairing first — see the sync/async convention above). `llm`'s `submit_batch`/`get_batch_status`/`fetch_batch_results` are wrapped the same way. Purely local/offline compute with its own bespoke method (`tesseract`, `whisperx`, `vad`, `clustering`, `chunking`, `reranking`'s `flashrank`/`huggingface`, `stable_diffusion`, `classification`, `moderation`'s `local`, `embeddings`'s `bm25`, and the local `faiss` vector store) is exempt from both, retrying a deterministic local failure wastes CPU/GPU time instead of recovering from it
- Tracing has zero vendor-specific code: `core/tracing.py` only knows OpenTelemetry primitives, paired with `core/genai_tracing.py` for the GenAI-specific layer on top of it. `core/genai_tracing.traced_operation_span(operation, attributes=None)` (a thin wrapper over `core/tracing.traced_span()` that names the span after `operation` and sets `gen_ai.operation.name` to match) is used unconditionally throughout the codebase (agent loop in `agents/executor.py`, tool calls in `agents/tools/registry.py`, LLM calls in each `integrations/llm/<provider>/provider.py`) to record one uniform set of `gen_ai.*`-attributed spans (keys centralized in `core/genai_tracing.GenAIAttributes`; token usage recorded via `core/genai_tracing.record_token_usage()`), with latency captured automatically as span duration. Without the `agent_platform[tracing]` extra installed, or with nothing configured, these calls are cheap no-ops via OpenTelemetry's default no-op tracer.
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
