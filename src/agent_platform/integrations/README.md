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

Non-secret, construction-time transport settings (`base_url`, `timeout`, `max_retries`) are a separate, single `ClientOptions` type (`core/credentials.py`), not subclassed per provider since these fields mean the same thing for every vendor. Only providers that are actually network-bound and read these fields (currently `llm`, `embeddings`, `moderation`) accept a `client_options` constructor argument. The per-call side of the same two fields (`timeout`, `max_retries`) lives on `core/config.py::RequestOptions`, a `ProviderConfig` mixin that a domain's own config inherits alongside `ModelConfig` where relevant (`GenerationConfig(RequestOptions, ModelConfig)`, `EmbeddingConfig(RequestOptions, ModelConfig)`, `ModerationConfig(RequestOptions)`); `resolve_timeout`/`resolve_max_retries` prefer the per-call `RequestOptions` value, falling back to `client_options`. Domains not yet wired onto `client_options`/`RequestOptions` (`reranking`, `translation`, `ocr`, `speech_to_text`, `vector_store`, `image_generation`) still only support the `credentials`-level defaults baked into each vendor SDK call, no per-call timeout/retry override yet.

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

### LLM error contract

`NativeLLMProvider` translates every vendor failure into a `ProviderError` with a stable `code`. `generate`, `agenerate` and `stream` raise only these (plus `ConfigError` for a bad config, `MissingCredentialError` for a missing key, and `asyncio.CancelledError`, which is never translated). `stream` translates a failure before or after the first chunk.

| code | error type | `retryable` |
|---|---|---|
| `llm_context_window` | `LLMContextWindowError` | no |
| `llm_rate_limited` | `LLMRateLimitError`, with `retry_after` when the vendor sent `Retry-After` | yes |
| `llm_timeout` | `LLMTimeoutError` | yes |
| `provider_unavailable` | `ProviderError` (5xx, 408, 409, connection error, a vendor error event such as `overloaded_error`) | yes |
| `provider_bad_request` | `ProviderError` (any other failure) | no |

`retryable` says the same call may succeed later: `retry_async`/`retry_sync` stop at once on `False`, and a `CircuitBreaker` only counts retryable errors. `retry_after` is the seconds the vendor asked to wait; the retry loop waits at least that long, capped at 30 s.

A new LLM provider implements the abstract `_stream` (not `stream`, which the base wraps to translate errors) and, where its SDK differs from the defaults, the hooks `_status_code`, `_retry_after`, `_error_kinds`, `_is_context_window`, `_is_timeout` and `_is_transient`. It sets `_api_key_env` to the name of its key's env var, or leaves it `None` when it needs no key; `build_llm_with_fallback` warns at startup when that key is missing.

### Non-LLM error contract

The `@with_retry(translate=...)` methods of the other network providers (embeddings, moderation, reranking, translation, OCR, speech-to-text, vector stores, image generation, batch reads) follow the same rule through one shared classifier, `integrations/sdk_errors.py::translate_sdk_error(exc, *, operation, transient_types, retry_statuses)`. Each failed attempt is translated before `with_retry` decides:

- The HTTP status is read duck-typed from the error or its cause chain (`status_code`, `status`, `http_status`, `code`, `response.status_code`, botocore's `ResponseMetadata`). A 408, 409, 429 or 5xx, an SDK `should_retry`, a timeout or a connection error is retryable: `ProviderError` code `provider_unavailable`, retried up to 3 attempts.
- An error with a non-transient status (a 400) is `provider_bad_request`, not retryable, so it costs one call.
- An error with no status (a local bug such as a `ValueError`) is a `ProviderError` with no code, not retryable; the API serves it as 500 `internal_error`.
- The message starts with the operation text (the method's `error_logged` message). A raw SDK error never reaches the caller, and wrappers inside the retried call no longer swallow it. The final failure is logged once at the API boundary, not by the provider.
- Image generation passes `retry_statuses={429, 503}`: only a rejection before the work started is retried, so a timeout or a 500 (the vendor may have done and billed the work) fails after one call. A local `ValueError` such as dall-e-3 `validate_size` costs no SDK call. `submit_batch` (OpenAI, Anthropic) is never retried; `get_batch_status` and `fetch_batch_results` are.

The platform's `with_retry` is the only retry layer: SDK retries are off on DALL-E, OpenAI Whisper, Deepgram, Textract, OpenAI embeddings and moderation, Cohere, Voyage, Azure translation (`retry_total=0`), Google Vision (`retry=None`) and Pinecone's asyncio client. So `ClientOptions.max_retries` no longer reaches these SDKs; the platform keeps 3 attempts. DeepL's setting is process-wide: `deepl.http_client.max_network_retries = 0` is set at import and applies to every DeepL client in the process. Known limit: weaviate-client 4.16 retries a gRPC search that fails with `UNAVAILABLE` on its own and has no setting to turn that off.

DALL-E, OpenAI Whisper and Deepgram default to a 300 s timeout when none is configured.

### LLM config rules

- Every `llm` provider takes `model` and `options` beside `credentials` and `client_options`. `model` is the model the provider is bound to, taken from the `"<provider>:<model>"` string. No provider config has a default model.
- `options` come from `Settings.llm_options` (a JSON object keyed by provider slug, env `AGENT_PLATFORM_LLM_OPTIONS`). The provider validates them against its own config class when it is built; an unknown key, `model` or a bad value raises `ConfigError` naming the field, never the value. `build_llm_with_fallback` validates every entry, whether or not the chain uses it. An options entry or a chain entry whose provider extra is missing raises `ConfigError` naming the slug and the `llm-<slug>` extra to install.
- On every call (`generate`, `agenerate`, `stream`, batch submit), any config, including the base `GenerationConfig`, is resolved into the provider's config class: bound model, then options, then the fields the call config sets explicitly to a non-`None` value. From a config of another provider class, only the base fields are read. An empty model on both sides raises `ConfigError`.
- `reasoning_effort` and `parallel_tool_calls` are generic fields on `GenerationConfig`. Each adapter maps them to its native control, or logs a warning and skips the field when it has none. A provider-specific field in the resolved config wins over the generic one.
- Strict JSON schema is always requested; the schema name comes from the schema's `title`.

## Current Domains

| Domain | Providers | ABC |
|---|---|---|
| `llm/` | openai, anthropic, mistral, ollama, huggingface, google (each calls its native SDK directly, no LangChain; share `_base.py::NativeLLMProvider`). `openai`/`anthropic` also implement `core/interfaces/llm/batch.py::BaseBatchLLMProvider` for async batch inference (submit/poll/fetch), a separate mixin since Mistral/Ollama/HuggingFace/Google have no first-class batch API to implement it against. `stream()` across all 5 providers accepts a `tools` param and can yield `ToolCallDelta` fragments (`core/interfaces/llm/response.py`); OpenAI/Anthropic stream real per-fragment tool-call deltas natively, the rest emit one whole delta per call | `BaseLLMProvider` |
| `embeddings/` | openai, mistral, ollama, huggingface, google (each calls its native SDK directly, no LangChain; share `_base.py::NativeEmbeddingProvider`); `bm25` implements the separate `core/interfaces/embeddings/sparse.py::BaseSparseEmbeddingProvider` ABC (local feature-hashing term-frequency, no vendor call, no corpus-fit step) | `BaseEmbeddingProvider` (`bm25`: `BaseSparseEmbeddingProvider`) |
| `vad/` | silero, webrtc, pvcobra, ten | `BaseVAD` |
| `ocr/` | tesseract, google_vision, aws_textract, mistral | `BaseOCR` |
| `vector_store/` | chroma, qdrant, pinecone, weaviate, faiss (each calls its native SDK directly, no LangChain). `qdrant` additionally implements `add_hybrid`/`search_hybrid` (dense+sparse, fused server-side via `models.FusionQuery(fusion=Fusion.RRF)` over named `"dense"`/`"sparse"` vectors, paired with `embeddings/bm25` for the sparse side); Pinecone hybrid support is a deliberate non-goal for now since Pinecone's hybrid model expects a client-side-fitted sparse encoder, a different (stateful) shape than the stateless per-text embedding pattern used everywhere else in this codebase. **Namespace isolation**: only `pinecone` (namespace) and `weaviate` (tenant) honor `VectorStoreConfig.namespace`, and on both an empty string `""` is the shared default partition (Pinecone's default namespace is `""`; Weaviate treats `""` as unset), so it gives no isolation; `qdrant`, `chroma` and `faiss` fail closed, raising `ConfigError` from every operation when it is set (`_base.py::reject_namespace`), so tenants never silently share a collection. **FAISS on disk**: `index_path` holds `docstore.json` (versioned JSON: chunks via `model_dump`/`model_validate`, internal ids, `next_id`, dimension, metric, and the name, sha256 and `ntotal` of its index) and the content-addressed native index `index-<sha256>.faiss` it names. Each add/delete writes the new index under its own name, then replaces `docstore.json`, the single commit step; every file goes through an fsynced temp file in the same directory (mode from the umask) and a directory fsync. A failure or crash at any point leaves the previous pair loadable plus at most one orphan index file and one temp file, which the next save removes under the path lock (only files matching its own naming; other files in the directory are left alone). Cleanup and the directory fsync after the commit are best-effort: a failure there logs a warning and the call still succeeds. A docstore whose index is missing, mismatched (sha256, `ntotal == len(docs)`, every id below `next_id`), outside the `index-<sha256>.faiss` pattern or corrupt raises `ConfigError`; re-ingest. A legacy `docstore.pkl` is refused with `ConfigError` and never unpickled. A chunk id repeated within one `add` batch is rejected with `ValidationError` before any change (as Chroma does); re-adding an id already stored overwrites it. Reloading from JSON changes metadata types (datetime → str, tuple → list), so a filter built from the original Python values may stop matching after a restart; only a persisted store requires JSON-serializable metadata. In-memory state is keyed by resolved `index_path`. Each operation runs as one `asyncio.to_thread` call holding that path's per-instance `threading.Lock`, so a cancelled caller cannot release the lock while its thread still works. Known limits: searches are serialized with writes on the same path (no reader-writer lock), and instances or processes sharing one path are not coordinated | `BaseVectorStore` |
| `reranking/` | cohere, jina, voyage (each calls its native SDK or REST endpoint directly, no LangChain; share `_base.py::NativeReranker`), huggingface, flashrank (local, no shared base, see below) | `BaseReranker` |
| `chunking/` | recursive, markdown, html, latex, pdf | `BaseChunker` |
| `speech_to_text/` | whisperx, faster_whisper, deepgram, openai (whisperx/faster_whisper/openai share `_base.py::buffered_stream()` for client-side stream windowing; deepgram streams natively instead) | `BaseSpeechToText` |
| `translation/` | deepl, google_translate, azure (share `_base.py::NativeTranslator`) | `BaseTranslator` |
| `clustering/` | hdbscan, kmeans, gmm | `ClusteringAlgorithm` |
| `loader/` | strategies: pil (image), pyav (video/audio), soundfile (audio), unstructured (text/composite documents) | `BaseMediaLoader` |
| `image_generation/` | dalle, stable_diffusion | `BaseImageGenerator` |
| `classification/` | transformers (zero-shot) | `BaseClassificationProvider` |
| `moderation/` | openai (`omni-moderation-latest`, network-bound, shares `_base.py::NativeModerationProvider` with the same dual-client shape as `reranking/_base.py::NativeReranker`), local (offline keyword/regex rule set, no credentials) | `BaseModerationProvider` |
| `dataset/` | huggingface (wraps the `datasets` library's `load_dataset`; `map`/`filter`/`batch` on the returned splits delegate straight to the underlying `datasets.Dataset`'s own methods, so its Arrow storage, disk-cached `map()` fingerprinting, and streaming/`IterableDataset` support carry through unchanged, see `huggingface/mappers.py::resolve_splits` for native vs. ratio-based train/test/eval split resolution) | `BaseDatasetProvider` |
| `tracing/` | langfuse (`LangfuseTracer`, through the official Langfuse SDK; on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, a silent no-op otherwise). Unlike every other domain this one has no credentials model and is not built per call site: exactly one instance is installed process-wide via `core/observability.set_tracer()` at startup, and `core`'s `NoOpTracer` stands in when none is | `BaseTracer` |
| `mcp/` | stdio (`StdioMCPClient`, wraps the official `mcp` SDK's `stdio_client`/`ClientSession` to talk to a server launched as a local subprocess). The odd one out among all domains above: `BaseMCPClient` owns connection lifecycle (`connect()`/`aclose()`, also usable as `async with client:`) since an MCP server is a stateful session, not a stateless per-call provider; `agents/tools/mcp/` (not `integrations/`) is where a connected client's tools become `Tool`s | `BaseMCPClient` |

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

- Every provider method that does real work (a network call or blocking local compute) ships in a sync/async pair: the sync method keeps the plain name (`rerank`, `translate`, `generate`, `embed_document`, `detect`, `moderate`), the async twin is the same name with an `a` prefix (`arerank`, `atranslate`, `agenerate`, `aembed_document`, `adetect`, `amoderate`). This mirrors how modern provider SDKs (OpenAI, Anthropic, Cohere, Voyage, Qdrant, …) ship distinct sync and async clients rather than forcing every caller onto one or the other. Streaming methods (`stream()`) are the one exception and stay async-only, since a sync streaming generator has no equivalent caller pattern in this codebase; `llm`'s `stream()` additionally takes an optional `tools` param and can yield `ToolCallDelta` fragments on `StreamChunk`, at whatever granularity each vendor's streaming API actually provides. `llm`'s `submit_batch`/`get_batch_status`/`fetch_batch_results` (from `BaseBatchLLMProvider`, implemented by `openai`/`anthropic` only) are async-only by nature, batch jobs have no synchronous equivalent. Domains where every provider is purely local/offline compute with no vendor sync/async split to mirror (`chunking`, `clustering`, `classification`, `loader`) are not held to this pairing; `embeddings/bm25`'s `embed_sparse` (from the separate `BaseSparseEmbeddingProvider` ABC) is exempt for the same reason
- Synchronous library calls are wrapped with `asyncio.to_thread()` (either per call, or once in a domain's `_base.py` when the async method is templated as `asyncio.to_thread(self.<sync_method>, ...)`, see `translation/_base.py::NativeTranslator`)
- Provider-specific config is a frozen Pydantic model, and stays fully optional: every field has a default, so `config=None` (the ABC's default) always works. A provider never introduces a required config field to smuggle in a value the ABC's method signature should carry directly instead (see `core/README.md`'s "Required parameters never live on config")
- Errors are translated to `core/errors.py` types (`ProviderError`)
- No integration imports from another integration
- No integration imports from `components/`, `pipelines/`, `agents/`, `workflows/` or `evals/` (`core/`, `utils/` and `audio/` are fine)
- `langchain_base.py` (or its native counterpart, `_base.py`) is only used when multiple providers within the *same domain* share a common wrapper pattern, it is optional, and never crosses domain boundaries (an integration never depends on another integration, per the rule above). `chunking` is the only domain left with a `langchain_base.py`, wrapping `langchain-text-splitters` directly, since that's a genuinely useful splitting library, not a thin pass-through causing gaps. Every other domain's `_base.py` calls vendor SDKs or REST endpoints natively. `integrations/llm/_base.py::NativeLLMProvider` and `integrations/embeddings/_base.py::NativeEmbeddingProvider` factor out the tracing/retry/token-usage (llm) or `TextChunk`/query → `EmbeddingResponse` (embeddings) plumbing that's otherwise identical across every provider in that domain, leaving each provider file with only its vendor-specific client construction, request/response mapping, and (for `llm`) `stream()`. `integrations/reranking/_base.py::NativeReranker` factors the same shape for the network-bound reranking providers (`cohere`, `jina`, `voyage`): empty-input short-circuit, retry/error-translation, and result → `TextChunk` mapping, leaving each provider only its client construction and request/result field access. `huggingface` and `flashrank` (local/offline reranking) don't use it since they have no network call to wrap. `integrations/moderation/_base.py::NativeModerationProvider` follows the same dual-client shape as `NativeReranker` (minus the batch/index-mapping machinery, since moderation is one-text-in-one-result-out): `openai` uses it since its moderation endpoint is a real network call on both `OpenAI`/`AsyncOpenAI` clients; `local` (offline keyword/regex rules) doesn't. `integrations/translation/_base.py::NativeTranslator` factors the sync/async split itself: none of `deepl`/`google_translate`/`azure`'s SDKs ship a native async client, so `atranslate` is templated once as `asyncio.to_thread(self.translate, ...)` with retry/error-translation applied, and each provider only owns client construction and the blocking `_invoke()`. Providers across *different* domains that happen to share a vendor (e.g. `llm/openai` and `embeddings/openai`) still duplicate their client-construction snippet on purpose, that duplication is the intentional cost of each domain staying self-contained, not an oversight. `vector_store` has no shared base class: `add`/`delete`/`search`/`search_with_scores` bodies differ too much per vendor's filter DSL for a template to pay for itself; its `_base.py` only holds the free function `reject_namespace()` that the stores without namespace isolation call first in every operation; `add_hybrid`/`search_hybrid` (dense+sparse) are concrete methods on `BaseVectorStore` that default to `NotImplementedError`, since only `qdrant` implements them today. `speech_to_text` has a narrower shared helper instead of a base class: `integrations/speech_to_text/_base.py::buffered_stream()` is a free async generator (not a `BaseSpeechToText` subclass, since `_default_config()`/credentials/client construction differ too much across its providers to templatize) that factors the client-side windowing loop (accumulate frames until `config.min_duration_ms` is reached, transcribe the window, flush the remainder) shared verbatim by `whisperx`, `faster_whisper`, and `openai`'s `stream()`. `deepgram` does not use it: its `stream()` speaks a real low-latency socket protocol rather than batching client-side, so forcing it onto the same helper would be templating code that isn't actually identical. `ocr` and `image_generation` also have no shared base: each ABC in those domains has a single method (no sync/async or batch/stream pair to collapse) and vendor response shapes differ enough (e.g. OCR bounding boxes present in some providers, absent in others) that a template would cost more abstraction than the duplication it removes
- Each domain's `__init__.py` lazily re-exports its provider classes via a module-level `__getattr__` (PEP 562) over a `_PROVIDERS: dict[class_name, module_path]` map, so callers import `from agent_platform.integrations.<domain> import <ProviderClass>` instead of the nested `<domain>.<provider>.<provider>` path, without eagerly importing every provider's SDK. Add a provider by adding one entry to that map, no separate factory/registry layer
- Provider SDKs are not in the base install, each provider's package(s) are declared as a `<domain>-<provider>` extra in `pyproject.toml`. When adding a new provider, add its package(s) as a new extra and append it to that domain's bundle extra and to `all`
- Providers that make outbound network calls (OCR, translation, speech-to-text, DALL-E image generation, every `llm`/`embeddings`/`vector_store` provider, `reranking`'s `cohere`/`jina`/`voyage` via `NativeReranker`, and `moderation`'s `openai` via `NativeModerationProvider`) wrap their async call in `@error_logged(re_raise=ProviderError, message=...)` outer (`core/errors.py`) + `@with_retry()` inner (`core/retry.py`), retry happens first, and if every attempt fails the final exception is logged and translated to `ProviderError`; the sync twin of a sync/async pair is not wrapped this way. `llm`'s `generate`/`agenerate` are the exception: `NativeLLMProvider` runs one attempt per call of `core/retry.py`'s `retry_sync`/`retry_async` (`max_retries + 1` attempts), and inside each attempt translates the vendor exception to `ProviderError(retryable=...)`, with `retryable` true only for a timeout, a connection error, or HTTP 408, 409, 429 and 5xx (each provider overrides `_is_transient` for its SDK's types). A permanent error such as a 400 or 401 therefore fails after one call. The base logs the final failure once. The SDK's own retries are off on the clients `generate`/`agenerate` use; only the client `stream()` uses keeps them. Every `llm` client gets a 120 s timeout unless `config.timeout` or `ClientOptions.timeout` sets one. `llm`'s `get_batch_status`/`fetch_batch_results` are wrapped the same way; `submit_batch` is not retried (a retry after a timeout could create a second billed batch). The other providers use `@with_retry(translate=...)` as described in "Non-LLM error contract". Purely local/offline compute with its own bespoke method (`tesseract`, `whisperx`, `vad`, `clustering`, `chunking`, `reranking`'s `flashrank`/`huggingface`, `stable_diffusion`, `classification`, `moderation`'s `local`, `embeddings`'s `bm25`, and the local `faiss` vector store) is exempt from both, retrying a deterministic local failure wastes CPU/GPU time instead of recovering from it
- Tracing is a normal domain: `core/interfaces/tracing/` owns the `BaseTracer`/`Span` contract and the `NoOpTracer` default, `integrations/tracing/langfuse/` owns the only backend today. No layer above `integrations/` imports a tracing SDK (an import-linter contract enforces it for `core`), and `langfuse` is not a base dependency. Call sites use `core/observability.observe(<Observation>)` unconditionally (agent loop in `agents/executor.py`, tool calls in `agents/tools/registry.py`, LLM calls in each `integrations/llm/<provider>/provider.py`), describing what happened in the platform's own vocabulary, with latency captured automatically as span duration. With no tracer installed, or without the `agent_platform[tracing]` extra, these calls are cheap no-ops via `NoOpTracer`.
  - `integrations/tracing/langfuse/mapping.py` is the single place that vocabulary becomes Langfuse's: the operation name picks the observation type (`chat` is a `generation`, `execute_tool` a `tool`, `invoke_agent` an `agent`, `guardrail` a `guardrail`, `pipeline` a `chain`, anything else a `span`); on a `generation` the model, token usage and cost go to the SDK's own fields; tool payload content goes to `input`/`output`; every other platform key (the size, and the digest when capture is on) lands in `metadata` under its platform name. When the SDK or Langfuse renames a field, this module changes and no call site does.
  - `config/container.py::build_tracer(settings)` resolves `AGENT_PLATFORM_TRACING_PROVIDER` (`langfuse`, the default, or `none` to disable outright) against `integrations/tracing`'s `PROVIDER_ALIASES`, exactly like any other domain, and `core/observability.set_tracer()` installs the result once at process startup (e.g. `api/app.py`). If the backend's SDK isn't installed, `build_tracer` logs a warning and falls back to `NoOpTracer` rather than failing startup
  - `LangfuseTracer` is configured by the SDK's own environment variables: `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` switch it on, `LANGFUSE_BASE_URL` selects the server (Langfuse Cloud by default). With either key unset it logs one INFO line, never builds a `Langfuse` client and behaves like `NoOpTracer`. `LangfuseTracer(client=...)` takes a pre-built client for hosts that manage their own. The SDK v5 needs a Langfuse server of version 4.59 or later.
  - `AGENT_PLATFORM_CAPTURE_CONTENT` (read by `core`) is the only switch for sending prompt and tool payload content; without it only the payload size reaches Langfuse (no digest, no content), and an error span carries the exception type only (the tracer keeps the SDK from recording the exception message and stack trace).
  - To add another backend, implement `BaseTracer`/`Span` under `integrations/tracing/<backend>/` and add it to `PROVIDER_ALIASES`. `Span.trace_id`/`span_id` are exposed for backends whose scoring/feedback APIs attach data to a trace out of band (Langfuse scores)
