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
│   └── <provider>/
│       ├── __init__.py
│       ├── config.py      # Provider-specific Pydantic config
│       └── <provider>.py  # Concrete class implementing core/interfaces ABC
└── …
```

**Exception:** `loader/` uses `strategies/` instead of `<provider>/` because each strategy handles a different media type (text, image, audio, video) rather than a different vendor. The top-level `loader/text/`, `loader/image/`, `loader/audio/`, `loader/video/`, `loader/composite/` directories are empty placeholders left over from an earlier layout, all current implementation lives under `loader/strategies/`.

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
| `llm/` | openai, anthropic, mistral, ollama, huggingface | `BaseLLMProvider` |
| `embeddings/` | openai, mistral, ollama, huggingface | `BaseEmbeddingProvider` |
| `vad/` | silero, webrtc, pvcobra, ten | `BaseVAD` |
| `ocr/` | tesseract, google_vision, aws_textract, mistral | `BaseOCR` |
| `vector_store/` | chroma, qdrant, pinecone, weaviate, faiss | `BaseVectorStore` |
| `reranking/` | cohere, jina, huggingface, flashrank, voyage | `BaseReranker` |
| `chunking/` | recursive, markdown, html, latex, pdf | `BaseChunker` |
| `speech_to_text/` | whisperx, faster_whisper, deepgram, openai | `BaseSpeechToText` |
| `translation/` | deepl, google_translate, azure | `BaseTranslator` |
| `clustering/` | hdbscan, kmeans, gmm | `ClusteringAlgorithm` |
| `loader/` | strategies: pil (image), pyav (video/audio), soundfile (audio), unstructured (text/composite documents) | `BaseMediaLoader` |
| `image_generation/` | dalle, midjourney, stable_diffusion | `BaseImageGenerator` |

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
- `langchain_base.py` is only used when multiple providers share a common LangChain wrapper pattern, it is optional
- Each domain's `__init__.py` lazily re-exports its provider classes via a module-level `__getattr__` (PEP 562) over a `_PROVIDERS: dict[class_name, module_path]` map, so callers import `from agent_platform.integrations.<domain> import <ProviderClass>` instead of the nested `<domain>.<provider>.<provider>` path, without eagerly importing every provider's SDK. Add a provider by adding one entry to that map, no separate factory/registry layer
- Provider SDKs are not in the base install, each provider's package(s) are declared as a `<domain>-<provider>` extra in `pyproject.toml`. When adding a new provider, add its package(s) as a new extra and append it to that domain's bundle extra and to `all`
- Providers/shared bases that make outbound network calls (OCR, translation, speech-to-text, DALL-E/Midjourney image generation, and the shared `langchain_base.py` for `llm`, `embeddings`, `vector_store`, `reranking`) wrap their call in `@error_logged(re_raise=ProviderError, message=...)` outer + `@with_retry()` inner from `core/errors.py`, retry happens first, and if every attempt fails the final exception is logged and translated to `ProviderError`. The `reranking` shared base applies this uniformly to all providers built on it (including the locally-running `huggingface` cross-encoder), since one shared `rerank()` method serves them all. Purely local/offline compute with its own bespoke method (`tesseract`, `whisperx`, `vad`, `clustering`, `chunking`, `flashrank`, `stable_diffusion`) is exempt from both, retrying a deterministic local failure wastes CPU/GPU time instead of recovering from it
- LLM tracing is opt-in via `AGENT_PLATFORM_TRACING` (`none` | `langsmith` | `langfuse`), read by `core/tracing.TracingConfig` and consumed in `integrations/llm/langchain_base.py`. LangSmith needs `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`; Langfuse needs `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (+ optional `LANGFUSE_HOST`)
