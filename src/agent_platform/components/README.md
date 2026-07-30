# Components: Reusable Processing Units

## Design

`components/` is the **composition layer** between raw Integrations and full Pipelines. A component wraps one or more Integration providers into a higher-level, reusable processing unit with a standard interface.

Unlike an Integration (which implements a `core/interfaces` ABC for a specific library), a Component:
- Is framework-independent, it follows the `Component[InputT, OutputT]` ABC
- May compose multiple backends (e.g., embed + classify)
- May add logic not present in any single provider (e.g., similarity scoring, thresholding)
- Has no external library dependencies of its own, it delegates to Integrations

## Component ABC

`components/base.py` defines:

```python
class Component(ABC, Generic[Out]):
    @abstractmethod
    async def arun(self, input: In) -> Out: ...

    def run(self, input: In) -> Out:
        return asyncio.run(self.arun(input))
```

Every component receives typed input and returns typed output. The sync `run()` is a convenience wrapper around `arun()`.

## Directory Layout

Every component is a directory, one per component, matching the convention `integrations/<domain>/<provider>/` already uses: `component.py` always, `config.py` only if the component owns a config the domain-level `core/interfaces` config doesn't already cover, `strategies/` or other sub-modules only if the component actually needs them. No per-component `__init__.py`, callers import the submodule directly (`from agent_platform.components.embedder.component import Embedder`), same as integration providers.

```
components/
├── __init__.py
├── base.py                       # Component[InputT, OutputT] ABC
├── chunker/
│   └── component.py               # TextDocument → TextChunk (wraps BaseChunker)
├── embedder/
│   └── component.py               # TextChunk → EmbeddingResponse (wraps BaseEmbeddingProvider)
├── reranker/
│   └── component.py               # list[T] → list[T] (wraps BaseReranker)
├── ocr/
│   └── component.py               # (source, config) → list[TextChunk] (wraps BaseOCR)
├── speech_to_text/
│   └── component.py               # (AudioChunk, config) → Transcript, plus astream() (wraps BaseSpeechToText)
├── vector_search/
│   └── component.py               # (vector, k, filter, config) → list[(TextChunk, Score)] (wraps VectorStore.search_with_scores)
├── dataset/
│   └── component.py               # (path, config) → dict[DatasetSplit, BaseDatasetSplit[T]] (wraps BaseDatasetProvider, same per-call (data, config) input tuple as OCR/SpeechToText)
├── similarity_scorer/
│   └── component.py               # Chunks × label vectors → ClassificationResult (pure logic)
├── semantic_chunker/              # TextDocument → TextChunk, splits on embedding-similarity breakpoints
│   ├── component.py
│   └── config.py
├── contextual_chunker/            # TextDocument → TextChunk, prepends an LLM-generated context blurb
│   ├── component.py
│   └── config.py
├── embed_classifier/              # Embedding-based classifier
│   ├── component.py
│   └── config.py
└── llm_classifier/                # LLM-based classifier with pluggable strategies
    ├── component.py
    ├── config.py
    └── strategies/
        ├── zero_shot.py
        ├── few_shot.py
        ├── sentiment.py
        └── text_classification.py
```

## Component Patterns

### Wrapper component (single backend)

Wraps one Integration provider. The backend is injected at init, the component exposes a clean input/output contract.

```python
class Embedder(Component[EmbedderInput[EmbedConfigT], EmbeddingResponse]):
    def __init__(self, backend: BaseEmbeddingProvider) -> None: ...
    async def arun(self, input: EmbedderInput[EmbedConfigT]) -> EmbeddingResponse: ...
```

### Where config lives: constructor, or `arun()` input?

Components follow the same split `integrations/` uses between what's fixed at construction and what varies per call, just with different names for the fixed part. An Integration provider takes `credentials`/`client_options` (secrets and transport settings) in `__init__` and `config` (behavior for one call) as a per-call method argument, never stored on `self`. A Component takes its backend(s) (constructor injection, the component's equivalent of a fixed client) in `__init__`, and **always** bundles `config` into the `arun()` input alongside the data, as a `NamedTuple`: `EmbedderInput(chunks, config)`. `NamedTuple` is used instead of a plain `tuple[...]` alias so multi-field inputs (`VectorSearchInput`'s `vector`/`k`/`filter`/`config`, say) get named fields and positional unpacking stays available (`chunks, config = input`), rather than a position-only tuple where swapping two same-typed fields fails silently. Never store `config` on `self` in a component's `__init__`, even if every call site happens to pass the same value, that's exactly what the corresponding integration provider itself doesn't do either.

```python
class Embedder(Component[EmbedderInput[EmbedConfigT], EmbeddingResponse]):
    def __init__(self, backend: BaseEmbeddingProvider) -> None:
        self._backend = backend

    async def arun(self, input: EmbedderInput[EmbedConfigT]) -> EmbeddingResponse:
        chunks, config = input
        return await self._backend.aembed_document(chunks, config)
```

Every component in this directory follows this shape: `Chunker`, `Embedder`, `Generator`, `Loader`, `Reranker`, `VectorSearch`, `SemanticChunker`, `ContextualChunker`, `LLMClassifier`, `OCR`, `SpeechToText`, `Dataset`. `EmbeddingClassifier` and `SimilarityScorer` already took config per call (they compose other components rather than wrap a single backend) and needed no change.

Locking config into `__init__` ties one component instance to one config, when the whole point of constructor injection is to reuse the same instance with a different config each call, the same reason integration providers never store config on `self` either.

### Composite component (multiple backends)

Combines multiple Integration providers to produce a result that requires orchestration.

```python
class EmbeddingClassifier(Component[...]):
    def __init__(self, embedder: Embedder, scorer: SimilarityScorer): ...
    async def arun(self, input) -> ClassificationResult: ...
```

### Pure-logic component

No backend, all logic is self-contained.

```python
class SimilarityScorer(Component[SimilarityInput, ClassificationResult]):
    async def arun(self, input) -> ClassificationResult: ...
```

## How to Extend

### Add a component

```
components/<name>/
└── component.py      # The component class
```

1. Create `components/<name>/component.py`
2. Subclass `Component[InputT, OutputT]`
3. Accept the backend Integration(s) in `__init__`
4. Implement `arun()`, one async method that orchestrates the backend(s)
5. Add `config.py` alongside it only if the component needs its own config beyond the domain-level `core/interfaces` config; add `strategies/` or other sub-modules only if the component actually needs them

No `__init__.py`, callers import `components.<name>.component` directly.

### Component conventions

- Backends are injected (constructor injection), never instantiated inside the component
- Configs are typed Pydantic models
- Config always goes into the `arun()` input alongside the data as a `(data, config)` tuple, never into `__init__`, see "Where config lives" above
- Errors are translated to `core/errors.py` types
- Components may import from `core/`, `integrations/`, and `components/`, never from `pipelines/` or `agents/`
- Every component is a directory (`components/<name>/component.py`), no flat-file components, matching `integrations/<domain>/<provider>/`'s layout
