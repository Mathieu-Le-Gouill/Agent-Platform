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
│   └── component.py               # (vector, k, filter) → list[(TextChunk, Score)] (wraps VectorStore.search_with_scores)
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
class Embedder(Component[list[TextChunk], EmbeddingResponse]):
    def __init__(self, backend: BaseEmbeddingProvider, config=None): ...
    async def arun(self, input: list[TextChunk]) -> EmbeddingResponse: ...
```

### Where config lives: constructor, or `arun()` input?

One question decides it: **will every call to this component use the same config?**

- **Yes, same config every call** → put it in `__init__` as `self._config`, `arun()`'s input is just the data. This is the common case: `Chunker`, `Embedder`, `Loader` all work this way.
  ```python
  class Embedder(Component[list[TextChunk], EmbeddingResponse]):
      def __init__(self, backend: BaseEmbeddingProvider, config=None) -> None:
          self._config = config

      async def arun(self, input: list[TextChunk]) -> EmbeddingResponse:
          return await self._backend.aembed_document(input, self._config)
  ```

- **No, config can differ per call** → don't put it in `__init__` at all. Bundle it into the `arun()` input alongside the data, as a tuple: `(data, config)`.
  ```python
  OCRInput = tuple[str, OCRConfigT | None]

  class OCR(Component[OCRInput[OCRConfigT], list[TextChunk]]):
      def __init__(self, backend: BaseOCR[OCRConfigT]) -> None:
          self._backend = backend

      async def arun(self, input: OCRInput[OCRConfigT]) -> list[TextChunk]:
          source, config = input
          return await self._backend.extract(source=source, config=config)
  ```
  `components/ocr`, `components/speech_to_text`, and `components/dataset` all use this shape, because their backend methods (`BaseOCR.extract(source, config, ...)`, `BaseSpeechToText.transcribe(audio, config)`, `BaseDatasetProvider.load(record_type, path, config)`) already take config as one argument among several per call, not a fixed setting.

Getting this backwards locks one component instance to one config, when the whole point was to reuse it with a different config each call.

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
- Config goes in `__init__` if every call uses the same one, otherwise into the `arun()` input alongside the data, see "Where config lives" above
- Errors are translated to `core/errors.py` types
- Components may import from `core/`, `integrations/`, and `components/`, never from `pipelines/` or `agents/`
- Every component is a directory (`components/<name>/component.py`), no flat-file components, matching `integrations/<domain>/<provider>/`'s layout
