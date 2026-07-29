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
- Errors are translated to `core/errors.py` types
- Components may import from `core/`, `integrations/`, and `components/`, never from `pipelines/` or `agents/`
- Every component is a directory (`components/<name>/component.py`), no flat-file components, matching `integrations/<domain>/<provider>/`'s layout
