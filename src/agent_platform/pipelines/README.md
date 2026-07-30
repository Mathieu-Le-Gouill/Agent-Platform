# Pipelines: Status & Roadmap

Pipelines compose multiple integrations into multi-step, asynchronous flows.

## Current Pipelines

### Ingestion (`ingestion.py`)
**Status: Basic implementation**

Splits a `TextDocument` into `TextChunk`s using `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)`.

```
TextDocument → [RecursiveCharacterTextSplitter] → list[TextChunk]
```

**Missing:** Multi-format support (PDF via loader first), semantic chunking, metadata propagation, deduplication.

### Speech Translation (`speech_translation.py`)
**Status: Complete**

```
Audio → [STT] → Transcript → [Translator] → Translated Transcript
```

`SpeechTranslationPipeline.run()` transcribes an `AudioChunk` and translates each utterance (skipped when the detected language already matches the target). `.stream()` does the same over an `AsyncIterator[AudioChunk]`, translating each `Transcript` as the STT backend yields it.

### RAG - Ingest (`rag/ingest.py`)
**Status: Complete**

```python
async def ingest(sources, loader: Loader, chunker: Chunker, embedder: Embedder, store: VectorStore, loader_config=None, chunker_config=None, embedding_config=None, store_config=None) -> None:
    documents = await loader.arun((sources, loader_config))
    chunks = await chunker.arun(ChunkerInput(documents, chunker_config))
    embedding_response = await embedder.arun(EmbedderInput(chunks, embedding_config))
    vectors = [embedding.to_list() for embedding in embedding_response.embeddings]
    await store.add(chunks, vectors, config=store_config)
```

Chunks are embedded via `Embedder` before reaching the store; `VectorStore.add()` takes the precomputed `vectors` directly rather than relying on a vector store's own injected embedding backend.

### RAG - Query (`rag/query.py`)
**Status: Complete**

```python
async def query(query_text, embedder: Embedder, vector_search: VectorSearch, reranker: Reranker, generator: Generator, k=5, filter=None) -> LLMResponse:
    ...  # embed query → search → rerank → generate
```

## Planned Pipelines

| Pipeline | Dependencies | Priority |
|---|---|---|
| **TTS / Synthesis** | Speech synthesis provider | P1 |
| **Conversation** | RAG + speech translation + LLM | P1 |
| **Document QA** | RAG pipeline + LLM | P1 |
| **Multi-modal QA** | OCR + RAG + LLM | P2 |
| **Clustering Pipeline** | Clustering providers | P2 |
| **Classification Pipeline** | Classification providers (ready) | P2 |
