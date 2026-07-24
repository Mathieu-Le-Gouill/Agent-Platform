# Pipelines — Status & Roadmap

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
**Status: Commented-out pseudocode**

```
Audio → [STT] → Text → [Translator] → Translated Text
```

`SpeechTranslationPipeline` class exists with constructor and method signatures, but `run()` body is in a docstring comment. STT and Translation providers are now implemented (WhisperX, Deepgram, DeepL, Google Translate), so this pipeline needs actual wiring + streaming support.

### RAG — Ingest (`rag/ingest.py`)
**Status: Minimal**

```python
async def ingest(chunks: list[TextChunk], store: VectorStore) -> None:
    await store.add(chunks)
```

**Missing:** Embedding step, chunking, document preprocessing, metadata handling, batching.

### RAG — Query (`rag/query.py`)
**Status: Minimal**

```python
async def query(vector: list[float], store: VectorStore, k: int = 5) -> list[TextChunk]:
    return await store.search(vector, k=k)
```

**Missing:** Full pipeline (embed → search → rerank → generate).

## Planned Pipelines

| Pipeline | Dependencies | Priority |
|---|---|---|
| **Transcription** | Speech provider (ready) | P0 |
| **Translation** | Translation provider (ready) | P0 |
| **Audio Translation** | Speech + Translation providers (ready) | P0 |
| **TTS / Synthesis** | Speech synthesis provider | P1 |
| **Conversation** | All of the above + LLM | P1 |
| **Summarization** | LLM provider | P1 |
| **Document QA** | RAG pipeline + LLM | P1 |
| **Multi-modal QA** | OCR + RAG + LLM | P2 |
| **Clustering Pipeline** | Clustering providers | P2 |
| **Classification Pipeline** | Classification providers | P2 |

## Implementation Order

1. Add embedding step to `rag/ingest.py`
2. Implement full RAG query pipeline (embed → search → rerank → generate)
3. Wire `speech_translation.py` (STT and Translation providers already exist)
4. Add ingestion chain (loader → chunk → embed → store)
