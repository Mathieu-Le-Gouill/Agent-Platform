# CLAUDE.md — Agent Platform

Coding-agent guide for this repository. Read this before writing any code. For lookup protocol, layer rules, testing/error conventions, and extension guidance, the primary authority is `AGENTS.md` — this file only adds current-state context and near-term priorities on top of it.

## Quick orientation

- **`AGENTS.md`** — architecture, layer rules, testing/error conventions, extension guidance (authoritative)
- **`src/agent_platform/README.md`** — layer map and known issues
- **`src/agent_platform/<layer>/README.md`** — local patterns for that layer; read before touching it

## Current state (as of 2026-07)

The platform is **structurally complete** at the integration and agent layers. The main gaps are:
1. Pipelines are mostly stubs or minimal
2. A few tools are missing
3. Scaffold directories (`api/`, `workflows/`) are empty

### What is solid

- **Core** — interfaces, schemas, error hierarchy, credentials: complete and stable
- **Integrations** — 13 domains with providers: complete (see `integrations/README.md` for the full table)
- **Components** — `Embedder`, `Chunker`, `Reranker`, `SimilarityScorer`, `EmbeddingClassifier`, `LLMClassifier`: complete
- **Agents** — `Agent`, `AgentExecutor`, `ConversationAgent`, `ToolRegistry`, `TranscribeTool`, `SearchTool`, `OCRTool`: complete

### Known issues (prioritized)

| Priority | Issue | File |
|---|---|---|
| P0 | `main.py` missing but declared as entrypoint in `pyproject.toml` | `pyproject.toml` |
| P0 | RAG ingest pipeline missing embedding step | `pipelines/rag/ingest.py` |
| P0 | RAG query pipeline missing embed → rerank → generate steps | `pipelines/rag/query.py` |
| P1 | `speech_translation.py` body is pseudocode in a docstring | `pipelines/speech_translation.py` |
| P1 | `classification` integration has no providers, only `utils.py` | `integrations/classification/` |
| P2 | DI container commented out | `config/container.py` |

## Near-term build priorities

Work in this order unless the user says otherwise:

### 1. Fix RAG pipelines (P0)
`pipelines/rag/ingest.py` — add: loader → chunk → embed → store
`pipelines/rag/query.py` — add: embed query → search → rerank → generate

Both pipelines should accept injected backends (not hardcode providers). Use `Component[InputT, OutputT]` wrappers for each step.

### 2. Wire speech translation pipeline (P1)
`pipelines/speech_translation.py` — implement: audio → STT → translate. Both providers (WhisperX/Deepgram, DeepL/Google) are ready. Support streaming STT → chunked translation.

### 3. Add classification providers (P1)
`integrations/classification/` has no provider subdirectories yet. Add at minimum `transformers/` (zero-shot with `pipeline("zero-shot-classification")`). Follow the standard `<domain>/<provider>/config.py + provider.py` pattern.

### 4. Add missing tools (P2)
`agents/tools/translate.py`, `agents/tools/classify.py`, `agents/tools/generate_image.py`, `agents/tools/summarize.py`. Follow `OCRTool` as a reference — it is the cleanest existing example.

## See also

- `AGENTS.md §3` — testing conventions and commands
- `AGENTS.md §4` — error handling conventions
- `AGENTS.md §5` — extension guidance per layer
- `AGENTS.md §6` — architecture/import rules and code conventions
