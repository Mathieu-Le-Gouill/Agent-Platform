# CLAUDE.md: Agent Platform

Coding-agent guide for this repository. Read this before writing any code. For lookup protocol, layer rules, testing/error conventions, and extension guidance, the primary authority is `AGENTS.md`, this file only adds current-state context and near-term priorities on top of it.

## Quick orientation

- **`AGENTS.md`**, architecture, layer rules, testing/error conventions, extension guidance (authoritative)
- **`src/agent_platform/README.md`**, layer map and known issues
- **`src/agent_platform/<layer>/README.md`**, local patterns for that layer; read before touching it

## Current state (as of 2026-07-26, llm/embeddings/vector_store migrated off LangChain to native vendor SDKs; llm/embeddings gained a native shared `_base.py` per domain to remove cross-provider boilerplate; speech_to_text gained a narrower shared `_base.py::buffered_stream()` helper for its client-side streaming providers)

The platform is **structurally complete** at the integration, component, pipeline, agent, and evals layers. The remaining gap is `workflows/`, still an empty scaffold; `api/` has a real FastAPI entrypoint (`api/app.py`) wiring one `ConversationAgent` via `config/container.py::build_agent()`, driven by `config/settings.py::Settings`. That agent already carries `GenerateImageTool`/`TranscribeTool`, not chat only: `Settings.default_llm_model`/`default_image_model`/`default_audio_model` are `"<provider>:<model>"` strings (`core/config.py::parse_model_string`) resolved against each domain's `PROVIDER_ALIASES` map (`integrations/<domain>/__init__.py`) by `build_provider`/`build_provider_from_model_string`.

### What is solid

- **Core**, interfaces, schemas, error hierarchy, credentials: complete and stable, including `core/interfaces/classification/base.py` (`BaseClassificationProvider` ABC, added alongside the other domain ABCs)
- **Integrations**, 13 domains with providers: complete (see `integrations/README.md` for the full table), including `classification/transformers` (zero-shot via `pipeline("zero-shot-classification")`, consuming the `classification-transformers` extra). Provider SDKs install via per-provider `pyproject.toml` extras (e.g. `llm-openai`, `vector-store-chroma`), not a monolithic dependency list. The `llm`, `embeddings`, and `vector_store` domains (14 providers total) each call their vendor's native SDK directly now, no LangChain intermediate; each domain's shared `langchain_base.py` has been deleted along with the `langchain-*` deps it pulled in. `reranking` is the only domain that still has an (optional) shared LangChain wrapper. `llm` and `embeddings` each have a native, non-LangChain replacement instead: `integrations/<domain>/_base.py` (`NativeLLMProvider`/`NativeEmbeddingProvider`) factors the tracing/retry/token-usage or request/response plumbing that's identical across that domain's providers, so each provider file holds only its vendor-specific client construction and mapping logic; this never crosses domain boundaries (e.g. `llm/openai` and `embeddings/openai` still duplicate their small client-kwargs snippet on purpose, since integrations must stay self-contained per `integrations/README.md`). `speech_to_text` has a narrower shared `_base.py::buffered_stream()` (a free async generator, not a provider base class) factoring the client-side windowing loop shared by `whisperx`/`faster_whisper`/`openai`'s `stream()`; `deepgram` is exempt since its streaming is a native socket protocol, not client-side batching. `ocr`, `translation`, and `image_generation` were evaluated for the same pattern and deliberately left without a shared base: each has a single method per provider (no sync/async or batch/stream duplication to template) and vendor-specific response shapes, so a template would add abstraction without removing real duplication
- **Components**, `Embedder`, `Chunker`, `Loader`, `Reranker`, `Generator`, `VectorSearch`, `SimilarityScorer`, `EmbeddingClassifier`, `LLMClassifier`: complete
- **Pipelines**, `pipelines/rag/ingest.py` (loader → chunk → embed via `Embedder` → store, `BaseVectorStore.add()` takes precomputed vectors directly), `pipelines/rag/query.py` (embed query → search → rerank → generate), `pipelines/speech_translation.py` (`SpeechTranslationPipeline.run()`/`.stream()`, audio → STT → translate with streaming support): complete
- **Agents**, `Agent`, `AgentExecutor`, `ConversationAgent`, `ToolRegistry`, `TranscribeTool`, `SearchTool`, `OCRTool`, `TranslateTool`, `ClassifyTool`, `GenerateImageTool`, `SummarizeTool`: complete
- **Evals**, `evals/` package (`EvalCase`/`EvalDataset`/`EvalRunner`/`Scorer`/`EvalReport`, `agent-platform-eval` CLI, golden `ConversationAgent`/`AgentExecutor` tool-selection dataset), Phase 1 of the agents-layer roadmap: complete, see `evals/README.md`
- **Tooling**, mypy and import-linter are wired into CI (`.github/workflows/ci.yml`) alongside ruff and pytest; the layered-architecture and core-isolation rules in AGENTS.md §6 are enforced by `lint-imports`, not just convention; an optional `evals` CI job runs the golden dataset against a real provider when `OPENAI_API_KEY` is set (`continue-on-error: true` until a real-run baseline validates the threshold)
- **Test suite**, `tests/` (structural + coverage-gate pass done): `pyproject.toml [tool.pytest.ini_options] addopts` enforces `--cov-fail-under=90` locally and in CI (config lives in `pyproject.toml [tool.coverage.*]`, no `.coveragerc`). Mocking goes through the `pytest-mock` `mocker` fixture, not raw `unittest.mock.patch`; every test is auto-marked `unit` via a `pytest_collection_modifyitems` hook in `tests/conftest.py` unless explicitly marked `integration` (currently 0 tests are, the suite is fully mocked); shared helpers live in `tests/helpers/`. See AGENTS.md §3 for the full convention

### Known issues

None currently tracked. Git history carries the record of what was fixed and when, this section only tracks what's currently open.

## Near-term build priorities

A phased roadmap for closing the gap with modern agent platforms (evals,
orchestration, harness hardening, observability polish) lives in
`src/agent_platform/agents/README.md` ("Roadmap: closing the gap with modern
agent platforms"). Phase 1 (evals) is done, see `evals/README.md`. Phase 2
(`workflows/` orchestration, still an empty scaffold) is next: it needs a
user decision on the orchestration engine (LangGraph vs. an in-house DAG
executor) before implementation starts, see that README's Phase 2 section.
Keep that README's phase list in sync as items land, and update this
section only if the overall priority order changes.

## Keeping this file current

This file is a snapshot, not a spec, so it goes stale unless changes update it. When a change closes a row in "Known issues" or a step in "Near-term build priorities", remove or update that entry in the same change rather than leaving it for later. When a change makes "What is solid" or "Current state" inaccurate (a new layer becomes complete, `workflows/` stops being a stub, etc.), update those sections too and refresh the "as of" date.

## See also

- `AGENTS.md §3`, testing conventions and commands
- `AGENTS.md §4`, error handling conventions
- `AGENTS.md §5`, extension guidance per layer
- `AGENTS.md §6`, architecture/import rules and code conventions
