# CLAUDE.md: Agent Platform

Coding-agent guide for this repository. Read this before writing any code. For lookup protocol, layer rules, testing/error conventions, and extension guidance, the primary authority is `AGENTS.md`, this file only adds current-state context and near-term priorities on top of it.

## Quick orientation

- **`AGENTS.md`**, architecture, layer rules, testing/error conventions, extension guidance (authoritative)
- **`src/agent_platform/README.md`**, layer map and known issues
- **`src/agent_platform/<layer>/README.md`**, local patterns for that layer; read before touching it

## Current state (as of 2026-07-25)

The platform is **structurally complete** at the integration, component, pipeline, and agent layers. The remaining gap is `workflows/`, still an empty scaffold; `api/` has a real FastAPI entrypoint (`api/app.py`) wiring one `ConversationAgent` via `config/container.py::build_agent()`, driven by `config/settings.py::Settings`.

### What is solid

- **Core**, interfaces, schemas, error hierarchy, credentials: complete and stable, including `core/interfaces/classification/base.py` (`BaseClassificationProvider` ABC, added alongside the other domain ABCs)
- **Integrations**, 13 domains with providers: complete (see `integrations/README.md` for the full table), including `classification/transformers` (zero-shot via `pipeline("zero-shot-classification")`, consuming the `classification-transformers` extra). Provider SDKs install via per-provider `pyproject.toml` extras (e.g. `llm-openai`, `vector-store-chroma`), not a monolithic dependency list
- **Components**, `Embedder`, `Chunker`, `Loader`, `Reranker`, `Generator`, `VectorSearch`, `SimilarityScorer`, `EmbeddingClassifier`, `LLMClassifier`: complete
- **Pipelines**, `pipelines/rag/ingest.py` (loader → chunk → store, embedding happens inside the vector store's own injected backend, see `AGENTS.md §5`), `pipelines/rag/query.py` (embed query → search → rerank → generate), `pipelines/speech_translation.py` (`SpeechTranslationPipeline.run()`/`.stream()`, audio → STT → translate with streaming support): complete
- **Agents**, `Agent`, `AgentExecutor`, `ConversationAgent`, `ToolRegistry`, `TranscribeTool`, `SearchTool`, `OCRTool`, `TranslateTool`, `ClassifyTool`, `GenerateImageTool`, `SummarizeTool`: complete
- **Tooling**, mypy and import-linter are wired into CI (`.github/workflows/ci.yml`) alongside ruff and pytest; the layered-architecture and core-isolation rules in AGENTS.md §6 are enforced by `lint-imports`, not just convention
- **Test suite**, `tests/` (1673 tests, 1670 passed + 3 skipped, structural + coverage-gate pass done): `pyproject.toml [tool.pytest.ini_options] addopts` enforces `--cov-fail-under=90` locally and in CI (config lives in `pyproject.toml [tool.coverage.*]`, no `.coveragerc`); actual coverage is 97.67%. Mocking goes through the `pytest-mock` `mocker` fixture, not raw `unittest.mock.patch`; every test is auto-marked `unit` via a `pytest_collection_modifyitems` hook in `tests/conftest.py` unless explicitly marked `integration` (currently 0 tests are, the suite is fully mocked); shared helpers live in `tests/helpers/`. See AGENTS.md §3 for the full convention

### Known issues

None currently tracked. Git history carries the record of what was fixed and when, this section only tracks what's currently open.

## Near-term build priorities

Nothing urgent is queued. The next largest gap is `workflows/` (still an empty scaffold) if that becomes a priority; otherwise defer to the user.

## Keeping this file current

This file is a snapshot, not a spec, so it goes stale unless changes update it. When a change closes a row in "Known issues" or a step in "Near-term build priorities", remove or update that entry in the same change rather than leaving it for later. When a change makes "What is solid" or "Current state" inaccurate (a new layer becomes complete, `workflows/` stops being a stub, etc.), update those sections too and refresh the "as of" date.

## See also

- `AGENTS.md §3`, testing conventions and commands
- `AGENTS.md §4`, error handling conventions
- `AGENTS.md §5`, extension guidance per layer
- `AGENTS.md §6`, architecture/import rules and code conventions
