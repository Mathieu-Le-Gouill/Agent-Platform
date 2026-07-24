# CLAUDE.md: Agent Platform

Coding-agent guide for this repository. Read this before writing any code. For lookup protocol, layer rules, testing/error conventions, and extension guidance, the primary authority is `AGENTS.md`, this file only adds current-state context and near-term priorities on top of it.

## Quick orientation

- **`AGENTS.md`**, architecture, layer rules, testing/error conventions, extension guidance (authoritative)
- **`src/agent_platform/README.md`**, layer map and known issues
- **`src/agent_platform/<layer>/README.md`**, local patterns for that layer; read before touching it

## Current state (as of 2026-07-24)

The platform is **structurally complete** at the integration and agent layers. The main gaps are:
1. Pipelines are mostly stubs or minimal
2. A few tools are missing
3. `workflows/` is still an empty scaffold; `api/` now has a real FastAPI entrypoint (`api/app.py`) wiring one `ConversationAgent` via `config/container.py::build_agent()`, driven by `config/settings.py::Settings`

### What is solid

- **Core**, interfaces, schemas, error hierarchy, credentials: complete and stable
- **Integrations**, 12 domains with providers: complete (see `integrations/README.md` for the full table). Provider SDKs install via per-provider `pyproject.toml` extras (e.g. `llm-openai`, `vector-store-chroma`), not a monolithic dependency list
- **Components**, `Embedder`, `Chunker`, `Reranker`, `SimilarityScorer`, `EmbeddingClassifier`, `LLMClassifier`: complete
- **Agents**, `Agent`, `AgentExecutor`, `ConversationAgent`, `ToolRegistry`, `TranscribeTool`, `SearchTool`, `OCRTool`: complete
- **Tooling**, mypy and import-linter are wired into CI (`.github/workflows/ci.yml`) alongside ruff and pytest; the layered-architecture and core-isolation rules in AGENTS.md §6 are enforced by `lint-imports`, not just convention
- **Test suite**, `tests/` (1521 tests, structural + coverage-gate pass done): `pyproject.toml [tool.pytest.ini_options] addopts` enforces `--cov-fail-under=90` locally and in CI (config lives in `pyproject.toml [tool.coverage.*]`, no `.coveragerc`); mocking goes through the `pytest-mock` `mocker` fixture, not raw `unittest.mock.patch`; every test is auto-marked `unit` via a `pytest_collection_modifyitems` hook in `tests/conftest.py` unless explicitly marked `integration` (currently 0 tests are, the suite is fully mocked); shared helpers live in `tests/helpers/`. See AGENTS.md §3 for the full convention

### Known issues (prioritized)

| Priority | Issue | File |
|---|---|---|
| P0 | RAG ingest pipeline missing chunk/embed steps (currently just `store.add(chunks)`) | `pipelines/rag/ingest.py` |
| P0 | RAG query pipeline missing embed → rerank → generate steps (currently just `store.search(vector)`) | `pipelines/rag/query.py` |
| P1 | `speech_translation.py` body is pseudocode in a docstring, `run()` raises `NotImplementedError` | `pipelines/speech_translation.py` |
| P1 | `classification` has no integration/provider layer at all (interface only defines response models, no `base.py` ABC, no `integrations/classification/` directory); classification today is implemented entirely in `components/embed_classifier` and `components/llm_classifier` instead | `core/interfaces/classification/`, `integrations/classification/` (missing) |
| P1 | WhisperX diarization (`_diarize`/`_load_diarize_pipeline_sync`) does `from whisperx.diarize import ...` at call time, but that submodule fails to import in this environment (`pyannote.audio` -> `torchaudio.AudioMetaData` AttributeError from a torchaudio/pyannote version mismatch). Plain transcribe/align paths are unaffected; only `WhisperXConfig(diarize=True)` breaks at runtime. Tests cover the diarize logic via a `sys.modules` stand-in, not the real import chain | `integrations/speech_to_text/whisperx/whisperx.py` |
| P2 | `uv run --extra chunking-pdf` (and `--extra all`) fails: uv resolves `unstructured==0.18.32` -> `numba==0.53.1`, incompatible with Python >=3.10; `pip install -e ".[chunking-pdf]"` resolves a newer, working `unstructured` for the same file. Needs an explicit floor pin on `unstructured` in that extra | `pyproject.toml` |
| P3 | `classification-transformers` extra is declared in `pyproject.toml` but nothing imports `transformers`, no consuming code exists yet | `pyproject.toml` |
| P2 | Zero test coverage despite being listed as complete under "What is solid": `components/llm_classifier/*` (component, config, all strategy files), `components/embed_classifier/*`, `components/similarity_scorer.py`, `audio/dsp.py` (0%); `audio/io.py` (56%), `components/similarity.py` (77%), `components/chunker.py` (79%) are partially covered. Surfaced by the honest coverage gate (`AGENTS.md §3`); writing the missing tests was out of scope for the gate-adding pass itself | `components/llm_classifier/`, `components/embed_classifier/`, `components/similarity_scorer.py`, `audio/dsp.py`, `audio/io.py`, `components/similarity.py`, `components/chunker.py` |

## Near-term build priorities

Work in this order unless the user says otherwise:

### 1. Fix RAG pipelines (P0)
`pipelines/rag/ingest.py`, add: loader → chunk → embed → store
`pipelines/rag/query.py`, add: embed query → search → rerank → generate

Both pipelines should accept injected backends (not hardcode providers). Use `Component[InputT, OutputT]` wrappers for each step.

### 2. Wire speech translation pipeline (P1)
`pipelines/speech_translation.py`, implement the `run()` body (currently `NotImplementedError`): audio → STT → translate. Both providers (WhisperX/Deepgram, DeepL/Google) are ready. Support streaming STT → chunked translation.

### 3. Add classification providers (P1)
`integrations/classification/` does not exist yet. Add at minimum `transformers/` (zero-shot with `pipeline("zero-shot-classification")`), consuming the already-declared `classification-transformers` extra. Follow the standard `<domain>/<provider>/config.py + provider.py` pattern and add a `base.py` ABC under `core/interfaces/classification/` first.

### 4. Add missing tools (P2)
`agents/tools/translate.py`, `agents/tools/classify.py`, `agents/tools/generate_image.py`, `agents/tools/summarize.py`. Follow `OCRTool` as a reference, it is the cleanest existing example.

## Keeping this file current

This file is a snapshot, not a spec, so it goes stale unless changes update it. When a change closes a row in "Known issues" or a step in "Near-term build priorities", remove or update that entry in the same change rather than leaving it for later. When a change makes "What is solid" or "Current state" inaccurate (a new layer becomes complete, `workflows/` stops being a stub, etc.), update those sections too and refresh the "as of" date.

## See also

- `AGENTS.md §3`, testing conventions and commands
- `AGENTS.md §4`, error handling conventions
- `AGENTS.md §5`, extension guidance per layer
- `AGENTS.md §6`, architecture/import rules and code conventions
