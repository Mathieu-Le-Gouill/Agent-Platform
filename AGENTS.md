# AGENTS.md

This file is the entry point for any coding agent working in this repository.
Read it fully before making changes. It is intentionally short, it points to
where the real detail lives rather than duplicating it.

## 0. Lookup protocol (read this first, every time)

Before editing anything, follow this order:

1. Read this root `AGENTS.md`.
2. Identify the directory (or directories) you're about to touch.
3. Read `README.md` inside that directory, if one exists. It describes local
   conventions, extension points, and gotchas that override generic defaults.
4. Skim 1-2 sibling files in that directory to confirm the README matches
   current practice (READMEs can drift; code is ground truth if they conflict,
   but flag the mismatch instead of silently picking one).
5. Only then write or edit code.

If a directory has no README, treat the parent directory's README as the
active context and follow its conventions unless told otherwise.

## 1. Project overview

`agent_platform` is an async-first Python library for building LLM agents and
pipelines (RAG, speech translation, ingestion) on top of pluggable provider
integrations, LLMs (OpenAI, Anthropic, Mistral, Ollama, HuggingFace), vector
stores (Chroma, Qdrant, Pinecone, Weaviate, FAISS), embeddings, reranking,
OCR, speech-to-text, translation, VAD, clustering, classification, and image
generation, mostly via LangChain adapters. Everything is `pydantic` v2 typed
and layered strictly bottom-up: `core` → `integrations` → `components` →
`pipelines` → `agents`.

## 2. Directory map

Purpose only, enough to know which README to open next.

| Path | Purpose | Has local README? |
|---|---|---|
| `src/agent_platform/core/` | Near-zero-dependency foundation: ABCs, Pydantic schemas, error hierarchy, credentials, OTel-based tracing | yes |
| `src/agent_platform/integrations/` | Provider adapters per domain (llm, embeddings, vector_store, reranking, ocr, speech_to_text, translation, vad, chunking, clustering, loader, image_generation), 12 domains; `classification` has no integration/provider layer, see `src/agent_platform/README.md` known issues | yes |
| `src/agent_platform/components/` | Reusable processing units wrapping one or more integrations (`Chunker`, `Embedder`, `Reranker`, classifiers) | yes |
| `src/agent_platform/pipelines/` | Multi-step orchestration flows composing components (ingestion, RAG, speech translation) | yes |
| `src/agent_platform/agents/` | Top-level composition: `Agent`, `AgentExecutor`, `ConversationAgent`, `ToolRegistry`, tools | yes |
| `src/agent_platform/audio/` | DSP utilities, resampling, waveform chunking, tensor/numpy/base64 conversion | no |
| `src/agent_platform/utils/` | Shared helpers: async batching, env var parsing, score utilities | no |
| `src/agent_platform/config/` | Logging setup, `Settings` (pydantic-settings), `build_agent()` DI factory, generic `build_provider(domain_module, provider_name)` factory reused by any future domain wiring | no |
| `src/agent_platform/api/` | FastAPI app wiring `build_agent()` behind `/chat` and `/health` | no |
| `src/agent_platform/workflows/` | Scaffold stub (LangGraph state machine), not yet implemented | no |
| `tests/` | Test suite, mirrors `src/agent_platform` structure | no |

Read `src/agent_platform/README.md` first for the full layer diagram before
diving into a specific directory's README.

## 3. Testing conventions

Never hardcode or guess a venv path (`.venv`, `.test_venv`, an absolute path,
etc.). Run every command through `uv run`, it resolves the project's `.venv`
by walking up from the current directory to `pyproject.toml`, creating it on
first use if it doesn't exist yet. This works the same on any machine and in
any subdirectory, so there is nothing venv-related to write down beyond this
paragraph.

```bash
uv run --extra dev pytest -v                          # full suite (creates/updates .venv from pyproject on first run)
uv run pytest -v -m unit                               # unit tests only
uv run pytest -v tests/agents/test_executor.py          # single file
uv run mypy src/                                       # type-check
uv run lint-imports                                     # enforce the layered-architecture rules in §6
uv run ruff check .                                     # lint
uv run ruff format --check .                            # formatting
```

`uv` only manages the venv, not the dependency source of truth: provider
SDKs are still declared as `pyproject.toml` extras (see `README.md` Setup)
and CI still installs via `pip install -e ".[dev]"`, so `uv run --extra dev
--extra <providers-you-need>` is the local equivalent, not a separate
dependency system. No `uv.lock` is committed.

- Tests live in `tests/<mirror-of-src-path>`, one test file per source file
  (e.g. `agents/executor.py` → `tests/agents/test_executor.py`).
- Async tests run automatically (`asyncio_mode = "auto"` in `pyproject.toml`),
  no `@pytest.mark.asyncio` needed.
- **Rule:** any new function/branch/edge case gets a corresponding test in
  the same change. No exceptions for "small" changes, small changes are
  where regressions hide.
- Per-layer expectations:
  - **core**, schema/validation behavior only, no real providers
  - **integrations**, mock/fake the external SDK; no live network calls in
    the default run; mark `@pytest.mark.integration` if a live call is needed
  - **components**, test `arun()` including error propagation
  - **pipelines**, failure-path and retry logic with fake components
  - **agents**, think/act loop termination with mocked tools and LLM
- Prefer extending an existing test file's patterns (fixtures, mocks) over
  introducing a new testing style in the same module.
- CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`,
  `mypy src/`, `lint-imports`, and the pytest suite (with coverage) on every
  push/PR to `main`, on Python 3.11 and 3.12. A change that fails any of
  these locally will fail CI the same way.
- **Run the checks locally before committing/pushing, don't rely on CI to
  catch it first.** `pre-commit install` (see `README.md` Git hooks) wires
  ruff, mypy, and lint-imports into `git commit`, and the full `pytest`
  suite into `git push`; if hooks aren't installed, run the equivalent
  commands from this section manually.
- If a change touches shared/public behavior (function signatures, error
  types, schemas, provider interfaces), run the **full** `pytest` suite
  before pushing, not just the tests for the file you touched, breaking
  changes usually surface in a caller's test, not the callee's.

## 4. Error handling conventions

- All platform errors extend `PlatformError` (`core/errors.py`):
  `ProviderError`, `ConfigError`, `NotFoundError`, `ValidationError`,
  `MissingCredentialError`, plus layer-specific subtrees (`LLMError` in
  `integrations/llm/`, `AgentError`/`ToolError` in `agents/`).
- **Errors must be translated at layer boundaries, never leaked as internal
  types.** Concretely: an integration raises `ProviderError` (or a domain
  subclass); a `Tool` wrapping it must catch and re-raise as `ToolError`
  (use `agents/tools/_utils.py::safe_call()`); the agent loop must surface
  failures as `AgentError`.
- New error cases extend the existing local hierarchy for that layer
  (`core/errors.py`, `agents/errors.py`, `agents/tools/errors.py`) rather
  than introducing a new exception type or raising bare/generic exceptions.
- Never swallow errors silently.

## 5. Extension guidance

| Want to add | Start here |
|---|---|
| New LLM / embedding / OCR / vector store / … provider | `integrations/<domain>/<new_provider>/` implementing `core/interfaces/<domain>/base.py`; register in that domain's `_PROVIDERS` map in `integrations/<domain>/__init__.py`; add its package(s) as a `<domain>-<new_provider>` extra in `pyproject.toml` and append it to that domain's bundle extra and to `all` |
| New processing unit | `components/<name>.py` subclassing `Component[InputT, OutputT]`, see `components/README.md` |
| New orchestration flow | `pipelines/<name>.py` composing existing components, see `pipelines/README.md` |
| New agent tool | `agents/tools/<name>.py` implementing the `Tool` protocol, wrapped with `safe_call()`, see `agents/README.md` |
| New error type | `core/errors.py` (or the layer-local hierarchy) under the correct `PlatformError` branch |
| New shared data type | `core/schemas/<name>.py` + export from `core/schemas/__init__.py` |

When a change doesn't clearly belong to one directory, prefer the
narrowest-scoped directory and note the ambiguity in your summary of changes
rather than guessing silently.

## 6. General rules

- **Never import upward.** `core/` → nothing; `integrations/` → `core/`
  only; `components/` → `integrations/` + `core/`; `pipelines/` →
  `components/`; `agents/` → `pipelines/` + `components/`. This is not just
  convention: `[tool.importlinter]` in `pyproject.toml` enforces the layer
  order and that `core/` never imports a sibling layer; run `lint-imports`
  to check, CI fails the same way.
- **`core/` has no external deps** beyond `pydantic`, `abc`, `typing`,
  `uuid`, `datetime`, `opentelemetry` (tracing only).
- **All IO is async.** Use `asyncio.to_thread()` for blocking calls; `run()`
  on `Component` is a sync convenience wrapper around `arun()`, nothing more.
- **Backends are injected**, never instantiated inside a component or tool.
- **Pydantic v2** for all configs, schemas, and credentials (frozen models
  for credentials/configs).
- Import providers from the domain package, not the nested provider module:
  `from agent_platform.integrations.llm import OpenAILLM`, never
  `...llm.openai.openai import OpenAILLM`.
- No docstrings/comments unless the *why* is non-obvious, name things well
  instead.
- Don't introduce a new pattern (testing, error handling, folder structure)
  when an existing local convention already covers the case, check the
  directory README first.
- If a directory README and this file conflict, the directory README wins
  for anything scoped to that directory; this file wins for cross-cutting
  concerns (global build/test commands, repo-wide conventions).
- If no convention exists anywhere for what you're doing, pick the simplest
  approach consistent with the rest of the codebase and document it in the
  relevant README as part of your change.
- **Keep the directory README in sync with your change.** If what you did
  makes that directory's `README.md` inaccurate or incomplete, i.e. you
  added/removed/renamed a public component, changed an extension point,
  introduced a new convention, or fixed something the README lists as a
  known issue, update the README in the same change. Skip this for changes
  that don't shift what the README describes (internal refactors, bug fixes
  that don't change behavior described there, test-only changes).
