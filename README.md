# Agent Platform

A modular Python platform for building AI-powered conversational systems. Unifies audio, text, and image processing into composable pipelines and agent skills. Async-first, Pydantic v2, built on a strict layered architecture.

## Architecture

```
Skills / Agents   (LLM + Tools → autonomous behavior)
      ↓ uses
Tools             (call Components or Pipelines)
      ↓ calls
Pipelines         (orchestrate multiple Components)
      ↓ composes
Components        (compose one or more Integrations)
      ↓ composes
Integrations      (concrete providers for Core interfaces)
      ↓ implements
Core              (interfaces + schemas, no external deps except opentelemetry-api for tracing)
```

Each layer may only import from layers below it. See `AGENTS.md` for full rules.

## Directory Layout

```
src/agent_platform/
├── core/              # ABCs, schemas, error hierarchy, registry, tracing; zero external deps except opentelemetry-api
├── integrations/      # Concrete provider implementations (OpenAI, Anthropic, WhisperX, …)
├── components/        # Reusable processing units composing integrations
├── pipelines/         # Multi-step orchestration flows
├── agents/            # LLM reasoning loop, tool registry, conversation management
├── audio/             # Audio I/O + DSP utilities
├── utils/             # Shared helpers
├── config/            # Settings, logging, DI container (build_agent, build_provider)
├── api/               # FastAPI app, /chat and /health
└── workflows/         # LangGraph state machine (empty stub)
```

## Integrations at a Glance

| Domain | Providers |
|---|---|
| LLM | OpenAI, Anthropic, Mistral, Ollama, HuggingFace |
| Embeddings | OpenAI, Mistral, Ollama, HuggingFace |
| Vector Store | Chroma, FAISS, Pinecone, Qdrant, Weaviate |
| OCR | Tesseract, Google Vision, AWS Textract, Mistral |
| Speech-to-Text | WhisperX, Faster-Whisper, Deepgram, OpenAI |
| Translation | DeepL, Google Translate, Azure |
| Reranking | Cohere, Jina, HuggingFace, FlashRank, Voyage |
| Chunking | Recursive, Markdown, HTML, LaTeX, PDF |
| VAD | Silero, WebRTC, PVCobra, TEN |
| Clustering | HDBSCAN, KMeans, GMM |
| Loader | Unstructured, PIL, PyAV, SoundFile |
| Image Generation | DALL-E, Stable Diffusion, Midjourney |
| Classification | Transformers (zero-shot) |

## Setup

Provider SDKs are installed via extras, not bundled by default, so you only pull in what you use. Use [`uv`](https://docs.astral.sh/uv/) to create and run against the project's venv, it finds `.venv` automatically (creating it on first use) by walking up to `pyproject.toml`, so no venv path needs to be activated or remembered:

```bash
# Base install (framework only, no provider SDKs) + dev tooling, run any command to trigger it
uv run --extra dev pytest -v

# Add just the providers you need, e.g. OpenAI LLM + Chroma vector store
uv run --extra dev --extra llm-openai --extra vector-store-chroma pytest -v

# Or grab every provider in a domain
uv run --extra dev --extra llm pytest -v

# Or everything (parity with the old monolithic install); use pip here, not uv, see note below
pip install -e ".[dev,all]"
```

Extras are named `<domain>-<provider>` (e.g. `stt-whisperx`, `ocr-tesseract`); see `pyproject.toml` for the full list. Requires Python 3.11. Some providers have system dependencies (Tesseract, CUDA for WhisperX). No `uv` installed? `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"` works the same, `uv` just removes the activation step.

## Development

```bash
# Run tests
uv run pytest -v

# Run a specific test
uv run pytest -v tests/agents/test_executor.py::test_max_iterations

# Lint
uv run ruff check src/
```

### Git hooks

One-time setup after cloning:

```bash
uv run --extra dev pre-commit install --hook-type pre-commit --hook-type pre-push
```

This runs `ruff check`, `ruff format`, `mypy`, and `lint-imports` on every commit, and the full `pytest` suite before every push, the same checks CI runs on `main`. Config lives in `.pre-commit-config.yaml`. To run everything on demand without committing: `uv run pre-commit run --all-files`.

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <summary>`, e.g. `feat: add streaming support to ConversationAgent`, `fix: repair broken CI`, `docs: update README`. Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## For Coding Agents

Read `AGENTS.md` before making any changes. It documents the lookup protocol, layer rules, error hierarchy, testing conventions, and extension guidance.
