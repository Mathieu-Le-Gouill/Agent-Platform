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
Core              (interfaces + schemas, no external deps)
```

Each layer may only import from layers below it. See `AGENTS.md` for full rules.

## Directory Layout

```
src/agent_platform/
├── core/              # ABCs, schemas, error hierarchy, registry — zero external deps
├── integrations/      # Concrete provider implementations (OpenAI, Anthropic, WhisperX, …)
├── components/        # Reusable processing units composing integrations
├── pipelines/         # Multi-step orchestration flows
├── agents/            # LLM reasoning loop, tool registry, conversation management
├── audio/             # Audio I/O + DSP utilities
├── utils/             # Shared helpers
└── config/            # Logging + wiring
```

## Integrations at a Glance

| Domain | Providers |
|---|---|
| LLM | OpenAI, Anthropic, Mistral, Ollama, HuggingFace |
| Embeddings | OpenAI, Mistral, Ollama, HuggingFace |
| Vector Store | Chroma, Qdrant |
| OCR | Tesseract, Google Vision, AWS Textract |
| Speech-to-Text | WhisperX, Deepgram |
| Translation | DeepL, Google Translate |
| Reranking | Cohere, Jina, HuggingFace |
| Chunking | Recursive (LangChain) |
| VAD | Silero, WebRTC, PVCobra, TEN |
| Clustering | HDBSCAN, KMeans |
| Classification | Transformers, OpenAI |
| Loader | Unstructured, PIL, PyAV, SoundFile |
| Image Generation | DALL-E, Stable Diffusion, Midjourney |

## Setup

Provider SDKs are installed via extras, not bundled by default, so you only pull in what you use:

```bash
# Base install (framework only, no provider SDKs) + dev tooling
pip install -e ".[dev]"

# Add just the providers you need, e.g. OpenAI LLM + Chroma vector store
pip install -e ".[dev,llm-openai,vector-store-chroma]"

# Or grab every provider in a domain
pip install -e ".[dev,llm]"

# Or everything (parity with the old monolithic install)
pip install -e ".[dev,all]"
```

Extras are named `<domain>-<provider>` (e.g. `stt-whisperx`, `ocr-tesseract`); see `pyproject.toml` for the full list. Requires Python 3.11. Some providers have system dependencies (Tesseract, CUDA for WhisperX).

## Development

```bash
# Run tests
pytest -v

# Run a specific test
pytest -v tests/agents/test_executor.py::test_max_iterations

# Lint
ruff check src/
```

## For Coding Agents

Read `AGENTS.md` before making any changes. It documents the lookup protocol, layer rules, error hierarchy, testing conventions, and extension guidance.
