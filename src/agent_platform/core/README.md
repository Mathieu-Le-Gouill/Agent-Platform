# Core — Library-Independent Foundation

## Design

`core/` is the **zero-dependency foundation** of the platform. It defines everything that does not depend on an external library — pure Python ABCs, Pydantic schemas, enums, error types, and the provider registry. No integration, component, pipeline, or agent ever imports from outside `core/` for its base abstractions.

## Directory Layout

```
core/
├── base.py           # Entity (UUID mixin), Timestamped
├── errors.py         # PlatformError hierarchy (ProviderError, ConfigError, LLMError, AgentError, …)
├── registry.py       # ProviderRegistry[T] — generic name→provider lookup
├── credentials.py    # BaseCredentials, NoCredentials, ProviderCredentials
├── interfaces/       # ABCs for every capability (the "contract" layer)
│   ├── llm/
│   ├── embeddings/
│   ├── vad/
│   ├── ocr/
│   ├── vector_store/
│   ├── reranking/
│   ├── chunking/
│   ├── speech/
│   ├── translation/
│   ├── clustering/
│   ├── classification/
│   ├── loader/
│   └── image_generation/
└── schemas/          # Pydantic v2 data structures (shared across all layers)
    ├── document.py
    ├── chunk.py
    ├── embedding.py
    ├── score.py
    ├── token.py
    ├── span.py
    ├── cluster.py
    ├── message.py
    ├── conversation.py
    └── enums.py
```

### `interfaces/` — The Contract Layer

Each subdirectory follows a fixed pattern (example: `interfaces/llm/`):

```
interfaces/<domain>/
├── base.py           # ABC (e.g. BaseLLMProvider) with abstract async methods
├── config.py         # Pydantic config model for the domain
└── response.py       # Pydantic response model(s)
```

An interface ABC **never imports from `integrations/`**, **never imports from `components/`**, and **never imports from `pipelines/`**. It may only use `core/schemas/`, `core/errors.py`, and `core/credentials.py`.

### `schemas/` — Shared Data Structures

All Pydantic v2 models live here. They are the currency passed between layers — a component returns a schema type, a pipeline accepts one, a tool wraps one.

### `errors.py` — Error Hierarchy

```
PlatformError
├── ProviderError
├── ConfigError
├── NotFoundError
├── ValidationError
├── MissingCredentialError
├── LLMError
│   ├── LLMGenerationError
│   ├── LLMTimeoutError
│   └── LLMRateLimitError
├── AgentError
│   ├── AgentThinkError
│   ├── AgentActError
│   └── AgentMaxIterations
└── ToolError
```

Every error carries `code`, `retryable`, and `context` fields.

### `registry.py` — ProviderRegistry

A generic `dict[str, type[T]]` with `register()`, `get()`, `all()`. Used by Integrations to register themselves.

## How to Extend

### Add a new interface (new capability)

```
core/interfaces/<new_domain>/
├── __init__.py
├── base.py           # class Base<NewDomain>Provider(ABC): …
├── config.py         # class <NewDomain>Config(BaseModel): …
└── response.py       # class <NewDomain>Response(BaseModel): …
```

1. Define the ABC in `base.py` with `@abstractmethod` async methods
2. Define the config model in `config.py`
3. Define response models in `response.py`
4. Export from `core/interfaces/<domain>/__init__.py`
5. If the domain defines a new shared data type, add it to `core/schemas/`

### Add a new schema

1. Add a new file in `core/schemas/` (or extend an existing one)
2. Use `pydantic.BaseModel`
3. Export from `core/schemas/__init__.py`

### Add a new error

1. Add the class in `core/errors.py`
2. Add to `__all__` in the same file
3. Place it in the correct branch of the `PlatformError` hierarchy

### Add a new enum

1. Add to `core/schemas/enums.py`
2. Export from `core/schemas/__init__.py`

## Constraints

- **No imports from `integrations/`, `components/`, `pipelines/`, or `agents/`**
- **No external library imports** (no langchain, no openai, no numpy, etc.) — only `pydantic`, `abc`, `typing`, `uuid`, `datetime`
- ABCs use `Generic` TypeVars for type safety where appropriate
