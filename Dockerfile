FROM python:3.12-slim AS base

# Same uv version as the one that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /bin/uv

# Install into one venv on PATH with the image's Python; never download another.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

# Production dependencies, at the versions pinned in uv.lock. --locked fails the
# build when pyproject.toml changed without a re-lock.
RUN uv sync --locked --no-editable

FROM base AS test

COPY tests ./tests

RUN uv sync --locked --no-editable --extra dev

CMD ["pytest", "-v"]

FROM base AS production

# Extras for the default Settings models, plus Langfuse tracing.
RUN uv sync --locked --no-editable --extra api --extra llm-openai --extra image-dalle --extra stt-openai --extra tracing

# Bind all interfaces inside the container only; the Settings default stays loopback.
ENV AGENT_PLATFORM_API_HOST=0.0.0.0

EXPOSE 8000

CMD ["agent-platform-api"]
