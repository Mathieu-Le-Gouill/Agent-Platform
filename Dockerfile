FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src ./src

# Production dependencies
RUN pip install --no-cache-dir .

FROM base AS test

COPY tests ./tests

RUN pip install --no-cache-dir ".[dev]"

CMD ["pytest", "-v"]

FROM base AS production

# Extras for the default Settings models, plus Langfuse tracing.
RUN pip install --no-cache-dir ".[api,llm-openai,image-dalle,stt-openai,tracing]"

# Bind all interfaces inside the container only; the Settings default stays loopback.
ENV AGENT_PLATFORM_API_HOST=0.0.0.0

EXPOSE 8000

CMD ["agent-platform-api"]
