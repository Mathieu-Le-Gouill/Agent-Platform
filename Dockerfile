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

CMD ["platform"]