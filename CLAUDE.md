# CLAUDE.md: Agent Platform

Read `AGENTS.md` first - it is the primary authority for architecture, layer rules, testing/error conventions, and extension guidance.

## Quick orientation

- **`src/agent_platform/README.md`** - layer map and known issues
- **`src/agent_platform/<layer>/README.md`** - local patterns for that layer; read before touching it

## Current state

The platform is structurally complete at the integration, component, pipeline, agent, evals, and workflows layers. The `api/` directory has a real FastAPI entrypoint.

## What is solid

All layers: core, integrations (17 domains, incl. MCP and tracing), components, pipelines (RAG ingest/query, speech translation), agents (Agent, AgentExecutor, ConversationAgent, tools, guardrails), evals (EvalCase/Runner/Scorer, CLI), workflows (WorkflowGraph, agent/tool/handoff nodes, checkpointing), tooling (mypy, ruff, import-linter, pytest with coverage gate ≥90%).

Observability is end to end: spans for LLM calls, tool calls (incl. kind/source/payload facts), the agent loop, guardrails, workflow nodes, components, and pipelines. `core/observability/` owns the vocabulary as typed `Observation`s (`observe()`/`emit()`/`current()`, `@Fact`-annotated fields deriving span attributes and payload policy from one object) plus the `BaseTracer` contract; `integrations/tracing/langfuse/` owns the only backend (the Langfuse SDK, on with `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, a silent no-op without them), and its `mapping.py` is the single place platform names become Langfuse fields. Prompt/payload content capture is opt-in (`AGENT_PLATFORM_CAPTURE_CONTENT`); size and digest are always exported.

## Near-term build priorities

Detailed phased roadmap lives in `src/agent_platform/agents/README.md` ("Roadmap: closing the gap with modern agent platforms"). All phases (1-4: evals, harness hardening, tooling ecosystem, `workflows/` orchestration, production hygiene/observability) are done. No open roadmap items remain there; new work starts from a fresh gap analysis rather than this roadmap.

## Keeping this file current

Update the relevant section when a change closes a known issue, completes a priority item, or makes the current state snapshot inaccurate. Refresh the date below.

Last updated: 2026-11-14 (tracing backend is Langfuse only: the OpenTelemetry/OTLP backend and its `semconv.py` were replaced by `integrations/tracing/langfuse/`)
