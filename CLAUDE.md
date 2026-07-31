# CLAUDE.md: Agent Platform

Read `AGENTS.md` first - it is the primary authority for architecture, layer rules, testing/error conventions, and extension guidance.

## Quick orientation

- **`src/agent_platform/README.md`** - layer map and known issues
- **`src/agent_platform/<layer>/README.md`** - local patterns for that layer; read before touching it

## Current state

The platform is structurally complete at the integration, component, pipeline, agent, evals, and workflows layers. The `api/` directory has a real FastAPI entrypoint.

## What is solid

All layers: core, integrations (15 domains, incl. MCP), components, pipelines (RAG ingest/query, speech translation), agents (Agent, AgentExecutor, ConversationAgent, tools, guardrails), evals (EvalCase/Runner/Scorer, CLI), workflows (WorkflowGraph, agent/tool/handoff nodes, checkpointing), tooling (mypy, ruff, import-linter, pytest with coverage gate ≥90%).

## Near-term build priorities

Detailed phased roadmap lives in `src/agent_platform/agents/README.md` ("Roadmap: closing the gap with modern agent platforms"). Phases 1-3 (evals, harness hardening, tooling ecosystem) and Phase 2 (`workflows/` orchestration) are done. Phase 4 (production hygiene / observability polish: session-id span attribute, provider circuit breaker, token-usage rollup) is next.

## Keeping this file current

Update the relevant section when a change closes a known issue, completes a priority item, or makes the current state snapshot inaccurate. Refresh the date below.

Last updated: 2026-07-31
