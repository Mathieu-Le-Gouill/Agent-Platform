# Working notes: runtime-state-limits (2026-11-20)

Subject: runtime state and cost limits in embeddable Python agent libraries (Pydantic-AI, LangGraph, OpenAI Agents SDK).
Outcome: state of the art + architecture for agent_platform (core protocols + one reference backend in integrations/).
Depth: standard. Constraints: library embedded in host products; single instance first; no tenancy/auth in library;
state saved per completed turn (no mid-run checkpoints).

## Prior beliefs (hypotheses)

- H1 OpenAI Agents SDK has a `Session` protocol (get_items/add_items/pop_item/clear_session) with SQLite, Redis, SQLAlchemy backends; items stored as JSON of Responses-API items. [needs evidence]
- H2 Pydantic-AI has no session store in the library; host persists `message_history` via `ModelMessagesTypeAdapter`; new part types stay backward-compatible. [possibly outdated: a `Conversation` object and storage docs now exist]
- H3 LangGraph `BaseCheckpointSaver` (put/put_writes/get_tuple/list) with InMemory/Sqlite/Postgres savers, `JsonPlusSerializer`, durability modes sync/async/exit. [needs evidence]
- H4 Pydantic-AI `UsageLimits` (request, input/output/total tokens, tool calls) raises `UsageLimitExceeded`, checked before each request and after each response; per-conversation needs carried usage. [needs evidence]
- H5 OpenAI Agents SDK and LangGraph have no built-in token cap; LangGraph caps steps via recursion_limit, Agents SDK via max_turns. [needs research]
- H6 Pydantic-AI `FallbackModel` falls back only on `ModelHTTPError` by default (`fallback_on`); retries live in an HTTP transport (tenacity). [possibly outdated]
- H7 MCP stdio clients in Pydantic-AI and Agents SDK have an init/session timeout param and are async context managers; cleanup across tasks hits anyio cancel-scope errors. [needs evidence]

## Map

Terms: session, memory, conversation store, message history, thread, checkpoint, checkpointer, saver, durable execution,
usage limits, token budget, max_turns, recursion_limit, fallback model, retry policy, transient vs permanent error,
MCP client session, stdio transport, lifespan, AsyncExitStack.
Conflated pairs: docs claim vs observed behavior in issues; persistence per turn vs per step; library feature vs host pattern.
Components (dependency order) and interfaces:
1. Conversation store: receives the message list of a completed turn from ConversationAgent; returns history on load.
2. Workflow checkpoint store: receives WorkflowCheckpoint after each node; returns latest for resume.
3. Usage limits: receives TokenUsage per LLM response inside the agent loop; raises a typed error; per-conversation total persisted with component 1.
4. LLM error policy: classifies errors transient/permanent; retry layer and fallback chain consume it.
5. MCP client lifecycle: startup handshake timeout, shutdown cleanup of all clients.

Dimensions: d1 = component 1; d2 = component 2; d3 = component 3; d4 = components 4+5; d5 = disconfirming evidence.
