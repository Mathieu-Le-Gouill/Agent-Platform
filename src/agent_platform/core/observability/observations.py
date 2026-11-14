from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from agent_platform.core.observability.facts import Fact, Observation

__all__ = [
    "AgentRun",
    "ComponentRun",
    "GuardrailCheck",
    "Handoff",
    "LLMBatch",
    "LLMCall",
    "ModelCost",
    "PipelineRun",
    "Retrieval",
    "ToolKind",
    "ToolRun",
    "WorkflowNode",
]


class ToolKind(StrEnum):
    """How a tool is executed, as the platform sees it.

    `REMOTE` covers tools the platform hands to something outside itself,
    which is what an MCP server is; `ToolRun.source` carries the finer-grained
    "it was MCP specifically".
    """

    FUNCTION = "function"
    REMOTE = "remote"
    DATASTORE = "datastore"


# --- MODEL CALLS ---


@dataclass
class LLMCall(Observation):
    """One chat completion, buffered or streamed."""

    operation = "chat"

    provider: Annotated[str, Fact("llm.provider")]
    request_model: Annotated[str, Fact("llm.request.model")]
    response_model: Annotated[str | None, Fact("llm.response.model")] = None
    response_id: Annotated[str | None, Fact("llm.response.id")] = None
    finish_reasons: Annotated[
        tuple[str, ...] | None, Fact("llm.response.finish_reasons")
    ] = None
    tool_call_count: Annotated[int | None, Fact("llm.response.tool_call_count")] = None
    input_tokens: Annotated[int | None, Fact("llm.usage.input_tokens")] = None
    output_tokens: Annotated[int | None, Fact("llm.usage.output_tokens")] = None


@dataclass
class LLMBatch(Observation):
    """One step of a provider's offline batch API."""

    stage: str
    provider: Annotated[str, Fact("llm.provider")]
    batch_id: Annotated[str | None, Fact("llm.batch.id")] = None

    @property
    def operation_name(self) -> str:
        return f"llm_batch_{self.stage}"


@dataclass
class ModelCost(Observation):
    """What a model call cost, priced after the fact.

    An event rather than a span: the call it prices has already ended by the
    time an estimator can look at its token usage, so this lands on whatever
    span encloses the caller.
    """

    agent_name: Annotated[str, Fact("agent.name")]
    model: Annotated[str, Fact("llm.request.model")]
    cost: Annotated[float, Fact("llm.usage.cost")]


# --- AGENTS ---


@dataclass
class ToolRun(Observation):
    """One tool call the agent executed."""

    operation = "execute_tool"

    tool_name: Annotated[str, Fact("tool.name")]
    call_id: Annotated[str | None, Fact("tool.call.id")] = None
    kind: Annotated[str | None, Fact("tool.kind")] = None
    source: Annotated[str | None, Fact("tool.source")] = None
    description: Annotated[str | None, Fact("tool.description")] = None
    arguments: Annotated[Any | None, Fact("tool.arguments", payload=True)] = None
    result: Annotated[Any | None, Fact("tool.result", payload=True)] = None
    is_error: Annotated[bool | None, Fact("tool.result.is_error")] = None
    stream_chunks: Annotated[int | None, Fact("tool.stream.chunks")] = None


@dataclass
class AgentRun(Observation):
    """One run of the agent think/act loop, however it ended."""

    operation = "invoke_agent"

    agent_name: Annotated[str, Fact("agent.name")]
    conversation_id: Annotated[str | None, Fact("conversation.id")] = None
    max_iterations: Annotated[int | None, Fact("agent.max_iterations")] = None
    iterations: Annotated[int, Fact("agent.iterations")] = 0
    exit_reason: Annotated[str | None, Fact("agent.exit_reason")] = None
    recovery_attempted: Annotated[bool, Fact("agent.recovery_attempted")] = False
    tool_calls_requested: Annotated[int, Fact("agent.tool_calls_requested")] = 0
    tool_calls_executed: Annotated[int, Fact("agent.tool_calls_executed")] = 0


@dataclass
class GuardrailCheck(Observation):
    """One guardrail decision, on the way in or on the way out."""

    operation = "guardrail"

    guardrail_name: Annotated[str, Fact("guardrail.name")]
    stage: Annotated[str, Fact("guardrail.stage")]
    blocked: Annotated[bool, Fact("guardrail.blocked")] = False


# --- ORCHESTRATION ---


@dataclass
class WorkflowNode(Observation):
    """One node of a workflow graph, and where the run went next."""

    operation = "workflow_node"

    node: Annotated[str, Fact("workflow.node")]
    step: Annotated[int | None, Fact("workflow.step")] = None
    next_node: Annotated[str | None, Fact("workflow.next_node")] = None
    route: Annotated[str | None, Fact("workflow.route")] = None


@dataclass
class Handoff(Observation):
    """Who took over a workflow node. An event on the enclosing node span."""

    to: Annotated[str, Fact("workflow.handoff.to")]


@dataclass
class PipelineRun(Observation):
    """One pipeline entry point."""

    operation = "pipeline"

    name: Annotated[str, Fact("pipeline.name")]


@dataclass
class Retrieval(Observation):
    """How much context a retrieval step actually found.

    An event: it describes a step inside the enclosing pipeline span rather
    than a unit of work worth timing on its own.
    """

    top_k: Annotated[int, Fact("retrieval.top_k")]
    hits: Annotated[int, Fact("retrieval.hits")]


@dataclass
class ComponentRun(Observation):
    """One `Component.arun`, named after the component's own operation."""

    name: str

    @property
    def operation_name(self) -> str:
        return self.name
