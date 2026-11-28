from pydantic import BaseModel

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.agents.tools.base import Tool
from agent_platform.core.schemas.message import AssistantMessage


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get weather"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return {"temp": 22}


class StrictWeatherInput(BaseModel):
    location: str


class StrictWeatherTool(Tool):
    """Rejects a call without `location`, to drive the corrective retry."""

    name = "strict_weather"
    description = "Get weather, requires a location"
    input_schema = StrictWeatherInput

    async def run(self, **kwargs):
        return {"temp": 22, "location": kwargs["location"]}


class Answer(BaseModel):
    value: int


class BlockingOutputGuardrail:
    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        raise AgentGuardrailError("output blocked")


class RewritingGuardrail:
    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        return AssistantMessage(content="[redacted]")


def invoke_agent_span(recorded_spans):
    spans = [s for s in recorded_spans.get_finished_spans() if s.name == "invoke_agent"]
    assert len(spans) == 1
    return spans[0]
