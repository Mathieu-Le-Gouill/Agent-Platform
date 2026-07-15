import pytest

from agent_platform.core.schemas.enums import Language
from agent_platform.core.schemas.message import (
    AssistantMessage,
    MessageRole,
    Prompt,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)


class TestMessageRole:
    def test_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris"}

    def test_is_frozen(self):
        tc = ToolCall(id="call_1", name="fn", arguments={})
        with pytest.raises(ValueError):
            tc.id = "call_2"


class TestToolResult:
    def test_construction(self):
        tr = ToolResult(tool_call_id="call_1", name="get_weather", content="sunny")
        assert tr.tool_call_id == "call_1"
        assert tr.name == "get_weather"
        assert tr.content == "sunny"
        assert tr.is_error is False

    def test_is_error_true(self):
        tr = ToolResult(
            tool_call_id="call_1", name="fn", content="error", is_error=True
        )
        assert tr.is_error is True

    def test_is_error_default(self):
        tr = ToolResult(tool_call_id="c1", name="fn", content="ok")
        assert tr.is_error is False

    def test_is_frozen(self):
        tr = ToolResult(tool_call_id="c1", name="fn", content="ok")
        with pytest.raises(ValueError):
            tr.content = "changed"


class TestPromptBuild:
    def test_system_only(self):
        prompt = Prompt.build(system="You are a bot")
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], SystemMessage)
        assert prompt.messages[0].content == "You are a bot"

    def test_user_only(self):
        prompt = Prompt.build(user="Hello")
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], UserMessage)
        assert prompt.messages[0].content == "Hello"

    def test_history_only(self):
        msgs = [UserMessage(content="previous")]
        prompt = Prompt.build(history=msgs)
        assert len(prompt.messages) == 1
        assert prompt.messages[0] is msgs[0]

    def test_all_together(self):
        prompt = Prompt.build(
            system="Sys", user="User", history=[AssistantMessage(content="Hi")]
        )
        assert len(prompt.messages) == 3
        assert isinstance(prompt.messages[0], SystemMessage)
        assert isinstance(prompt.messages[1], AssistantMessage)
        assert isinstance(prompt.messages[2], UserMessage)

    def test_with_language(self):
        prompt = Prompt.build(system="Sys", user="User", language=Language.FR)
        assert prompt.messages[0].language == Language.FR
        assert prompt.messages[1].language == Language.FR

    def test_empty(self):
        prompt = Prompt.build()
        assert prompt.messages == []

    def test_system_with_language(self):
        prompt = Prompt.build(system="Hello", language=Language.EN)
        assert prompt.messages[0].language == Language.EN

    def test_user_with_language(self):
        prompt = Prompt.build(user="Bonjour", language=Language.FR)
        assert prompt.messages[0].language == Language.FR


class TestPromptMethods:
    def test_add_system(self):
        prompt = Prompt().add_system("You are a bot")
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], SystemMessage)
        assert prompt.messages[0].content == "You are a bot"

    def test_add_user(self):
        prompt = Prompt().add_user("Hello")
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], UserMessage)
        assert prompt.messages[0].content == "Hello"

    def test_add_assistant(self):
        prompt = Prompt().add_assistant("Hi")
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], AssistantMessage)
        assert prompt.messages[0].content == "Hi"
        assert prompt.messages[0].tool_calls == []

    def test_add_assistant_with_tool_calls(self):
        tc = ToolCall(id="1", name="fn", arguments={"x": 1})
        prompt = Prompt().add_assistant("Using tool", tool_calls=[tc])
        assert prompt.messages[0].tool_calls == [tc]
        assert prompt.messages[0].tool_calls[0] is tc

    def test_add_tool_result(self):
        tr = ToolResult(tool_call_id="1", name="fn", content="done")
        prompt = Prompt().add_tool_result(result=tr)
        assert len(prompt.messages) == 1
        assert isinstance(prompt.messages[0], ToolMessage)
        assert prompt.messages[0].result == tr

    def test_add_tool_result_with_kwargs(self):
        tr = ToolResult(tool_call_id="1", name="fn", content="done")
        prompt = Prompt().add_tool_result(result=tr, language=Language.EN)
        assert prompt.messages[0].language == Language.EN

    def test_chaining(self):
        prompt = (
            Prompt()
            .add_system("Sys")
            .add_user("User")
            .add_assistant("Assist")
            .add_tool_result(ToolResult(tool_call_id="1", name="fn", content="r"))
        )
        assert len(prompt.messages) == 4

    def test_add_system_with_kwargs(self):
        prompt = Prompt().add_system("Sys", language=Language.FR)
        assert prompt.messages[0].language == Language.FR

    def test_add_user_with_kwargs(self):
        prompt = Prompt().add_user("U", language=Language.EN)
        assert prompt.messages[0].language == Language.EN

    def test_add_assistant_with_kwargs(self):
        prompt = Prompt().add_assistant("A", language=Language.GE)
        assert prompt.messages[0].language == Language.GE


class TestLastUserMessage:
    def test_found(self):
        prompt = Prompt(messages=[SystemMessage(content="S"), UserMessage(content="U")])
        msg = prompt.last_user_message()
        assert msg is not None
        assert msg.content == "U"

    def test_none_when_no_user_message(self):
        prompt = Prompt(messages=[SystemMessage(content="S")])
        assert prompt.last_user_message() is None

    def test_returns_most_recent(self):
        prompt = Prompt(
            messages=[UserMessage(content="First"), UserMessage(content="Last")]
        )
        msg = prompt.last_user_message()
        assert msg.content == "Last"

    def test_none_when_empty(self):
        prompt = Prompt()
        assert prompt.last_user_message() is None

    def test_none_with_only_assistant(self):
        prompt = Prompt(messages=[AssistantMessage(content="Hi")])
        assert prompt.last_user_message() is None

    def test_none_with_only_tool(self):
        tr = ToolResult(tool_call_id="1", name="fn", content="r")
        prompt = Prompt(messages=[ToolMessage(result=tr)])
        assert prompt.last_user_message() is None


class TestSystemPrompt:
    def test_found(self):
        prompt = Prompt(
            messages=[SystemMessage(content="Sys Prompt"), UserMessage(content="U")]
        )
        assert prompt.system_prompt() == "Sys Prompt"

    def test_none_when_no_system_message(self):
        prompt = Prompt(messages=[UserMessage(content="U")])
        assert prompt.system_prompt() is None

    def test_none_when_empty(self):
        prompt = Prompt()
        assert prompt.system_prompt() is None

    def test_returns_first_system_message(self):
        prompt = Prompt(
            messages=[
                SystemMessage(content="First"),
                SystemMessage(content="Second"),
            ]
        )
        assert prompt.system_prompt() == "First"
