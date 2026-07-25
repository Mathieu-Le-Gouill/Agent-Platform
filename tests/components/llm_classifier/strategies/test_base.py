from agent_platform.components.llm_classifier.strategies.base import PromptStrategy


class _ConcreteStrategy:
    def build_system_prompt(self, config, candidate_labels, *, multi_label):
        return "prompt"


def test_concrete_implementation_satisfies_protocol():
    strategy: PromptStrategy = _ConcreteStrategy()
    assert strategy.build_system_prompt(None, [], multi_label=False) == "prompt"
