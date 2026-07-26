from agent_platform.integrations.moderation.local.config import LocalModerationConfig
from agent_platform.integrations.moderation.local.provider import LocalModeration


class TestLocalModeration:
    def test_default_config_has_rules(self):
        config = LocalModerationConfig()
        assert "violence" in config.rules

    def test_moderate_flags_matching_text(self):
        provider = LocalModeration()
        result = provider.moderate("I will murder you")

        assert result.flagged is True
        violence = next(c for c in result.categories if c.name == "violence")
        assert violence.flagged is True
        assert violence.score.value == 1.0

    def test_moderate_clean_text_not_flagged(self):
        provider = LocalModeration()
        result = provider.moderate("What a lovely day for a walk")

        assert result.flagged is False
        assert all(not c.flagged for c in result.categories)

    def test_moderate_is_case_insensitive(self):
        provider = LocalModeration()
        result = provider.moderate("I WILL MURDER YOU")

        assert result.flagged is True

    def test_moderate_uses_custom_rules(self):
        provider = LocalModeration()
        config = LocalModerationConfig(rules={"custom": ["forbidden phrase"]})
        result = provider.moderate("this contains a forbidden phrase", config=config)

        assert result.flagged is True
        assert result.categories[0].name == "custom"

    async def test_amoderate_matches_sync(self):
        provider = LocalModeration()
        sync_result = provider.moderate("I will murder you")
        async_result = await provider.amoderate("I will murder you")

        assert async_result == sync_result
