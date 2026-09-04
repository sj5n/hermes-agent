"""Per-request enabled_toolsets narrowing on the api_server platform.

Motivating use case: a client-side "incognito" mode that drops the
memory/skills toolsets for one turn so nothing about that conversation can
be persisted (the background memory-review trigger and skill writes are
gated on tool *availability*, not a session flag). The override is
subtractive-only: a request can narrow the platform's configured toolsets
but can never grant one the config doesn't already expose.
"""
from unittest.mock import patch, MagicMock

from gateway.platforms.api_server import _request_toolset_override, _request_agent_overrides


class TestRequestToolsetOverride:
    def test_absent_field_is_no_override(self):
        assert _request_toolset_override({}) is None
        assert _request_toolset_override({"messages": []}) is None

    def test_non_list_field_is_ignored(self):
        assert _request_toolset_override({"enabled_toolsets": "memory"}) is None
        assert _request_toolset_override({"enabled_toolsets": None}) is None

    def test_list_is_cleaned(self):
        out = _request_toolset_override(
            {"enabled_toolsets": ["web", " terminal ", "", 42, "file"]}
        )
        assert out == ["web", "terminal", "file"]

    def test_request_agent_overrides_carries_it_through(self):
        overrides = _request_agent_overrides(
            {"model": "x", "enabled_toolsets": ["web", "terminal"]}
        )
        assert overrides["enabled_toolsets_override"] == ["web", "terminal"]

    def test_request_agent_overrides_omits_key_when_absent(self):
        overrides = _request_agent_overrides({"model": "x"})
        assert "enabled_toolsets_override" not in overrides


class TestCreateAgentAppliesToolsetOverride:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_override_intersects_with_platform_config(self):
        """A request can only ever REMOVE toolsets, never add one the
        platform config doesn't already grant."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch(
                 "hermes_cli.tools_config._get_platform_tools",
                 return_value={"web", "terminal", "memory", "skills", "session_search"},
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            # Incognito request: narrow to everything except memory/skills.
            # "browser" isn't in the platform config at all — must NOT
            # appear in the effective toolset (subtractive-only contract).
            adapter._create_agent(
                enabled_toolsets_override=["web", "terminal", "session_search", "browser"],
            )

            call_kwargs = mock_agent_cls.call_args
            toolsets = set(call_kwargs.kwargs.get("enabled_toolsets"))
            assert toolsets == {"web", "terminal", "session_search"}
            assert "memory" not in toolsets
            assert "skills" not in toolsets
            assert "browser" not in toolsets  # never granted by config

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_no_override_keeps_full_platform_config(self):
        """Omitting the field entirely must not change today's behavior."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch(
                 "hermes_cli.tools_config._get_platform_tools",
                 return_value={"web", "terminal", "memory"},
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {
                "api_key": "test-key", "base_url": None, "provider": None,
                "api_mode": None, "command": None, "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            call_kwargs = mock_agent_cls.call_args
            toolsets = set(call_kwargs.kwargs.get("enabled_toolsets"))
            assert toolsets == {"web", "terminal", "memory"}
