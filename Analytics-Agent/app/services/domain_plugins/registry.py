"""app/services/domain_plugins/registry.py — PluginRegistry for the Agent
3 redesign (plan "zany-giggling-crayon").

Deliberately a small, explicit list rather than filesystem auto-discovery
(no `config/domains/*/plugin.py` globbing) — Phase 2 only has one real
plugin (Insurance); auto-discovery adds real complexity for zero benefit
until there's more than one to discover. Adding a plugin later is still a
one-line change here, not a redesign.
"""

from __future__ import annotations

from app.services.domain_plugins.base import DomainPlugin
from app.services.domain_plugins.insurance.plugin import InsurancePlugin

_PLUGINS: list[DomainPlugin] = [
    InsurancePlugin(),
]


class PluginRegistry:
    def __init__(self, plugins: list[DomainPlugin] | None = None):
        self._plugins = list(plugins) if plugins is not None else list(_PLUGINS)

    def find_plugin(self, detected_domain: str | None) -> DomainPlugin | None:
        if not detected_domain:
            return None
        for plugin in self._plugins:
            if plugin.applies_to(detected_domain):
                return plugin
        return None
