"""Loading plugins.

A plugin is any importable module (or file path) that defines

    def register(auditor: Auditor) -> None:
        auditor.add(my_stage)

Plugins can add, replace, remove or reconfigure stages. Installed packages can
also advertise plugins under the `ai_auditor.plugins` entry-point group; those
are only loaded when you call `load_entry_point_plugins` (or pass
`--entry-point-plugins` on the CLI), never implicitly.
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

from .core import Auditor

ENTRY_POINT_GROUP = "ai_auditor.plugins"


class PluginError(RuntimeError):
    pass


def _import(spec: str) -> ModuleType:
    path = Path(spec)
    if spec.endswith(".py") and path.exists():
        mod_spec = importlib.util.spec_from_file_location(f"ai_auditor_plugin_{path.stem}", path)
        if mod_spec is None or mod_spec.loader is None:
            raise PluginError(f"cannot load plugin file {spec}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(spec)
    except ImportError as e:
        raise PluginError(f"cannot import plugin '{spec}': {e}") from e


def load_plugin(auditor: Auditor, spec: str) -> None:
    """Import `spec` (module name or .py path) and call its register(auditor)."""
    module = _import(spec)
    register = getattr(module, "register", None)
    if not callable(register):
        raise PluginError(f"plugin '{spec}' has no register(auditor) function")
    register(auditor)


def load_entry_point_plugins(auditor: Auditor) -> list[str]:
    """Load every installed plugin advertised under the ai_auditor.plugins group."""
    loaded = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        register = ep.load()
        if not callable(register):
            raise PluginError(f"entry point '{ep.name}' does not point at a register(auditor) function")
        register(auditor)
        loaded.append(ep.name)
    return loaded
