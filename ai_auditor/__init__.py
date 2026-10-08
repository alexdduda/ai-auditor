"""ai-auditor: a framework for auditing AI-generated answers.

Core:     Auditor, Stage, stage, AuditContext, AuditResult, EVIDENCE
Models:   LLM (protocol), AnthropicLLM, Completion
Plugins:  load_plugin, load_entry_point_plugins
Built-in: ai_auditor.stages (the default audit)
"""

from .core import (
    EVIDENCE,
    AuditContext,
    AuditGraphError,
    Auditor,
    AuditResult,
    Stage,
    StageError,
    StageEvent,
    stage,
)
from .llm import LLM, AnthropicLLM, AuditGenerationError, Completion
from .plugins import PluginError, load_entry_point_plugins, load_plugin

__all__ = [
    "EVIDENCE", "AuditContext", "AuditGraphError", "Auditor", "AuditResult", "Stage", "StageError", "StageEvent",
    "stage", "LLM", "AnthropicLLM", "AuditGenerationError", "Completion", "PluginError", "load_plugin",
    "load_entry_point_plugins",
]
