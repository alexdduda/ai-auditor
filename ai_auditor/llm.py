"""The model interface stages use, and its Anthropic implementation.

Stages call `ctx.llm.complete(...)` or `ctx.llm.parse(...)` rather than the SDK
directly, so an audit can run against a different model, a recorded fixture,
or a fake in tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import anthropic
from anthropic.types import Message
from pydantic import BaseModel

from .style import STYLE_INSTRUCTION

DEFAULT_MODEL = "claude-opus-4-8"

T = TypeVar("T", bound=BaseModel)


class AuditGenerationError(RuntimeError):
    """A model call in the audit didn't produce usable output (refusal, truncation)."""


@dataclass
class Completion:
    text: str
    #: Types of every content block returned, e.g. "bash_code_execution_tool_result".
    block_types: list[str] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, prompt: str, *, effort: str = "medium", max_tokens: int = 4000, tools: list[dict] | None = None) -> Completion: ...

    def parse(self, prompt: str, schema: type[T], *, effort: str = "medium", max_tokens: int = 4000) -> T: ...


class AnthropicLLM:
    """Claude via the Anthropic SDK, with adaptive thinking."""

    def __init__(self, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None,
                 system: str | None = STYLE_INSTRUCTION):
        self.model = model
        self.system = system
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self) -> anthropic.Anthropic:
        with self._lock:
            if self._client is None:
                self._client = anthropic.Anthropic()
            return self._client

    def complete(self, prompt: str, *, effort: str = "medium", max_tokens: int = 4000, tools: list[dict] | None = None) -> Completion:
        kwargs: dict[str, Any] = {}
        if self.system:
            kwargs["system"] = self.system
        if tools:
            kwargs["tools"] = tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        check_response(response)
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        return Completion(text=text, block_types=[b.type for b in response.content])

    def parse(self, prompt: str, schema: type[T], *, effort: str = "medium", max_tokens: int = 4000) -> T:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            **({"system": self.system} if self.system else {}),
        )
        check_response(response)
        return response.parsed_output


def check_response(response: Message) -> Message:
    """Raise a clear error on a refusal or truncated response; otherwise pass through."""
    if response.stop_reason == "refusal":
        raise AuditGenerationError(
            "Claude declined to respond (stop_reason=refusal); the problem or "
            "solution may have touched a restricted topic."
        )
    if response.stop_reason == "max_tokens":
        raise AuditGenerationError(
            "Claude's response was truncated before finishing "
            "(stop_reason=max_tokens); try a shorter problem statement."
        )
    return response
