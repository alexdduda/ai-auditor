import anthropic
from anthropic.types import Message

MODEL = "claude-opus-4-8"

_client: anthropic.Anthropic | None = None


class AuditGenerationError(RuntimeError):
    """Raised when a Claude call in the audit pipeline doesn't produce usable output."""


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def check_response(response: Message) -> Message:
    """Raise a clear error on a refusal or truncated response; otherwise pass through."""
    if response.stop_reason == "refusal":
        raise AuditGenerationError(
            "Claude declined to respond (stop_reason=refusal) — the problem or "
            "solution may have touched a restricted topic."
        )
    if response.stop_reason == "max_tokens":
        raise AuditGenerationError(
            "Claude's response was truncated before finishing "
            "(stop_reason=max_tokens); try a shorter problem statement."
        )
    return response


def extract_text(response: Message) -> str:
    """Check the response, then join its text blocks (skipping thinking/tool-use)."""
    check_response(response)
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
