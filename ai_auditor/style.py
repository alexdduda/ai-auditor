"""Output style: no em dashes, no emojis.

Two layers: STYLE_INSTRUCTION asks the model (sent as the system prompt by
AnthropicLLM), and `clean` guarantees it on every generated output, including
string fields inside structured (pydantic) results.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

STYLE_INSTRUCTION = (
    "Formatting rules for everything you write: do not use em dashes; use commas, "
    "colons, parentheses or separate sentences instead. Do not use emojis."
)

# Spaced or unspaced em dash (and the horizontal bar U+2015) between words.
_EM_DASH = re.compile(r"\s*[—―]\s*")

# Emoji and pictograph ranges, plus the joiners/selectors/skin tones that build them.
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # mahjong, cards, emoticons, symbols & pictographs, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U00002B00-\U00002BFF"  # arrows/symbols block that holds star and circle emoji
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0000FE0E\U0000FE0F"   # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining keycap
    "]+"
)


def clean(text: str) -> str:
    """Remove emojis and replace em dashes with a comma."""
    text = _EMOJI.sub("", text)
    text = _EM_DASH.sub(", ", text)
    # Tidy what the substitutions can leave behind.
    text = re.sub(r",\s*([,.;:!?)])", r"\1", text)
    text = re.sub(r"(^|\n)[ \t]*,\s*", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+(?=\n|$|[.,;:!?)])", "", text)
    return text


def clean_value(value: Any) -> Any:
    """Apply `clean` to strings anywhere inside a stage output."""
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, BaseModel):
        return value.model_copy(update={k: clean_value(v) for k, v in value.__dict__.items()})
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clean_value(v) for v in value)
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    return value
