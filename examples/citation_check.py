"""Example plugin: a deterministic (no-model) evidence stage.

Flags answers that state facts without citing anything, and feeds that into
the confidence score. Load with:  ai-auditor --plugin examples/citation_check.py "..."
"""

import re

from pydantic import BaseModel

from ai_auditor import Auditor, stage

URL = re.compile(r"https?://\S+")


class CitationReport(BaseModel):
    urls: list[str]
    note: str


@stage(
    requires=("solution",), evidence=True, title="CITATIONS", order=45,
    render=lambda r: "\n".join(f"  - {u}" for u in r.urls) or f"  {r.note}",
    summarize=lambda r: r.note,
)
def citations(ctx) -> CitationReport:
    """Count the URLs the solution cites."""
    urls = URL.findall(ctx.solution)
    note = f"The solution cites {len(urls)} source(s)." if urls else "The solution cites no sources."
    return CitationReport(urls=urls, note=note)


def register(auditor: Auditor) -> None:
    auditor.add(citations)
