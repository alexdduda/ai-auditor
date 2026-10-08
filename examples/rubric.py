"""Example plugin: a model-graded rubric stage plus a stage replacement.

- Adds `rubric`, which grades the solution against your own criteria with
  ctx.llm (so it uses whatever model the auditor was configured with).
- Turns off web search by replacing the built-in `evidence` stage.
"""

from pydantic import BaseModel

from ai_auditor import Auditor, stage

CRITERIA = [
    "States the final answer explicitly.",
    "Shows the reasoning or derivation, not just the result.",
    "Mentions at least one limitation or edge case.",
]


class Criterion(BaseModel):
    criterion: str
    met: bool
    why: str


class Rubric(BaseModel):
    results: list[Criterion]


@stage(
    requires=("solution",), evidence=True, title="RUBRIC", order=35,
    render=lambda r: "\n".join(f"  [{'MET' if c.met else 'NOT MET'}] {c.criterion} — {c.why}" for c in r.results),
    summarize=lambda r: f"{sum(c.met for c in r.results)}/{len(r.results)} rubric criteria met",
)
def rubric(ctx) -> Rubric:
    """Grade the solution against a fixed rubric."""
    listed = "\n".join(f"- {c}" for c in CRITERIA)
    return ctx.llm.parse(
        f"Grade the solution against each criterion.\n\nCriteria:\n{listed}\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}",
        Rubric, effort="low", max_tokens=2000,
    )


@stage(name="evidence", requires=("solution",), title="EVIDENCE GATHERED", order=40)
def no_web_evidence(ctx) -> str:
    """Offline replacement for the web-search evidence stage."""
    return "(web search disabled by the rubric plugin)"


def register(auditor: Auditor) -> None:
    auditor.add(rubric)
    auditor.replace(no_web_evidence)
