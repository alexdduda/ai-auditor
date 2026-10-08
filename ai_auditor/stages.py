"""The built-in audit, expressed as ordinary stages.

These are plugins like any other: replace, remove or reconfigure them on an
Auditor. Prompts are unchanged from the original pipeline.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel

from .core import EVIDENCE, AuditContext, Stage, stage

_RESULT_LINE = re.compile(r"RESULT:\s*(PASS|FAIL|N/A)", re.IGNORECASE)


# ── models ──────────────────────────────────────────────────────────────────
class AssumptionList(BaseModel):
    assumptions: List[str]


class AssumptionVerification(BaseModel):
    assumption: str
    verdict: Literal["valid", "questionable", "invalid"]
    explanation: str


class VerificationList(BaseModel):
    verifications: List[AssumptionVerification]


class ExternalTestResult(BaseModel):
    ran: bool
    summary: str
    passed: Optional[bool] = None


class AlternativeList(BaseModel):
    alternatives: List[str]


class Contradiction(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]


class ContradictionList(BaseModel):
    contradictions: List[Contradiction]


class ConfidenceResult(BaseModel):
    score: float  # 0.0 (almost certainly wrong) to 1.0 (very high confidence)
    reasoning: str


# ── formatting ──────────────────────────────────────────────────────────────
def format_verifications(verifications: List[AssumptionVerification]) -> str:
    if not verifications:
        return "(no assumptions extracted)"
    return "\n".join(f"- [{v.verdict}] {v.assumption} — {v.explanation}" for v in verifications)


def format_contradictions(contradictions: List[Contradiction]) -> str:
    if not contradictions:
        return "(none found)"
    return "\n".join(f"- [{c.severity}] {c.description}" for c in contradictions)


def _render_verifications(vs: List[AssumptionVerification]) -> str:
    if not vs:
        return "  (none extracted)"
    return "\n".join(f"  [{v.verdict.upper()}] {v.assumption}\n    {v.explanation}" for v in vs)


def _render_list(empty: str):
    return lambda items: "\n".join(f"  - {a}" for a in items) if items else f"  {empty}"


def _render_contradictions(cs: List[Contradiction]) -> str:
    if not cs:
        return "  (none found)"
    return "\n".join(f"  [{c.severity.upper()}] {c.description}" for c in cs)


def _render_external(t: ExternalTestResult) -> str:
    if not t.ran:
        return f"  Not applicable: {t.summary}"
    status = {True: "PASSED", False: "FAILED", None: "INCONCLUSIVE"}[t.passed]
    return f"  [{status}] {t.summary}"


# ── solver ──────────────────────────────────────────────────────────────────
def generate_solution(ctx: AuditContext) -> str:
    """Default solver: ask the model to answer the problem."""
    return ctx.llm.complete(ctx.problem, effort="high", max_tokens=4000).text


# ── stages ──────────────────────────────────────────────────────────────────
@stage(requires=("solution",))
def assumptions(ctx: AuditContext) -> List[str]:
    """Extract the assumptions the solution depends on."""
    return ctx.llm.parse(
        "Identify the implicit and explicit assumptions underlying this "
        "solution. List only assumptions that, if false, would change or "
        "invalidate the solution.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}",
        AssumptionList, effort="medium", max_tokens=2000,
    ).assumptions


@stage(requires=("assumptions",), evidence=True, title="ASSUMPTIONS", order=20,
       render=_render_verifications, summarize=format_verifications)
def verifications(ctx: AuditContext) -> List[AssumptionVerification]:
    """Judge whether each assumption holds."""
    items = ctx["assumptions"]
    if not items:
        return []
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(items))
    return ctx.llm.parse(
        "For each assumption below, judge whether it holds given the "
        "problem and solution. Use 'invalid' if the assumption is false, "
        "'questionable' if it's uncertain or only partially true, 'valid' "
        "if it clearly holds.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}\n\n"
        f"Assumptions:\n{numbered}",
        VerificationList, effort="high", max_tokens=4000,
    ).verifications


@stage(requires=("solution",), evidence=True, title="EXTERNAL TESTS", order=60,
       render=_render_external, summarize=lambda t: t.summary)
def external_test(ctx: AuditContext) -> ExternalTestResult:
    """If the solution is checkable by code, run code against it (server-side sandbox)."""
    completion = ctx.llm.complete(
        "If this problem and solution can be checked by writing and "
        "running code (e.g. verifying a calculation, testing an "
        "algorithm, checking a proof step numerically), do so now and "
        "report the result. If it cannot be checked this way, say so "
        "plainly and do not run any code.\n\n"
        "End your response with exactly one of these lines, verbatim: "
        "'RESULT: PASS' if the code confirms the solution, "
        "'RESULT: FAIL' if it contradicts the solution, or "
        "'RESULT: N/A' if no code was run.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}",
        effort="medium", max_tokens=4000,
        tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
    )
    ran = "bash_code_execution_tool_result" in completion.block_types
    text = completion.text
    passed = None
    match = _RESULT_LINE.search(text)
    if match:
        passed = {"PASS": True, "FAIL": False, "N/A": None}[match.group(1).upper()]
        text = text[: match.start()].rstrip()
    return ExternalTestResult(ran=ran, summary=text, passed=passed)


@stage(requires=("solution",), title="ALTERNATIVES CONSIDERED", order=30, render=_render_list("(none generated)"))
def alternatives(ctx: AuditContext) -> List[str]:
    """Propose materially different alternative answers."""
    return ctx.llm.parse(
        "Propose 2-4 genuinely different alternative solutions or "
        "answers to this problem, distinct from the one given. Favor "
        "alternatives that would change the outcome if correct, not "
        "minor variations.\n\n"
        f"Problem:\n{ctx.problem}\n\nGiven solution:\n{ctx.solution}",
        AlternativeList, effort="medium", max_tokens=2000,
    ).alternatives


@stage(requires=("solution",), title="EVIDENCE GATHERED", order=40, render=lambda e: e or "  (none gathered)")
def evidence(ctx: AuditContext) -> str:
    """Search the web for evidence bearing on the solution's correctness."""
    return ctx.llm.complete(
        "Search for information that would confirm or refute this "
        "solution. Summarize what you find, citing sources, and note "
        "whether it supports or contradicts the solution. If the "
        "problem doesn't call for external facts (e.g. it's a "
        "self-contained logic or math problem), say so and don't "
        "search.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}",
        effort="medium", max_tokens=4000,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
    ).text


@stage(requires=("alternatives", "evidence"), evidence=True, title="CONTRADICTIONS / COUNTER-EVIDENCE", order=50,
       render=_render_contradictions, summarize=format_contradictions)
def contradictions(ctx: AuditContext) -> List[Contradiction]:
    """Find genuine conflicts between the solution and the alternatives/evidence."""
    alt_text = "\n".join(f"- {a}" for a in ctx["alternatives"]) or "(none proposed)"
    return ctx.llm.parse(
        "Identify any contradictions or counter-evidence against the "
        "given solution, drawing on the alternative solutions and the "
        "gathered evidence below. Only report genuine conflicts, not "
        "stylistic differences. If nothing contradicts the solution, "
        "return an empty list.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}\n\n"
        f"Alternative solutions considered:\n{alt_text}\n\n"
        f"Gathered evidence:\n{ctx['evidence'] or '(none gathered)'}",
        ContradictionList, effort="high", max_tokens=3000,
    ).contradictions


@stage(requires=(EVIDENCE,), order=10)
def confidence(ctx: AuditContext) -> ConfidenceResult:
    """Score 0–1 from every evidence stage (built-in and plugin)."""
    sections = "\n\n".join(f"{s.title or s.name}:\n{s.summarize_output(out)}" for s, out in ctx.evidence())
    return ctx.llm.parse(
        "Based on the audit below, assign a confidence score from 0.0 "
        "(almost certainly wrong) to 1.0 (very high confidence) that "
        "the solution is correct. Weigh invalid/questionable "
        "assumptions, failed external tests, and contradictions "
        "against it.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}\n\n"
        f"{sections or '(no evidence stages ran)'}",
        ConfidenceResult, effort="medium", max_tokens=1500,
    )


@stage(requires=("confidence", "contradictions"), title="REMAINING UNCERTAINTY", order=90)
def uncertainty(ctx: AuditContext) -> str:
    """Explain concretely what remains unresolved."""
    conf = ctx["confidence"]
    return ctx.llm.complete(
        "In a short paragraph, explain what remains uncertain about "
        "this solution and what would increase or decrease confidence "
        "in it further. Be concrete about what's unresolved, not "
        "generic hedging.\n\n"
        f"Problem:\n{ctx.problem}\n\nSolution:\n{ctx.solution}\n\n"
        f"Confidence score: {conf.score}\n"
        f"Confidence reasoning: {conf.reasoning}\n\n"
        f"Contradictions found:\n{format_contradictions(ctx['contradictions'])}",
        effort="medium", max_tokens=1500,
    ).text


DEFAULT_STAGES: tuple[Stage, ...] = (
    assumptions, verifications, external_test, alternatives, evidence, contradictions, confidence, uncertainty,
)
