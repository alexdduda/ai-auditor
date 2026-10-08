"""Text report: confidence first, then every titled stage in `order`.

Plugin stages with a `title` appear automatically; errors and skipped stages
are listed instead of silently disappearing.
"""

from __future__ import annotations

from .core import AuditResult


def build_report(result: AuditResult) -> str:
    rule = "=" * 70
    lines = [rule, "AI AUDIT REPORT", rule, "", "PROBLEM", result.problem, "", "SOLUTION", result.solution, ""]

    conf = result.outputs.get("confidence")
    if conf is not None:
        lines += [f"CONFIDENCE: {conf.score:.2f}", conf.reasoning, ""]

    for s in sorted(result.stages.values(), key=lambda s: (s.order, s.name)):
        if not s.title:
            continue
        if s.name in result.outputs:
            lines += [s.title, s.render_output(result.outputs[s.name]), ""]
        elif s.name in result.errors:
            lines += [s.title, f"  (stage failed: {result.errors[s.name]})", ""]
        elif s.name in result.skipped:
            lines += [s.title, f"  (skipped: {result.skipped[s.name]})", ""]

    if result.errors and "confidence" not in result.outputs:
        lines += ["NOTE", "  Confidence could not be computed because a required stage failed.", ""]
    lines.append(rule)
    return "\n".join(lines)
