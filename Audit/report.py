from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AuditResult


def build_report(result: "AuditResult") -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("AI AUDIT REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("PROBLEM")
    lines.append(result.problem)
    lines.append("")

    lines.append("SOLUTION")
    lines.append(result.solution)
    lines.append("")

    lines.append(f"CONFIDENCE: {result.confidence.score:.2f}")
    lines.append(result.confidence.reasoning)
    lines.append("")

    lines.append("ASSUMPTIONS")
    if result.verifications:
        for v in result.verifications:
            lines.append(f"  [{v.verdict.upper()}] {v.assumption}")
            lines.append(f"    {v.explanation}")
    else:
        lines.append("  (none extracted)")
    lines.append("")

    lines.append("ALTERNATIVES CONSIDERED")
    if result.alternatives:
        for a in result.alternatives:
            lines.append(f"  - {a}")
    else:
        lines.append("  (none generated)")
    lines.append("")

    lines.append("EVIDENCE GATHERED")
    lines.append(result.evidence or "  (none gathered)")
    lines.append("")

    lines.append("CONTRADICTIONS / COUNTER-EVIDENCE")
    if result.contradictions:
        for c in result.contradictions:
            lines.append(f"  [{c.severity.upper()}] {c.description}")
    else:
        lines.append("  (none found)")
    lines.append("")

    lines.append("EXTERNAL TESTS")
    if result.external_test.ran:
        if result.external_test.passed is True:
            status = "PASSED"
        elif result.external_test.passed is False:
            status = "FAILED"
        else:
            status = "INCONCLUSIVE"
        lines.append(f"  [{status}] {result.external_test.summary}")
    else:
        lines.append(f"  Not applicable: {result.external_test.summary}")
    lines.append("")

    lines.append("REMAINING UNCERTAINTY")
    lines.append(result.uncertainty)
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
