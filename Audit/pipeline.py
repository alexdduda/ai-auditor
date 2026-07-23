from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from .assumptions import extract_assumptions
from .client import MODEL, extract_text, get_client
from .confidence import ConfidenceResult, compute_confidence, explain_uncertainty
from .contradiction import Contradiction, find_contradictions
from .evidence import gather_evidence, generate_alternatives
from .report import build_report
from .verifier import (
    AssumptionVerification,
    ExternalTestResult,
    run_external_tests,
    verify_assumptions,
)


@dataclass
class AuditResult:
    problem: str
    solution: str
    verifications: List[AssumptionVerification]
    alternatives: List[str]
    evidence: str
    contradictions: List[Contradiction]
    external_test: ExternalTestResult
    confidence: ConfidenceResult
    uncertainty: str
    report: str = ""

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "solution": self.solution,
            "confidence": self.confidence.model_dump(),
            "assumptions": [v.model_dump() for v in self.verifications],
            "alternatives": self.alternatives,
            "evidence": self.evidence,
            "contradictions": [c.model_dump() for c in self.contradictions],
            "external_test": self.external_test.model_dump(),
            "uncertainty": self.uncertainty,
        }


def generate_solution(problem: str) -> str:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": problem}],
    )
    return extract_text(response)


def run_audit(problem: str) -> AuditResult:
    solution = generate_solution(problem)

    # Stages that depend only on (problem, solution) run concurrently —
    # each is a blocking network call, so this cuts wall-clock time roughly
    # 4x over running them one after another.
    with ThreadPoolExecutor(max_workers=4) as pool:
        assumptions_future = pool.submit(extract_assumptions, problem, solution)
        external_test_future = pool.submit(run_external_tests, problem, solution)
        alternatives_future = pool.submit(generate_alternatives, problem, solution)
        evidence_future = pool.submit(gather_evidence, problem, solution)

        assumptions = assumptions_future.result()
        external_test = external_test_future.result()
        alternatives = alternatives_future.result()
        evidence = evidence_future.result()

    # Verifying assumptions and finding contradictions are independent of
    # each other, so run them side by side too.
    with ThreadPoolExecutor(max_workers=2) as pool:
        verifications_future = pool.submit(
            verify_assumptions, problem, solution, assumptions
        )
        contradictions_future = pool.submit(
            find_contradictions, problem, solution, alternatives, evidence
        )

        verifications = verifications_future.result()
        contradictions = contradictions_future.result()

    confidence = compute_confidence(
        problem, solution, verifications, external_test, contradictions
    )
    uncertainty = explain_uncertainty(problem, solution, confidence, contradictions)

    result = AuditResult(
        problem=problem,
        solution=solution,
        verifications=verifications,
        alternatives=alternatives,
        evidence=evidence,
        contradictions=contradictions,
        external_test=external_test,
        confidence=confidence,
        uncertainty=uncertainty,
    )
    result.report = build_report(result)
    return result
