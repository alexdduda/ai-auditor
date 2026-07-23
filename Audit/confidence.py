from typing import List

from pydantic import BaseModel

from .client import MODEL, check_response, extract_text, get_client
from .contradiction import Contradiction, format_contradictions
from .verifier import AssumptionVerification, ExternalTestResult, format_verifications


class ConfidenceResult(BaseModel):
    score: float  # 0.0 (almost certainly wrong) to 1.0 (very high confidence)
    reasoning: str


def compute_confidence(
    problem: str,
    solution: str,
    verifications: List[AssumptionVerification],
    external_test: ExternalTestResult,
    contradictions: List[Contradiction],
) -> ConfidenceResult:
    client = get_client()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1500,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Based on the audit below, assign a confidence score from 0.0 "
                    "(almost certainly wrong) to 1.0 (very high confidence) that "
                    "the solution is correct. Weigh invalid/questionable "
                    "assumptions, failed external tests, and contradictions "
                    "against it.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}\n\n"
                    f"Assumption verifications:\n{format_verifications(verifications)}\n\n"
                    f"External test result: {external_test.summary}\n\n"
                    f"Contradictions found:\n{format_contradictions(contradictions)}"
                ),
            }
        ],
        output_format=ConfidenceResult,
    )
    check_response(response)
    return response.parsed_output


def explain_uncertainty(
    problem: str,
    solution: str,
    confidence: ConfidenceResult,
    contradictions: List[Contradiction],
) -> str:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": (
                    "In a short paragraph, explain what remains uncertain about "
                    "this solution and what would increase or decrease confidence "
                    "in it further. Be concrete about what's unresolved, not "
                    "generic hedging.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}\n\n"
                    f"Confidence score: {confidence.score}\n"
                    f"Confidence reasoning: {confidence.reasoning}\n\n"
                    f"Contradictions found:\n{format_contradictions(contradictions)}"
                ),
            }
        ],
    )
    return extract_text(response)
