import re
from typing import List, Literal, Optional

from pydantic import BaseModel

from .client import MODEL, check_response, extract_text, get_client

_RESULT_LINE = re.compile(r"RESULT:\s*(PASS|FAIL|N/A)", re.IGNORECASE)


class AssumptionVerification(BaseModel):
    assumption: str
    verdict: Literal["valid", "questionable", "invalid"]
    explanation: str


class VerificationList(BaseModel):
    verifications: List[AssumptionVerification]


def verify_assumptions(
    problem: str, solution: str, assumptions: List[str]
) -> List[AssumptionVerification]:
    if not assumptions:
        return []

    client = get_client()
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(assumptions))
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "user",
                "content": (
                    "For each assumption below, judge whether it holds given the "
                    "problem and solution. Use 'invalid' if the assumption is false, "
                    "'questionable' if it's uncertain or only partially true, 'valid' "
                    "if it clearly holds.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}\n\n"
                    f"Assumptions:\n{numbered}"
                ),
            }
        ],
        output_format=VerificationList,
    )
    check_response(response)
    return response.parsed_output.verifications


def format_verifications(verifications: List[AssumptionVerification]) -> str:
    if not verifications:
        return "(no assumptions extracted)"
    return "\n".join(
        f"- [{v.verdict}] {v.assumption} — {v.explanation}" for v in verifications
    )


class ExternalTestResult(BaseModel):
    ran: bool
    summary: str
    passed: Optional[bool] = None


def run_external_tests(problem: str, solution: str) -> ExternalTestResult:
    """Best-effort: if the solution is checkable by code (math, logic, small
    algorithms), execute it via the code execution tool and report the outcome.
    """
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
        messages=[
            {
                "role": "user",
                "content": (
                    "If this problem and solution can be checked by writing and "
                    "running code (e.g. verifying a calculation, testing an "
                    "algorithm, checking a proof step numerically), do so now and "
                    "report the result. If it cannot be checked this way, say so "
                    "plainly and do not run any code.\n\n"
                    "End your response with exactly one of these lines, verbatim: "
                    "'RESULT: PASS' if the code confirms the solution, "
                    "'RESULT: FAIL' if it contradicts the solution, or "
                    "'RESULT: N/A' if no code was run.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}"
                ),
            }
        ],
    )

    ran = any(
        block.type == "bash_code_execution_tool_result" for block in response.content
    )
    text = extract_text(response)

    passed = None
    match = _RESULT_LINE.search(text)
    if match:
        verdict = match.group(1).upper()
        passed = {"PASS": True, "FAIL": False, "N/A": None}[verdict]
        text = text[: match.start()].rstrip()

    return ExternalTestResult(ran=ran, summary=text, passed=passed)
