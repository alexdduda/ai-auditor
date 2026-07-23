from typing import List, Literal

from pydantic import BaseModel

from .client import MODEL, check_response, get_client


class Contradiction(BaseModel):
    description: str
    severity: Literal["low", "medium", "high"]


class ContradictionList(BaseModel):
    contradictions: List[Contradiction]


def find_contradictions(
    problem: str,
    solution: str,
    alternatives: List[str],
    evidence: str,
) -> List[Contradiction]:
    client = get_client()
    alt_text = "\n".join(f"- {a}" for a in alternatives) or "(none proposed)"
    response = client.messages.parse(
        model=MODEL,
        max_tokens=3000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Identify any contradictions or counter-evidence against the "
                    "given solution, drawing on the alternative solutions and the "
                    "gathered evidence below. Only report genuine conflicts, not "
                    "stylistic differences. If nothing contradicts the solution, "
                    "return an empty list.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}\n\n"
                    f"Alternative solutions considered:\n{alt_text}\n\n"
                    f"Gathered evidence:\n{evidence or '(none gathered)'}"
                ),
            }
        ],
        output_format=ContradictionList,
    )
    check_response(response)
    return response.parsed_output.contradictions


def format_contradictions(contradictions: List[Contradiction]) -> str:
    if not contradictions:
        return "(none found)"
    return "\n".join(f"- [{c.severity}] {c.description}" for c in contradictions)
