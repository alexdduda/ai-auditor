from typing import List

from pydantic import BaseModel

from .client import MODEL, check_response, extract_text, get_client


class AlternativeList(BaseModel):
    alternatives: List[str]


def generate_alternatives(problem: str, solution: str) -> List[str]:
    client = get_client()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Propose 2-4 genuinely different alternative solutions or "
                    "answers to this problem, distinct from the one given. Favor "
                    "alternatives that would change the outcome if correct, not "
                    "minor variations.\n\n"
                    f"Problem:\n{problem}\n\nGiven solution:\n{solution}"
                ),
            }
        ],
        output_format=AlternativeList,
    )
    check_response(response)
    return response.parsed_output.alternatives


def gather_evidence(problem: str, solution: str) -> str:
    """Search the web for evidence bearing on the solution's correctness."""
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": (
                    "Search for information that would confirm or refute this "
                    "solution. Summarize what you find, citing sources, and note "
                    "whether it supports or contradicts the solution. If the "
                    "problem doesn't call for external facts (e.g. it's a "
                    "self-contained logic or math problem), say so and don't "
                    "search.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}"
                ),
            }
        ],
    )
    return extract_text(response)
