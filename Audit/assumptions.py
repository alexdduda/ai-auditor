from typing import List

from pydantic import BaseModel

from .client import MODEL, check_response, get_client


class AssumptionList(BaseModel):
    assumptions: List[str]


def extract_assumptions(problem: str, solution: str) -> List[str]:
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
                    "Identify the implicit and explicit assumptions underlying this "
                    "solution. List only assumptions that, if false, would change or "
                    "invalidate the solution.\n\n"
                    f"Problem:\n{problem}\n\nSolution:\n{solution}"
                ),
            }
        ],
        output_format=AssumptionList,
    )
    check_response(response)
    return response.parsed_output.assumptions
