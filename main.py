import argparse
import json
import sys

import anthropic

from Audit.client import AuditGenerationError
from Audit.pipeline import run_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-auditor",
        description=(
            "Audit an AI-generated solution: generate a solution, extract and "
            "verify its assumptions, search for counter-evidence, and report a "
            "confidence score with remaining uncertainty."
        ),
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="The problem to solve and audit. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write the result to FILE instead of stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full audit result as JSON instead of the text report.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    problem = args.problem
    if not problem:
        print("Enter the problem to audit (Ctrl+D to finish):", file=sys.stderr)
        problem = sys.stdin.read().strip()

    if not problem:
        print("No problem provided.", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_audit(problem)
    except AuditGenerationError as e:
        print(f"Audit failed: {e}", file=sys.stderr)
        sys.exit(1)
    except anthropic.AuthenticationError:
        print("Invalid or missing ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after", "a bit")
        print(
            f"Rate limited by the Anthropic API. Retry after {retry_after}s.",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.APIConnectionError:
        print(
            "Network error reaching the Anthropic API. Check your connection "
            "and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"Anthropic API error ({e.status_code}): {e.message}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result.to_dict(), indent=2) if args.json else result.report

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
