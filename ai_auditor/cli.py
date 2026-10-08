from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

from .core import AuditGraphError, Auditor, StageError, StageEvent
from .llm import DEFAULT_MODEL, AnthropicLLM, AuditGenerationError
from .plugins import PluginError, load_entry_point_plugins, load_plugin


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-auditor",
        description=(
            "Audit an AI-generated solution: generate a solution (or take yours), "
            "extract and verify its assumptions, search for counter-evidence, and "
            "report a confidence score with remaining uncertainty."
        ),
    )
    parser.add_argument("problem", nargs="?", help="The problem to solve and audit. Reads from stdin if omitted.")
    parser.add_argument("-o", "--output", metavar="FILE", help="Write the result to FILE instead of stdout.")
    parser.add_argument("--json", action="store_true", help="Output the full audit result as JSON instead of the text report.")
    parser.add_argument("--solution", metavar="TEXT|@FILE", help="Audit this solution instead of generating one (@path reads a file).")
    parser.add_argument("--plugin", action="append", default=[], metavar="MODULE|FILE.py", help="Load a plugin (repeatable).")
    parser.add_argument("--entry-point-plugins", action="store_true", help="Also load installed plugins from the ai_auditor.plugins entry-point group.")
    parser.add_argument("--skip", action="append", default=[], metavar="STAGE", help="Remove a stage and everything that depends on it (repeatable).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model for every stage (default: {DEFAULT_MODEL}).")
    parser.add_argument("--continue-on-error", action="store_true", help="Record stage failures and keep going instead of aborting.")
    parser.add_argument("--list-stages", action="store_true", help="Print the stage graph (after plugins/--skip) and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log stage progress to stderr.")
    return parser.parse_args(argv)


def _has_credentials() -> bool:
    """Cheap pre-flight so a missing key fails before any stage runs (and costs anything).

    Mirrors the SDK's sources: API key / auth token env vars, federation env vars,
    or a profile written by `ant auth login`.
    """
    env = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID")
    if any(os.environ.get(k) for k in env):
        return True
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return (config_home / "anthropic").is_dir()


def build_auditor(args: argparse.Namespace) -> Auditor:
    def log(e: StageEvent) -> None:
        extra = f" ({e.seconds:.1f}s)" if e.status in ("finished", "failed") else ""
        print(f"[{e.status}] {e.stage}{extra}{': ' + e.error if e.error else ''}", file=sys.stderr)

    auditor = Auditor.default(
        llm=AnthropicLLM(model=args.model), fail_fast=not args.continue_on_error,
        on_event=log if args.verbose else None,
    )
    if args.entry_point_plugins:
        load_entry_point_plugins(auditor)
    for spec in args.plugin:
        load_plugin(auditor, spec)
    for name in args.skip:
        auditor.remove(name, cascade=True)
    return auditor


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        auditor = build_auditor(args)
        waves = auditor.plan()
    except (PluginError, AuditGraphError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.list_stages:
        stages = auditor.stages
        for i, wave in enumerate(waves, 1):
            for name in wave:
                s = stages[name]
                flags = " [evidence]" if s.evidence else ""
                print(f"wave {i}: {name}{flags}: requires {', '.join(s.requires) or 'nothing'}. {s.description}")
        return

    if not _has_credentials():
        print(
            "No Anthropic API credentials found. Set ANTHROPIC_API_KEY (e.g. `export "
            "ANTHROPIC_API_KEY=sk-ant-...`, key from console.anthropic.com), or sign in with "
            "the `ant` CLI (`ant auth login`). Nothing was sent.",
            file=sys.stderr,
        )
        sys.exit(1)

    problem = args.problem
    if not problem:
        print("Enter the problem to audit (Ctrl+D to finish):", file=sys.stderr)
        problem = sys.stdin.read().strip()
    if not problem:
        print("No problem provided.", file=sys.stderr)
        sys.exit(1)

    solution = args.solution
    if solution and solution.startswith("@"):
        with open(solution[1:]) as f:
            solution = f.read()

    try:
        result = auditor.run(problem, solution=solution)
    except StageError as e:
        cause = e.error
        if isinstance(cause, anthropic.AuthenticationError):
            print("Invalid or missing ANTHROPIC_API_KEY.", file=sys.stderr)
        elif isinstance(cause, anthropic.RateLimitError):
            retry_after = cause.response.headers.get("retry-after", "a bit")
            print(f"Rate limited by the Anthropic API. Retry after {retry_after}s.", file=sys.stderr)
        elif isinstance(cause, anthropic.APIConnectionError):
            print("Network error reaching the Anthropic API. Check your connection and retry.", file=sys.stderr)
        elif isinstance(cause, anthropic.APIStatusError):
            print(f"Anthropic API error ({cause.status_code}) in stage '{e.stage}': {cause.message}", file=sys.stderr)
        else:
            print(f"Audit failed in stage '{e.stage}': {cause}", file=sys.stderr)
        sys.exit(1)
    except AuditGenerationError as e:  # raised by the solver, before any stage runs
        print(f"Audit failed: {e}", file=sys.stderr)
        sys.exit(1)
    except anthropic.AuthenticationError:
        print("Invalid or missing ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"Anthropic API error while generating the solution: {e}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result.to_dict(), indent=2) if args.json else result.report
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output)
    if result.errors:
        sys.exit(3)


if __name__ == "__main__":
    main()
