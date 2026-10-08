"""Backward-compatible entry point. Prefer the installed `ai-auditor` command."""

from ai_auditor.cli import main

if __name__ == "__main__":
    main()
