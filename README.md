# ai-auditor

A general-purpose AI auditing pipeline: generate a solution to a problem, then
put it through a structured audit — extract its assumptions, verify them,
generate alternatives, search for counter-evidence, run external checks where
possible, and report a confidence score with an explanation of remaining
uncertainty.

```
Problem
  → Generate Solution
  → Extract Assumptions ─────────┐
  → Verify Assumptions           │
  → Run External Tests (if possible)
  → Generate Alternatives ───────┤
  → Search for Counter-Evidence  │
  → Compute Confidence
  → Explain Remaining Uncertainty
  → Final Answer (report)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Print a text report to stdout
.venv/bin/python main.py "Is it true that a group of crows is called a murder?"

# Read the problem from stdin instead
echo "Some problem statement" | .venv/bin/python main.py

# Write the report to a file instead of stdout
.venv/bin/python main.py "..." -o report.txt

# Get the full structured result as JSON
.venv/bin/python main.py "..." --json -o result.json
```

## Project layout

```
Audit/
├── client.py         # shared Anthropic client, model constant, response/error checks
├── pipeline.py        # orchestrator: runs every stage and assembles AuditResult
├── assumptions.py      # Extract Assumptions
├── verifier.py         # Verify Assumptions + Run External Tests (code execution tool)
├── evidence.py         # Generate Alternatives + Search for evidence (web search tool)
├── contradiction.py    # Search for Counter-Evidence (contradictions vs. alternatives/evidence)
├── confidence.py       # Compute Confidence + Explain Remaining Uncertainty
└── report.py           # Final Answer — assembles everything into a text report
main.py                 # CLI entry point
```

Every stage calls Claude directly via the `anthropic` SDK (model: `claude-opus-4-8`,
adaptive thinking). Stages that depend only on the initial `(problem, solution)`
pair run concurrently to cut wall-clock latency.

## Notes

- `run_external_tests` (in `verifier.py`) only executes code when the problem is
  actually checkable that way (math, algorithms, proofs) — otherwise it reports
  "not applicable."
- A Claude refusal or a truncated (`max_tokens`) response at any stage raises an
  `AuditGenerationError` instead of silently continuing with empty output.
