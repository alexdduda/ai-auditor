# ai-auditor

A framework for auditing AI-generated answers. An audit is a graph of **stages**: you write a stage as a plain function, declare which other stages' outputs it needs, and the framework does the rest:
- runs every stage as soon as its inputs are ready, with independent stages in parallel
- feeds evidence into the confidence score
- handles failures
- assembles the report

The built-in audit is itself just a set of stages: generate a solution → extract and verify assumptions → run code checks → propose alternatives → search for counter-evidence → score confidence → explain remaining uncertainty. You can add to it, replace parts, or drop parts without touching the framework.

```
wave 1  assumptions   external_test*   alternatives   evidence      (+ your stages)
wave 2  verifications*                 contradictions*
wave 3  confidence   ← every stage marked evidence (*), including plugins
wave 4  uncertainty
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Command line

```bash
ai-auditor "Is it true that a group of crows is called a murder?"     # generate + audit
ai-auditor "What is 17 * 23?" --solution "391"                        # audit YOUR answer
ai-auditor "..." --solution @answer.md --json -o result.json
ai-auditor "..." --plugin examples/citation_check.py                  # add a stage
ai-auditor "..." --skip evidence                                      # drop web search (and what depends on it)
ai-auditor --list-stages --plugin examples/rubric.py                  # show the graph, no API calls
ai-auditor "..." --model claude-opus-5-5 --continue-on-error -v
```

`python main.py ...` still works.

Exit codes:
- `0`: ok
- `1`: the audit failed
- `2`: bad configuration (unknown plugin or stage, cycle)
- `3`: completed with failed stages (only with `--continue-on-error`)

## Writing a stage

```python
from ai_auditor import Auditor, stage

@stage(requires=("solution",), evidence=True, title="WORD COUNT",
       summarize=lambda n: f"the answer is {n} words long")
def word_count(ctx):
    """Flag padded answers."""
    return len(ctx.solution.split())

auditor = Auditor.default().add(word_count)
result = auditor.run("Explain TCP slow start in under 100 words.")
print(result.report)          # includes a WORD COUNT section
result["word_count"]          # the raw output
```

A stage receives a context (`ctx`):

| | |
|---|---|
| `ctx.problem`, `ctx.solution` | the inputs |
| `ctx["other_stage"]` | an upstream output; only stages listed in `requires` are readable |
| `ctx.llm.complete(prompt, effort=, max_tokens=, tools=)` | text completion → `Completion(text, block_types)` |
| `ctx.llm.parse(prompt, PydanticModel, effort=, max_tokens=)` | structured output → model instance |
| `ctx.evidence()` | `(stage, output)` pairs for evidence stages that are readable here |

`@stage` options:
- **`requires`:** stage names, or `"problem"`/`"solution"`. `EVIDENCE` means every evidence stage.
- **`evidence=True`:** feeds `summarize(output)` into the confidence prompt.
- **`title`, `render`, `order`:** control the stage's report section. With no `title`, the stage is hidden from the report.
- **`name`:** defaults to the function name. Reuse an existing name with `replace()` to override a built-in stage.

## Composing audits

```python
auditor = Auditor.default(fail_fast=False, on_event=print)
auditor.add(my_stage)                              # new stage
auditor.replace(my_offline_evidence_stage)         # same name as a built-in
auditor.remove("evidence", cascade=True)           # also removes contradictions, uncertainty
auditor.configure("external_test", evidence=False) # stop it affecting confidence
auditor.plan()                                     # validate; returns the waves
```

Behavior details:
- **Removing stages:** removing a stage that other stages require raises an error unless you pass `cascade=True`. Stages that depend on `EVIDENCE`, such as `confidence`, adapt to whichever evidence stages remain.
- **Errors (`fail_fast=False`):** a failing stage is recorded in `result.errors`, every stage downstream of it is listed in `result.skipped`, and the report shows both. `fail_fast=True` (the default) raises `StageError`.
- **Building from scratch:** `Auditor([...])` with your own stages, `llm` and `solver` gives a non-default audit. `Auditor.default()` is just the built-in stages plus `AnthropicLLM`, the default solver and the text reporter.

## Plugins

A plugin is a module or `.py` file with a `register(auditor)` function. See [`examples/`](examples/):
- [`citation_check.py`](examples/citation_check.py): a deterministic (no-model) evidence stage.
- [`rubric.py`](examples/rubric.py): a model-graded rubric stage, plus a replacement for the web-search stage.

Installed packages can also expose plugins under the `ai_auditor.plugins` entry-point group:

```toml
[project.entry-points."ai_auditor.plugins"]
my_checks = "my_package.audit:register"
```

Entry-point plugins load only with `--entry-point-plugins` or `load_entry_point_plugins(auditor)`, never implicitly.

## Models

Stages talk to `ctx.llm`, not the SDK. `AnthropicLLM(model=...)` is the default implementation: adaptive thinking, structured outputs via `messages.parse`, and server-side code-execution and web-search tools for the built-in stages. The default model is `claude-opus-4-8`; change it with `--model` or `AnthropicLLM(model=...)`.

**Output style.** Generated text never contains em dashes or emojis. `AnthropicLLM` asks for this in its system prompt, and `Auditor.default()` enforces it with an output filter (`ai_auditor.style.clean_value`) applied to every stage output and to a generated solution. A solution you pass with `--solution` is never altered, because it is what's being audited. Pass `output_filter=None` / `AnthropicLLM(system=None)` to turn this off.

Any object with `complete` and `parse` works. The tests use a fake one, so they make no API calls:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Layout

```
ai_auditor/
├── core.py      # Stage, @stage, AuditContext, Auditor (graph, scheduling, errors, events), AuditResult
├── llm.py       # LLM protocol, AnthropicLLM, refusal/truncation checks
├── stages.py    # the built-in audit as stages (prompts unchanged from the original pipeline)
├── report.py    # text report; plugin stages with a title appear automatically
├── plugins.py   # load_plugin, entry-point discovery
└── cli.py       # ai-auditor command
examples/        # example plugins
tests/           # framework tests (fake LLM, no network)
```

## Changes from 0.1 (pipeline)

- **Package rename:** the package moved from `Audit/` to `ai_auditor/`. `main.py` is kept as a shim.
- **JSON output:** `--json` now nests outputs under `"stages"`, keyed by stage name. `assumptions` is the raw list; the per-assumption verdicts moved to `verifications`. There are also new `errors`, `skipped` and `timings` fields.
- **Confidence prompt:** it now lists evidence under each stage's report title, e.g. "ASSUMPTIONS:" instead of "Assumption verifications:". This is what lets plugin evidence reach it. The instructions themselves are unchanged.
