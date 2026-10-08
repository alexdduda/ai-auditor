"""The framework core: stages, the audit context, and the Auditor that runs them.

An audit is a directed acyclic graph of *stages*. Each stage declares which
other stages' outputs it needs (`requires`); the Auditor resolves the graph,
runs every stage as soon as its inputs are ready (independent stages run
concurrently), and collects the outputs into an AuditResult.

You extend an audit by adding stages — no changes to the framework needed:

    from ai_auditor import Auditor, stage

    @stage(requires=("solution",), evidence=True, title="WORD COUNT")
    def word_count(ctx):
        return len(ctx.solution.split())

    auditor = Auditor.default().add(word_count)
    result = auditor.run("Explain TCP slow start in under 100 words.")

Stages marked `evidence=True` are automatically fed into the confidence
stage, so a custom checker changes the final score without touching it.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping

from pydantic import BaseModel

#: Names that are always available to stages without being produced by one.
BUILTIN_INPUTS = ("problem", "solution")

#: Placeholder in `requires` meaning "every stage marked evidence=True".
EVIDENCE = "*evidence"


class StageError(RuntimeError):
    """A stage raised. `stage` is its name; the original exception is chained."""

    def __init__(self, stage: str, error: BaseException):
        super().__init__(f"stage '{stage}' failed: {error}")
        self.stage = stage
        self.error = error


class AuditGraphError(ValueError):
    """The stage graph is invalid: unknown dependency, duplicate name or cycle."""


@dataclass(frozen=True)
class Stage:
    """One step of an audit.

    fn:        called with the AuditContext; returns this stage's output.
    requires:  names of stages (or "problem"/"solution") whose outputs fn reads.
               Use EVIDENCE to depend on every evidence stage.
    evidence:  feed this stage's `summarize(output)` into the confidence stage.
    title:     heading in the text report; None hides the stage from the report.
    render:    output -> report body (default: str(output)).
    summarize: output -> short text for the confidence prompt (default: render).
    order:     position in the report (lower first).
    """

    name: str
    fn: Callable[["AuditContext"], Any]
    requires: tuple[str, ...] = ()
    evidence: bool = False
    title: str | None = None
    render: Callable[[Any], str] | None = None
    summarize: Callable[[Any], str] | None = None
    order: int = 100
    description: str = ""

    def render_output(self, output: Any) -> str:
        return self.render(output) if self.render else _default_render(output)

    def summarize_output(self, output: Any) -> str:
        return self.summarize(output) if self.summarize else self.render_output(output)


def stage(
    name: str | None = None,
    *,
    requires: Iterable[str] = (),
    evidence: bool = False,
    title: str | None = None,
    render: Callable[[Any], str] | None = None,
    summarize: Callable[[Any], str] | None = None,
    order: int = 100,
) -> Callable[[Callable[["AuditContext"], Any]], Stage]:
    """Decorator turning `fn(ctx) -> output` into a Stage named after the function."""

    def wrap(fn: Callable[["AuditContext"], Any]) -> Stage:
        return Stage(
            name=name or fn.__name__,
            fn=fn,
            requires=tuple(requires),
            evidence=evidence,
            title=title,
            render=render,
            summarize=summarize,
            order=order,
            description=(fn.__doc__ or "").strip().split("\n")[0],
        )

    return wrap


class AuditContext:
    """What a stage sees: the problem, the solution, upstream outputs, the LLM."""

    def __init__(self, problem: str, solution: str | None, llm: Any, stages: Mapping[str, Stage],
                 outputs: dict[str, Any] | None = None, allowed: frozenset[str] | None = None):
        self.problem = problem
        self._solution = solution
        self.llm = llm
        self.stages = stages
        self.outputs: dict[str, Any] = {} if outputs is None else outputs
        #: Stage names this context may read; None = unrestricted (the Auditor's own view).
        self._allowed = allowed

    def scoped(self, allowed: Iterable[str]) -> "AuditContext":
        """A view that can only read the given upstream outputs, so a stage that
        forgets to declare a dependency fails every time instead of racing."""
        return AuditContext(self.problem, self._solution, self.llm, self.stages, self.outputs, frozenset(allowed))

    @property
    def solution(self) -> str:
        if self._solution is None:
            raise RuntimeError("solution is not available yet; declare requires=('solution',)")
        return self._solution

    def __getitem__(self, name: str) -> Any:
        if name == "problem":
            return self.problem
        if name == "solution":
            return self.solution
        if (self._allowed is not None and name not in self._allowed) or name not in self.outputs:
            raise KeyError(f"output of stage '{name}' is not available; is it in `requires`?")
        return self.outputs[name]

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except (KeyError, RuntimeError):
            return default

    def evidence(self) -> list[tuple[Stage, Any]]:
        """Outputs of evidence stages that have completed, in report order."""
        done = [s for s in self.stages.values() if s.evidence and s.name in self.outputs
                and (self._allowed is None or s.name in self._allowed)]
        return [(s, self.outputs[s.name]) for s in sorted(done, key=lambda s: (s.order, s.name))]


@dataclass
class StageEvent:
    stage: str
    status: Literal["started", "finished", "failed", "skipped"]
    seconds: float = 0.0
    error: str | None = None


@dataclass
class AuditResult:
    problem: str
    solution: str
    outputs: dict[str, Any]
    stages: dict[str, Stage]
    errors: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    report: str = ""

    def __getitem__(self, name: str) -> Any:
        return self.outputs[name]

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "solution": self.solution,
            "stages": {name: _jsonable(value) for name, value in self.outputs.items()},
            "errors": self.errors,
            "skipped": self.skipped,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


Solver = Callable[[AuditContext], str]


class Auditor:
    """Holds a set of stages and runs them as a dependency graph.

    solver:          produces the solution when run() is not given one.
    max_workers:     how many stages may run at once.
    fail_fast:       True: first stage failure raises StageError.
                     False: record the error, skip stages that depend on it, keep going.
    on_event:        called with a StageEvent as stages start/finish/fail/skip.
    """

    def __init__(
        self,
        stages: Iterable[Stage] = (),
        *,
        llm: Any = None,
        solver: Solver | None = None,
        max_workers: int = 4,
        fail_fast: bool = True,
        on_event: Callable[[StageEvent], None] | None = None,
        reporter: Callable[[AuditResult], str] | None = None,
    ):
        self._stages: dict[str, Stage] = {}
        for s in stages:
            self.add(s)
        self.llm = llm
        self.solver = solver
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.on_event = on_event
        self.reporter = reporter

    @classmethod
    def default(cls, **kwargs: Any) -> "Auditor":
        """The built-in audit: assumptions, tests, alternatives, evidence, confidence…"""
        from .llm import AnthropicLLM
        from .report import build_report
        from .stages import DEFAULT_STAGES, generate_solution

        kwargs.setdefault("llm", AnthropicLLM())
        kwargs.setdefault("solver", generate_solution)
        kwargs.setdefault("reporter", build_report)
        return cls(DEFAULT_STAGES, **kwargs)

    # ── composition ────────────────────────────────────────────────────────
    @property
    def stages(self) -> dict[str, Stage]:
        return dict(self._stages)

    def add(self, s: Stage) -> "Auditor":
        if s.name in BUILTIN_INPUTS:
            raise AuditGraphError(f"'{s.name}' is reserved")
        if s.name in self._stages:
            raise AuditGraphError(f"duplicate stage '{s.name}' (use replace() to override it)")
        self._stages[s.name] = s
        return self

    def replace(self, s: Stage) -> "Auditor":
        if s.name not in self._stages:
            raise AuditGraphError(f"no stage named '{s.name}' to replace")
        self._stages[s.name] = s
        return self

    def remove(self, name: str, *, cascade: bool = False) -> "Auditor":
        """Remove a stage. With cascade=True, also remove every stage that depends on it."""
        if name not in self._stages:
            raise AuditGraphError(f"no stage named '{name}'")
        # Explicit dependencies only: a stage requiring EVIDENCE adapts to whichever evidence stages remain.
        dependents = [s.name for s in self._stages.values() if name in s.requires]
        if dependents and not cascade:
            raise AuditGraphError(f"cannot remove '{name}': required by {', '.join(sorted(dependents))} (pass cascade=True)")
        del self._stages[name]
        for d in dependents:
            if d in self._stages:
                self.remove(d, cascade=True)
        return self

    def configure(self, name: str, **changes: Any) -> "Auditor":
        """Change a stage's settings, e.g. configure("evidence", evidence=False)."""
        self._stages[name] = replace(self._stages[name], **changes)
        return self

    # ── graph ──────────────────────────────────────────────────────────────
    def _resolve_requires(self, s: Stage) -> tuple[str, ...]:
        out: list[str] = []
        for r in s.requires:
            if r == EVIDENCE:
                out.extend(sorted(x.name for x in self._stages.values() if x.evidence and x.name != s.name))
            else:
                out.append(r)
        return tuple(dict.fromkeys(out))

    def plan(self) -> list[list[str]]:
        """Validate the graph and return it as waves of stages that can run together."""
        deps = {name: self._resolve_requires(s) for name, s in self._stages.items()}
        for name, reqs in deps.items():
            for r in reqs:
                if r not in self._stages and r not in BUILTIN_INPUTS:
                    raise AuditGraphError(f"stage '{name}' requires unknown stage '{r}'")
        remaining = {n: {r for r in reqs if r not in BUILTIN_INPUTS} for n, reqs in deps.items()}
        waves: list[list[str]] = []
        done: set[str] = set()
        while remaining:
            ready = sorted(n for n, r in remaining.items() if r <= done)
            if not ready:
                raise AuditGraphError(f"dependency cycle among: {', '.join(sorted(remaining))}")
            waves.append(ready)
            done.update(ready)
            for n in ready:
                del remaining[n]
        return waves

    # ── execution ──────────────────────────────────────────────────────────
    def run(self, problem: str, *, solution: str | None = None) -> AuditResult:
        """Audit `solution` for `problem`. Without a solution, the solver generates one."""
        self.plan()  # validate before spending anything
        ctx = AuditContext(problem, solution, self.llm, self._stages)
        timings: dict[str, float] = {}
        if solution is None:
            if self.solver is None:
                raise ValueError("no solution given and this Auditor has no solver")
            t0 = time.perf_counter()
            self._emit(StageEvent("solution", "started"))
            ctx._solution = self.solver(ctx)
            timings["solution"] = time.perf_counter() - t0
            self._emit(StageEvent("solution", "finished", timings["solution"]))

        deps = {n: {r for r in self._resolve_requires(s) if r not in BUILTIN_INPUTS} for n, s in self._stages.items()}
        errors: dict[str, str] = {}
        skipped: dict[str, str] = {}
        pending = dict(deps)
        running: dict[Future, tuple[str, float]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while pending or running:
                # Skip stages whose inputs failed or were skipped.
                for n in sorted(pending):
                    bad = sorted(r for r in pending[n] if r in errors or r in skipped)
                    if bad:
                        skipped[n] = f"upstream {', '.join(bad)} did not complete"
                        self._emit(StageEvent(n, "skipped", error=skipped[n]))
                        del pending[n]
                for n in sorted(pending):
                    if pending[n] <= set(ctx.outputs):
                        self._emit(StageEvent(n, "started"))
                        view = ctx.scoped(self._resolve_requires(self._stages[n]))
                        running[pool.submit(self._stages[n].fn, view)] = (n, time.perf_counter())
                        del pending[n]
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    n, t0 = running.pop(fut)
                    timings[n] = time.perf_counter() - t0
                    try:
                        ctx.outputs[n] = fut.result()
                        self._emit(StageEvent(n, "finished", timings[n]))
                    except Exception as e:  # noqa: BLE001 — stages are user code
                        self._emit(StageEvent(n, "failed", timings[n], error=str(e)))
                        if self.fail_fast:
                            for other in running:
                                other.cancel()
                            raise StageError(n, e) from e
                        errors[n] = f"{type(e).__name__}: {e}"

        result = AuditResult(
            problem=problem, solution=ctx.solution, outputs=dict(ctx.outputs), stages=dict(self._stages),
            errors=errors, skipped=skipped, timings=timings,
        )
        result.report = (self.reporter or _plain_report)(result)
        return result

    def _emit(self, event: StageEvent) -> None:
        if self.on_event:
            self.on_event(event)


def _default_render(output: Any) -> str:
    if isinstance(output, BaseModel):
        return output.model_dump_json(indent=2)
    if isinstance(output, (list, tuple)):
        return "\n".join(f"- {_default_render(x)}" for x in output) or "(none)"
    return str(output)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _plain_report(result: AuditResult) -> str:
    lines = ["PROBLEM", result.problem, "", "SOLUTION", result.solution, ""]
    for s in sorted(result.stages.values(), key=lambda s: (s.order, s.name)):
        if s.title and s.name in result.outputs:
            lines += [s.title, s.render_output(result.outputs[s.name]), ""]
    return "\n".join(lines)
