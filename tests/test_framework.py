"""Framework tests. No network: every stage talks to FakeLLM."""

from __future__ import annotations

import contextlib
import os
from unittest import mock
import io
import json
import threading
import time
import unittest
from pathlib import Path

from ai_auditor import (
    AuditGraphError, Auditor, Completion, StageError, load_plugin, stage,
)
from ai_auditor.cli import main as cli_main
from ai_auditor.report import build_report
from ai_auditor.stages import DEFAULT_STAGES
from ai_auditor import stages as S

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Deterministic stand-in for AnthropicLLM. Records every call."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[tuple[str, str]] = []  # (kind, prompt)
        self.lock = threading.Lock()

    def _record(self, kind: str, prompt: str) -> None:
        with self.lock:
            self.calls.append((kind, prompt))
        if self.delay:
            time.sleep(self.delay)

    def complete(self, prompt, *, effort="medium", max_tokens=4000, tools=None):
        self._record("complete", prompt)
        if tools and tools[0]["name"] == "code_execution":
            return Completion("Checked 2+2 numerically.\nRESULT: PASS", ["text", "bash_code_execution_tool_result", "text"])
        if tools and tools[0]["name"] == "web_search":
            return Completion("Sources agree that 2+2=4.", ["text"])
        if prompt.startswith("In a short paragraph"):
            return Completion("Little remains uncertain.")
        return Completion("4")

    def parse(self, prompt, schema, *, effort="medium", max_tokens=4000):
        self._record(f"parse:{schema.__name__}", prompt)
        if schema is S.AssumptionList:
            return S.AssumptionList(assumptions=["Base-10 arithmetic"])
        if schema is S.VerificationList:
            return S.VerificationList(verifications=[S.AssumptionVerification(assumption="Base-10 arithmetic", verdict="valid", explanation="Standard.")])
        if schema is S.AlternativeList:
            return S.AlternativeList(alternatives=["5 (in a joke)"])
        if schema is S.ContradictionList:
            return S.ContradictionList(contradictions=[])
        if schema is S.ConfidenceResult:
            return S.ConfidenceResult(score=0.97, reasoning="Verified by code and sources.")
        raise AssertionError(f"unexpected schema {schema}")

    def prompts(self, kind: str) -> list[str]:
        return [p for k, p in self.calls if k == kind]


def default_auditor(llm=None, **kw) -> Auditor:
    return Auditor(DEFAULT_STAGES, llm=llm or FakeLLM(), solver=S.generate_solution, reporter=build_report, **kw)


class DefaultAuditTests(unittest.TestCase):
    def test_runs_end_to_end_and_reports_every_section(self):
        llm = FakeLLM()
        result = default_auditor(llm).run("What is 2+2?")
        self.assertEqual(result.solution, "4")
        self.assertEqual(result["confidence"].score, 0.97)
        self.assertTrue(result["external_test"].ran)
        self.assertTrue(result["external_test"].passed)
        for heading in ("CONFIDENCE: 0.97", "ASSUMPTIONS", "ALTERNATIVES CONSIDERED", "EVIDENCE GATHERED",
                        "CONTRADICTIONS / COUNTER-EVIDENCE", "EXTERNAL TESTS", "REMAINING UNCERTAINTY"):
            self.assertIn(heading, result.report)
        json.dumps(result.to_dict())  # serializable
        self.assertEqual(result.errors, {})

    def test_plan_matches_original_pipeline_shape(self):
        self.assertEqual(default_auditor().plan(), [
            ["alternatives", "assumptions", "evidence", "external_test"],
            ["contradictions", "verifications"],
            ["confidence"],
            ["uncertainty"],
        ])

    def test_independent_stages_run_concurrently(self):
        llm = FakeLLM(delay=0.2)
        t0 = time.perf_counter()
        default_auditor(llm).run("What is 2+2?", solution="4")
        elapsed = time.perf_counter() - t0
        # 8 stages x 0.2s sequentially = 1.6s; 4 waves -> ~0.8s
        self.assertLess(elapsed, 1.3, elapsed)

    def test_given_solution_skips_the_solver(self):
        llm = FakeLLM()
        result = default_auditor(llm).run("What is 2+2?", solution="Four.")
        self.assertEqual(result.solution, "Four.")
        self.assertNotIn("What is 2+2?", llm.prompts("complete"))
        self.assertNotIn("solution", result.timings)


class ExtensionTests(unittest.TestCase):
    def test_custom_evidence_stage_feeds_confidence_and_report(self):
        @stage(requires=("solution",), evidence=True, title="LENGTH CHECK", summarize=lambda n: f"answer has {n} words")
        def length_check(ctx):
            return len(ctx.solution.split())

        llm = FakeLLM()
        auditor = default_auditor(llm).add(length_check)
        self.assertIn("length_check", auditor.plan()[0])
        self.assertIn("length_check", auditor._resolve_requires(auditor.stages["confidence"]))
        result = auditor.run("What is 2+2?", solution="It is 4")
        [conf_prompt] = llm.prompts("parse:ConfidenceResult")
        self.assertIn("LENGTH CHECK:\nanswer has 3 words", conf_prompt)
        self.assertIn("LENGTH CHECK\n3", result.report)

    def test_replace_and_configure(self):
        @stage(name="evidence", requires=("solution",), title="EVIDENCE GATHERED", order=40)
        def offline(ctx):
            return "offline"

        llm = FakeLLM()
        auditor = default_auditor(llm).replace(offline).configure("external_test", evidence=False)
        result = auditor.run("q", solution="a")
        self.assertEqual(result["evidence"], "offline")
        self.assertNotIn("EXTERNAL TESTS:", llm.prompts("parse:ConfidenceResult")[0])

    def test_remove_requires_cascade_for_dependents(self):
        auditor = default_auditor()
        with self.assertRaises(AuditGraphError):
            auditor.remove("contradictions")
        auditor.remove("contradictions", cascade=True)
        self.assertNotIn("uncertainty", auditor.stages)
        self.assertIn("confidence", auditor.stages)  # EVIDENCE dependency adapts
        auditor.run("q", solution="a")

    def test_graph_errors(self):
        @stage(requires=("nope",))
        def broken(ctx):
            return 1

        with self.assertRaises(AuditGraphError):
            default_auditor().add(broken).plan()

        @stage(requires=("b",))
        def a(ctx):
            return 1

        @stage(requires=("a",))
        def b(ctx):
            return 1

        with self.assertRaises(AuditGraphError) as cm:
            Auditor([a, b]).plan()
        self.assertIn("cycle", str(cm.exception))
        with self.assertRaises(AuditGraphError):
            default_auditor().add(S.confidence)  # duplicate
        with self.assertRaises(AuditGraphError):
            Auditor().add(stage(name="solution")(lambda ctx: 1))  # reserved

    def test_fail_fast_raises_stage_error(self):
        @stage(requires=("solution",))
        def explode(ctx):
            raise ValueError("boom")

        with self.assertRaises(StageError) as cm:
            default_auditor().add(explode).run("q", solution="a")
        self.assertEqual(cm.exception.stage, "explode")

    def test_continue_on_error_records_and_skips_dependents(self):
        @stage(requires=("solution",), evidence=True, title="FLAKY")
        def flaky(ctx):
            raise RuntimeError("upstream service down")

        events = []
        auditor = default_auditor(fail_fast=False, on_event=events.append).add(flaky)
        result = auditor.run("q", solution="a")
        self.assertIn("flaky", result.errors)
        self.assertIn("confidence", result.skipped)   # needs every evidence stage
        self.assertIn("uncertainty", result.skipped)  # needs confidence
        self.assertIn("verifications", result.outputs)
        self.assertIn("(stage failed: RuntimeError: upstream service down)", result.report)
        self.assertIn("Confidence could not be computed", result.report)
        statuses = {(e.stage, e.status) for e in events}
        self.assertIn(("flaky", "failed"), statuses)
        self.assertIn(("confidence", "skipped"), statuses)

    def test_stage_cannot_read_undeclared_outputs(self):
        @stage(requires=("solution",))
        def sneaky(ctx):
            return ctx["alternatives"]

        result = default_auditor(fail_fast=False).add(sneaky).run("q", solution="a")
        self.assertIn("sneaky", result.errors)
        self.assertIn("KeyError", result.errors["sneaky"])

    def test_auditor_without_builtins(self):
        @stage(requires=("problem",))
        def echo(ctx):
            return ctx.problem.upper()

        result = Auditor([echo]).run("hi", solution="x")
        self.assertEqual(result["echo"], "HI")


class PluginTests(unittest.TestCase):
    def test_citation_plugin_from_file(self):
        llm = FakeLLM()
        auditor = default_auditor(llm)
        load_plugin(auditor, str(ROOT / "examples" / "citation_check.py"))
        result = auditor.run("q", solution="See https://example.org/a for details.")
        self.assertEqual(result["citations"].urls, ["https://example.org/a"])
        self.assertIn("The solution cites 1 source(s).", llm.prompts("parse:ConfidenceResult")[0])
        self.assertIn("CITATIONS", result.report)

    def test_rubric_plugin_replaces_web_search(self):
        auditor = default_auditor()
        load_plugin(auditor, str(ROOT / "examples" / "rubric.py"))
        self.assertIn("rubric", auditor.stages)
        self.assertEqual(auditor.stages["evidence"].description, "Offline replacement for the web-search evidence stage.")

    def test_cli_list_stages_with_plugin_and_skip(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli_main(["--list-stages", "--plugin", str(ROOT / "examples" / "citation_check.py"), "--skip", "evidence"])
        text = out.getvalue()
        self.assertIn("citations [evidence]", text)
        self.assertNotIn("contradictions", text)  # depended on evidence
        self.assertNotIn("uncertainty", text)
        self.assertIn("wave", text)

    def test_cli_rejects_bad_plugin(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cli_main(["--list-stages", "--plugin", "no_such_module_xyz"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("cannot import plugin", err.getvalue())


class StyleTests(unittest.TestCase):
    def test_clean_removes_em_dashes_and_emojis(self):
        from ai_auditor.style import clean
        self.assertEqual(clean("Honestly — very little remains."), "Honestly, very little remains.")
        self.assertEqual(clean("closed—verifiable"), "closed, verifiable")
        self.assertEqual(clean("Checks out ✅ twice 👍."), "Checks out twice.")
        self.assertEqual(clean("— leading dash"), "leading dash")
        self.assertEqual(clean("ranges like 3–5 stay"), "ranges like 3–5 stay")  # en dash kept

    def test_generated_outputs_are_cleaned_but_user_solution_is_not(self):
        from ai_auditor.style import clean_value

        class DashyLLM(FakeLLM):
            def complete(self, prompt, **kw):
                c = super().complete(prompt, **kw)
                return Completion(c.text + " — sure 🚀", c.block_types)

            def parse(self, prompt, schema, **kw):
                out = super().parse(prompt, schema, **kw)
                if schema is S.ConfidenceResult:
                    return S.ConfidenceResult(score=out.score, reasoning="High — verified ✨")
                return out

        user_solution = "4 — obviously"
        result = default_auditor(DashyLLM(), output_filter=clean_value).run("What is 2+2?", solution=user_solution)
        self.assertEqual(result.solution, user_solution)  # the audited text is never altered
        report_without_solution = result.report.replace(user_solution, "")
        self.assertNotIn("—", report_without_solution)
        self.assertNotIn("🚀", result.report)
        self.assertEqual(result["confidence"].reasoning, "High, verified")
        generated = default_auditor(DashyLLM(), output_filter=clean_value).run("What is 2+2?")
        self.assertNotIn("—", generated.solution)

    def test_anthropic_llm_sends_style_instruction(self):
        from ai_auditor.llm import AnthropicLLM
        from ai_auditor.style import STYLE_INSTRUCTION
        client = mock.MagicMock()
        client.messages.create.return_value = mock.MagicMock(stop_reason="end_turn", content=[])
        client.messages.parse.return_value = mock.MagicMock(stop_reason="end_turn", parsed_output="ok")
        llm = AnthropicLLM(client=client)
        llm.complete("hi")
        llm.parse("hi", S.AssumptionList)
        self.assertEqual(client.messages.create.call_args.kwargs["system"], STYLE_INSTRUCTION)
        self.assertEqual(client.messages.parse.call_args.kwargs["system"], STYLE_INSTRUCTION)
        AnthropicLLM(client=client, system=None).complete("hi")
        self.assertNotIn("system", client.messages.create.call_args.kwargs)


class CredentialPreflightTests(unittest.TestCase):
    def test_missing_credentials_fail_before_any_stage(self):
        err = io.StringIO()
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
        env["XDG_CONFIG_HOME"] = "/nonexistent-config-home"
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stderr(err), \
                mock.patch("ai_auditor.core.Auditor.run") as run, self.assertRaises(SystemExit) as cm:
            cli_main(["What is 2+2?", "--solution", "4"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("No Anthropic API credentials found", err.getvalue())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
