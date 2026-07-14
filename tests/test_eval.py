"""Tests for the eval harness.

Everything here runs without an API key and without a network: a stub client
plays the part of the Messages API, so the plumbing (request shape, judge
parsing, A/B de-swapping, curve maths, policy selection) is testable on its own.
The one thing these tests deliberately cannot check is whether the judge's
grades are any good — that is what the golden set and a real run are for.
"""

import json

import pytest

from tokenlens.eval import judge
from tokenlens.eval.api import ApiError, text_of
from tokenlens.eval.harness import (
    Arm,
    CaseResult,
    build_body,
    build_curve,
    build_policy,
    run_eval,
)
from tokenlens.eval.tasks import GoldenTask, load_tasks


def _task(tid="t1", cls="long-doc-qa"):
    return GoldenTask(
        id=tid,
        task_class=cls,
        question="What colour is the roof?",
        context="The roof is green. " * 40,
        reference="Green.",
    )


class StubClient:
    """Stands in for the Messages API.

    `answers` maps a substring of the outgoing prompt to the text we reply with,
    so a test can make the compressed request produce a worse answer than the
    cleartext one. Judge calls (detected by the output_config) return a canned
    graded verdict.
    """

    def __init__(self, answers=None, grades=(90, 90), tokens=100):
        self.answers = answers or {}
        self.grades = grades
        self.tokens = tokens
        self.calls = []
        self.judge_prompts = []

    def count_tokens(self, payload):
        # A compressed payload is shorter; make token counts track text length.
        return len(json.dumps(payload)) // 10

    def messages(self, payload):
        prompt = json.dumps(payload)
        self.calls.append(payload)
        if "output_config" in payload:
            self.judge_prompts.append(payload["messages"][0]["content"])
            a, b = self.grades
            return _msg(json.dumps({
                "reasoning": "r", "grade_a": a, "grade_b": b, "note": "n",
            }))
        for needle, answer in self.answers.items():
            if needle in prompt:
                return _msg(answer)
        return _msg("default answer")


def _msg(text):
    return {"content": [{"type": "text", "text": text}]}


# --- request shape --------------------------------------------------------
def test_build_body_has_a_cached_system_prefix_and_a_volatile_tail():
    payload = json.loads(build_body(_task(), "claude-opus-4-8"))
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    blocks = payload["messages"][0]["content"]
    assert [b["text"] for b in blocks] == [_task().context, _task().question]
    # the tail must carry no breakpoint, or nothing would be compressible
    assert all("cache_control" not in b for b in blocks)


def test_bundled_golden_set_loads_and_covers_several_classes():
    tasks = load_tasks()
    assert len(tasks) >= 8
    assert len({t.task_class for t in tasks}) >= 5
    # every reference must be checkable against its own context
    assert all(t.reference and t.question and t.context for t in tasks)


def test_load_tasks_rejects_a_task_missing_a_reference(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "task_class": "c",
                             "question": "q", "context": "ctx"}) + "\n")
    with pytest.raises(ValueError, match="reference"):
        load_tasks(str(p))


# --- the judge ------------------------------------------------------------
def test_judge_maps_grades_back_through_the_ab_swap():
    """The judge sees A/B in a random order; we must un-swap before recording."""
    swapped = judge._swap_for("case:arm")
    client = StubClient(grades=(30, 80))  # A=30, B=80
    j = judge.grade(
        client, case_key="case:arm", request_text="q",
        cleartext_answer="good", compressed_answer="bad",
    )
    if swapped:  # compressed was shown as A
        assert (j.compressed_grade, j.cleartext_grade) == (30, 80)
    else:
        assert (j.cleartext_grade, j.compressed_grade) == (30, 80)
    assert j.swapped is swapped


def test_ab_position_is_deterministic_but_not_constant():
    keys = [f"task{i}:llmlingua2 r=0.6" for i in range(24)]
    flips = [judge._swap_for(k) for k in keys]
    assert flips == [judge._swap_for(k) for k in keys]  # reproducible
    assert 0 < sum(flips) < len(flips)                  # actually shuffles


def test_judge_never_sees_which_answer_was_compressed():
    client = StubClient(grades=(90, 90))
    judge.grade(client, case_key="k", request_text="the request",
                cleartext_answer="ALPHA", compressed_answer="BETA")
    prompt = client.judge_prompts[0].lower()
    assert "compress" not in prompt
    assert "answer_a" in prompt and "answer_b" in prompt


def test_retention_is_capped_at_one_and_survives_a_zero_baseline():
    assert judge.Judgement(80, 100, "", "", False).retention == 1.0   # got luckier
    assert judge.Judgement(0, 0, "", "", False).retention == 1.0      # broken task
    assert judge.Judgement(100, 75, "", "", False).retention == 0.75


def test_judge_rejects_a_non_json_reply():
    with pytest.raises(ApiError, match="did not return JSON"):
        judge.parse_response(_msg("Sure! Here's my assessment:"))


def test_judge_clamps_out_of_range_grades():
    parsed = judge.parse_response(_msg(json.dumps(
        {"reasoning": "r", "grade_a": 140, "grade_b": -20, "note": "n"})))
    assert (parsed["grade_a"], parsed["grade_b"]) == (100, 0)


def test_judge_request_is_bounded():
    prompt = judge.build_prompt("x" * 50_000, None, "a", "b")
    assert len(prompt) < 12_000
    assert "elided by the judge" in prompt


# --- the run --------------------------------------------------------------
def test_run_eval_grades_each_arm_against_one_shared_baseline():
    client = StubClient(grades=(70, 70))
    report = run_eval([_task()], client, arms=[Arm("safe")],
                      model="m", concurrency=1, progress=False)
    # one baseline completion + one compressed completion + one judge call
    completions = [c for c in client.calls if "output_config" not in c]
    assert len(completions) == 2
    assert len(client.judge_prompts) == 1
    assert report.cases[0].ok


def test_an_untouched_request_is_not_sent_to_the_judge():
    """Nothing eligible means the bytes went upstream unchanged. Don't pay to
    confirm the model agrees with itself."""
    code = GoldenTask("c1", "code-context", "What does it return?",
                      "    def f(x):\n        return x + 1\n", "x plus one")
    client = StubClient()
    report = run_eval([code], client, arms=[Arm("safe")],
                      model="m", concurrency=1, progress=False)
    case = report.cases[0]
    assert case.compressed_blocks == 0
    assert case.retention == 1.0
    assert client.judge_prompts == []
    assert "untouched" in case.note


def test_a_failed_api_call_records_an_error_and_does_not_crash_the_run():
    class Boom(StubClient):
        def messages(self, payload):
            raise ApiError(529, "overloaded")

    report = run_eval([_task()], Boom(), arms=[Arm("safe")],
                      model="m", concurrency=1, progress=False)
    assert report.cases[0].error is not None
    assert report.errors and "overloaded" in report.errors[0]
    assert report.curve == []  # a failed case must not become a data point


# --- curve and policy -----------------------------------------------------
def _case(arm, retention, base=1000, comp=600, cls="long-doc-qa", blocks=1):
    return CaseResult(
        task_id=f"t-{arm}-{retention}", task_class=cls, arm=arm,
        method="llmlingua2", rate=0.6, baseline_tokens=base, compressed_tokens=comp,
        cleartext_grade=100, compressed_grade=int(100 * retention),
        retention=retention, compressed_blocks=blocks,
    )


def test_curve_is_ordered_most_aggressive_first():
    cases = [_case("r=0.4", 0.8, comp=400), _case("r=0.8", 1.0, comp=900)]
    curve = build_curve(cases, tolerance=0.99)
    assert [s.arm for s in curve] == ["r=0.4", "r=0.8"]
    assert curve[0].savings_pct == 60.0


def test_an_arm_that_degrades_quality_fails_however_much_it_saves():
    curve = build_curve([_case("r=0.4", 0.90), _case("r=0.4", 0.92)], tolerance=0.99)
    assert curve[0].passes is False
    assert curve[0].degraded == 2
    assert curve[0].worst_retention == 0.90


def test_policy_picks_the_most_aggressive_arm_that_held_quality():
    cases = [
        _case("aggressive", 0.80, comp=300),   # biggest saving, fails
        _case("middle", 1.0, comp=500),        # passes
        _case("timid", 1.0, comp=900),         # passes, saves less
    ]
    by_class = {"long-doc-qa": build_curve(cases, 0.99)}
    policy = build_policy(by_class, build_curve(cases, 0.99), 0.99)
    assert policy["long-doc-qa"]["arm"] == "middle"
    assert policy["default"]["arm"] == "middle"


def test_policy_is_none_when_nothing_holds_quality():
    cases = [_case("a", 0.5), _case("b", 0.7)]
    curve = build_curve(cases, 0.99)
    policy = build_policy({"long-doc-qa": curve}, curve, 0.99)
    assert policy["long-doc-qa"]["method"] == "none"
    assert "no arm held quality" in policy["long-doc-qa"]["reason"]


def test_the_control_arm_compresses_nothing_and_measures_the_judges_noise():
    """The control judges cleartext against cleartext. It should score 100%; what
    it actually scores is the error bar on every other row."""
    from tokenlens.eval.harness import noise_floor

    client = StubClient(grades=(100, 90))   # the judge disagrees with itself
    report = run_eval([_task()], client, arms=[Arm("control")],
                      model="m", concurrency=1, progress=False)
    case = report.cases[0]
    assert case.compressed_tokens == case.baseline_tokens  # nothing was compressed
    assert case.retention < 1.0                            # ...yet quality "dropped"
    assert noise_floor(report.curve) == pytest.approx(1.0 - case.retention, abs=1e-3)
    assert "control" in case.note


def test_an_arm_inside_the_noise_floor_is_unresolved_not_passing():
    control = _case("control", 0.90); control.method = "control"
    gentle = _case("r=0.8", 0.97, comp=900)   # loses 3% — but the floor is 10%
    curve = build_curve([control, gentle], tolerance=0.99)
    by_arm = {s.arm: s for s in curve}
    assert by_arm["control"].is_control
    assert by_arm["r=0.8"].resolved is False   # 3% loss < 10% noise: unmeasurable


def test_policy_will_not_promote_an_arm_that_beat_a_failure_below_it():
    """Quality cannot improve as you delete more. When it appears to, that is
    noise, and picking the winner is how noise becomes a production setting."""
    cases = [
        _case("r=0.8", 0.80, comp=900),   # gentle arm FAILS
        _case("r=0.4", 1.00, comp=400),   # aggressive arm 'passes' — impossible
    ]
    curve = build_curve(cases, 0.99)
    policy = build_policy({"long-doc-qa": curve}, curve, 0.99)
    p = policy["long-doc-qa"]
    assert p["method"] == "none"          # the old max(savings) picked r=0.4 here
    assert "r=0.4" not in str(p.get("arm", ""))


def test_the_bar_drops_by_the_noise_floor_so_the_judge_is_not_billed_to_the_compressor():
    """A 99% tolerance against a judge that scores the *uncompressed* answer 94%
    is unpassable by construction. Only the loss beyond the floor is compression's."""
    from tokenlens.eval.harness import quality_threshold

    control = _case("control", 0.94); control.method = "control"
    arm = _case("r=0.8", 0.95, comp=850)      # loses 5% gross, 0% net of the floor
    curve = build_curve([control, arm], tolerance=0.99)
    by_arm = {s.arm: s for s in curve}
    assert quality_threshold(0.99, 0.06) == pytest.approx(0.93)
    assert by_arm["r=0.8"].threshold == pytest.approx(0.93)
    assert by_arm["r=0.8"].passes is True      # would have "failed" a flat 99%
    assert by_arm["r=0.8"].resolved is False   # ...but only provisionally

    policy = build_policy({"long-doc-qa": curve}, curve, 0.99)
    assert policy["long-doc-qa"]["confidence"] == "provisional"
    assert "not proven" not in policy["long-doc-qa"]["caveat"]  # says failure-to-detect
    assert "inside the" in policy["long-doc-qa"]["caveat"]


def test_a_steady_judge_gets_the_tolerance_the_user_actually_asked_for():
    """Floor 0 must leave the bar exactly where the user set it — the correction
    is for a broken instrument, not a discount everyone gets."""
    control = _case("control", 1.0); control.method = "control"
    arm = _case("r=0.8", 0.98, comp=850)      # 2% loss, real, and over tolerance
    curve = build_curve([control, arm], tolerance=0.99)
    by_arm = {s.arm: s for s in curve}
    assert by_arm["r=0.8"].threshold == pytest.approx(0.99)
    assert by_arm["r=0.8"].passes is False
    assert by_arm["r=0.8"].resolved is True   # a clean judge resolves a 2% drop


def test_a_class_whose_control_is_wild_is_refused_not_certified():
    """If the judge loses 16% grading cleartext against cleartext, no compression
    result on that class means anything — including a flattering one."""
    control = _case("control", 0.84); control.method = "control"
    flattering = _case("r=0.4", 0.99, comp=400)
    curve = build_curve([control, flattering], tolerance=0.99)
    policy = build_policy({"summarization": curve}, curve, 0.99)
    p = policy["summarization"]
    assert p["method"] == "none"
    assert p["unmeasurable"] is True
    assert "control arm compressed nothing" in p["reason"]


def test_repeats_average_the_cell_and_record_the_spread():
    class Flaky(StubClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def messages(self, payload):
            if "output_config" in payload:
                self.n += 1
                # judge grades the same pair 100/100 then 100/60
                self.grades = (100, 100) if self.n == 1 else (100, 60)
            return super().messages(payload)

    report = run_eval([_task()], Flaky(), arms=[Arm("safe")], model="m",
                      repeats=2, concurrency=1, progress=False)
    case = report.cases[0]
    assert case.repeats == 2
    assert len(case.retentions) == 2
    assert case.retention == pytest.approx(sum(case.retentions) / 2)
    assert case.spread > 0        # the two repeats disagreed — that is the signal


def test_policy_is_chosen_per_class_not_globally():
    """Extraction can be far more fragile than summarisation. Averaging the two
    is exactly the mistake the harness exists to prevent."""
    cases = (
        [_case("r=0.4", 1.0, cls="summarization")] * 3 +
        [_case("r=0.4", 0.6, cls="extraction")] * 3
    )
    by_class = {
        c: build_curve([x for x in cases if x.task_class == c], 0.99)
        for c in ("summarization", "extraction")
    }
    policy = build_policy(by_class, build_curve(cases, 0.99), 0.99)
    assert policy["summarization"]["arm"] == "r=0.4"
    assert policy["extraction"]["method"] == "none"


# --- the proxy's shadow-judge plumbing ------------------------------------
def test_the_answer_is_reassembled_from_a_streamed_response():
    """Streaming is the common case for agents, so the judge has to be able to
    grade an answer it only ever saw as SSE deltas."""
    from tokenlens.proxy import _extract_text_from_sse

    sse = "\n".join([
        'event: message_start',
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"The roof "}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"is green."}}',
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","text":"hmm"}}',
        'data: {"type":"message_delta","usage":{"output_tokens":4}}',
        'data: [DONE]',
        '',
    ])
    assert _extract_text_from_sse(sse) == "The roof is green."


def test_the_judge_is_only_shown_the_user_turns():
    from tokenlens.proxy import _user_text

    payload = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "the question"},
                                     {"type": "tool_result", "content": "ignored"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "an earlier answer"}]},
    ]}
    text = _user_text(payload)
    assert "the question" in text
    assert "an earlier answer" not in text  # grading against our own prior output


# --- report shape ---------------------------------------------------------
def test_report_serialises_to_json():
    client = StubClient(grades=(88, 88))
    report = run_eval([_task(), _task("t2", "extraction")], client,
                      arms=[Arm("safe")], model="m", concurrency=1, progress=False)
    blob = json.dumps(report.to_dict())
    assert json.loads(blob)["policy"]
    assert text_of(_msg("hi")) == "hi"
