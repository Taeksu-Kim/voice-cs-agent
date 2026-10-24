"""이상 탐지 (csagent/anomaly.py) — 규칙 신호·채점 신호·가드레일 집계·케이스 묶음."""
import json

import yaml

from csagent.agent.loop import FALLBACK
from csagent.anomaly import check_signals, guard_counts, rule_signals, write_case
from csagent.evaluation.repro import ReproSpec


def _ev(turn, kind, data, step=1):
    return {"turn": turn, "kind": kind, "data": data, "t": 0, "step": step, "at": f"t{turn}.s{step}.{kind}"}


def _row(trace=(), turns=(), **kw):
    return {"scenario_id": "x", "repeat": 0, "trace": list(trace), "turns": [list(t) for t in turns],
            "disposition": "resolved", "end_reason": "agent_ended", **kw}


def _kinds(signals):
    return [(s.kind, s.level, s.at) for s in signals]


def test_trace_rules_locate_defects():
    row = _row(trace=[_ev(1, "tool_result", {"name": "report_card_lost", "ok": False, "error": "Timeout"}),
                      _ev(1, "execution_unknown", {"name": "report_card_lost", "args": {}}),
                      _ev(2, "spoken", {"text": FALLBACK}, step=2),
                      _ev(3, "turn_error", {"error": "KeyError"}, step=0)],
               end_reason="max_turns")
    assert _kinds(rule_signals(row)) == [("tool_failed", "warn", "t1.s1"), ("execution_unknown", "bad", "t1.s1"),
                                         ("fallback", "bad", "t2.s2"), ("turn_error", "bad", "t3.s0"),
                                         ("max_turns", "warn", "")]


def test_repeat_and_length_from_heard_history():
    long = "가" * 151
    row = _row(turns=[("agent", "안녕하세요"), ("customer", "환불돼요?"), ("agent", "판매자 답변 후 가능해요."),
                      ("customer", "그럼 언제요?"), ("agent", "판매자 답변 후 가능해요."),
                      ("customer", "그럼 언제요?"), ("agent", "판매자 답변 후 가능해요."),    # 같은 질문에 같은 답은 정상
                      ("customer", "자세히요"), ("agent", long)])
    assert _kinds(rule_signals(row)) == [("repeat", "warn", "t2"), ("over_len", "warn", "t4")]


def test_unresolved_only_for_unscored_calls():
    assert _kinds(rule_signals({"history": [], "trace": [], "disposition": "unresolved"})) == [("unresolved", "warn", "")]
    assert rule_signals(_row(disposition="unresolved", checks=[])) == []   # 평가는 disposition 채점이 기대값과 비교


def test_check_signals_split_grounded_turns_and_skip_covered():
    row = _row(checks=[{"name": "grounded", "passed": False, "detail": "3턴: 「A」 unsupported; 5턴: 「B」 contradicted"},
                       {"name": "no_repeat", "passed": False, "detail": "반복 1회"},
                       {"name": "disposition", "passed": False, "detail": "unresolved (기대 ['resolved'])"},
                       {"name": "milestones", "passed": True, "detail": ""}])
    assert _kinds(check_signals(row)) == [("check:grounded", "bad", "t3"), ("check:grounded", "bad", "t5"),
                                          ("check:disposition", "bad", "")]


def test_guards_are_counted_not_signals():
    row = _row(trace=[_ev(1, "gate", {"name": "x", "allowed": False, "reason": "result_unheard"}),
                      _ev(1, "unheard_drop", {"text": "확인해 볼게요."}), _ev(1, "gate", {"name": "y", "allowed": True})])
    assert guard_counts(row) == {"gate:result_unheard": 1, "unheard_drop": 1} and rule_signals(row) == []


def test_case_bundle_has_valid_repro_spec(tmp_path):
    src = tmp_path / "runs" / "r1"
    src.mkdir(parents=True)
    trace = [_ev(1, "state", {}, step=0), _ev(1, "llm", {"call_id": "abc", "role": "general"})]
    (src / "llm_calls.jsonl").write_text(json.dumps({"call_id": "abc"}) + "\n" + json.dumps({"call_id": "zzz"}) + "\n")
    row = _row(trace=trace, turns=[("customer", "a"), ("agent", "나" * 160)])
    item = {"source": src, "key": "x:0", "row": row, "signals": rule_signals(row), "guards": {}}
    d = write_case(tmp_path / "cases", tmp_path, item)
    spec = ReproSpec.from_dict(yaml.safe_load((d / "repro.yaml").read_text(encoding="utf-8")))
    assert (spec.case, spec.at, spec.mode, spec.fail_if) == ("runs/r1/x:0", "t1", "turn", [{"len_over": 150}])
    assert [json.loads(l)["call_id"] for l in (d / "llm_calls.jsonl").read_text().splitlines()] == ["abc"]
    assert "flowchart" in (d / "trace.md").read_text(encoding="utf-8")


def test_spec_falls_back_to_step_without_state_snapshot(tmp_path):
    src = tmp_path / "runs" / "old"
    src.mkdir(parents=True)
    row = _row(turns=[("customer", "a"), ("agent", "나" * 160)])
    d = write_case(tmp_path / "cases", tmp_path, {"source": src, "key": "x:0", "row": row,
                                                  "signals": rule_signals(row), "guards": {}})
    spec = yaml.safe_load((d / "repro.yaml").read_text(encoding="utf-8"))
    assert (spec["mode"], spec["at"], spec["fail_if"]) == ("step", "t1.s1", [{"len_over": 150}])
