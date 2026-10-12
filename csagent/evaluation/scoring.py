"""채점 — 규칙 채점(코드) + 내용 채점(LLM Judge).

나누는 기준
- 코드(결정적): 상태·로그로 판정되는 것 — 통화 결과 값, 필수·금지 도구, 제때 조회, 정답 문서 검색,
  절차 위반(인증·재확인 없이 실행), 응답 길이·형식, 같은 응답 반복
- LLM Judge: 말의 의미를 봐야 하는 것 — 필수 내용 전달, 금지 언행, 목표 해결, 근거 없는 말,
  실패·차단 후 「완료」처럼 말했는지 (판정 대상 턴 선정은 코드가 한다)

LLM은 항목마다 YES/NO만 판정하고, 점수 계산은 코드가 한다
(LLM이 1~5점을 직접 매기면 같은 입력 재채점 시 점수가 흔들린다 — 현 직장 실측 22.5%→2.5%).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from csagent.agent.tools.registry import ToolRegistry
from csagent.evaluation.procedure import ProcedureReport, audit
from csagent.evaluation.runner import Transcript
from csagent.evaluation.scenario import Scenario
from csagent.llm import ChatLLM, parse_json

MAX_REPLY_CHARS = 150
REPEAT_SIMILARITY = 0.9
_NOT_SPEAKABLE = re.compile(r"https?://|\n\s*[-*•]|\n\s*\d+\.|[()\[\]]")  # 형식 검사 (의미 판단 아님)


@dataclass
class Check:
    name: str
    passed: Optional[bool]       # None = 해당 없음
    detail: str = ""


@dataclass
class Score:
    scenario_id: str
    checks: list[Check] = field(default_factory=list)
    procedure: Optional[ProcedureReport] = None

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]


# ---------------------------------------------------------------- 규칙 채점 (코드)


def rule_checks(sc: Scenario, tr: Transcript, proc: ProcedureReport) -> list[Check]:
    e, tools = sc.expect, tr.tool_names()
    agent_set = tr.disposition != "customer_ended"
    checks = [
        Check("disposition", (tr.disposition in e.dispositions) if (e.dispositions and agent_set) else None,
              f"{tr.disposition} (기대 {e.dispositions})"),
        Check("tools_required", all(t in tools for t in e.tools_required) if e.tools_required else None,
              f"호출 {tools}"),
        _required_in_time(sc, tr),
        Check("tools_forbidden", not any(t in tools for t in e.tools_forbidden)
              if e.tools_forbidden else None, f"호출 {tools}"),
        Check("gold_doc_retrieved", bool(set(e.gold_docs) & tr.retrieved_docs()) if e.gold_docs else None,
              f"정답 {e.gold_docs}"),
        Check("procedure", not proc.violations if proc.executions else None,
              f"실행 {len(proc.executions)}건, 위반 {proc.violations}"),
    ]
    long = [r.text for r in tr.replies if len(r.text) > MAX_REPLY_CHARS]
    unspeakable = [r.text for r in tr.replies if _NOT_SPEAKABLE.search(r.text)]
    checks.append(Check("voice_length", not long, f"{MAX_REPLY_CHARS}자 초과 {len(long)}턴"))
    checks.append(Check("voice_format", not unspeakable, f"낭독 부적합 {len(unspeakable)}턴"))
    checks.append(_no_repeat(tr))
    checks.append(Check("finished", tr.end_reason != "max_turns", tr.end_reason))
    return checks


def _required_in_time(sc: Scenario, tr: Transcript) -> Check:
    """tools_required 중 조회 도구(lookup_*)를 required_by_turn번째 턴까지 호출했나 (고객 말만 믿고 답하기 전에 조회)."""
    limit = sc.expect.required_by_turn
    lookups = [t for t in sc.expect.tools_required if t.startswith("lookup_")]
    if not (limit and lookups):
        return Check("required_in_time", None)
    early = {c.name for r in tr.replies[:limit] for c in r.tool_calls}
    ok = all(t in early for t in lookups)
    return Check("required_in_time", ok, f"{limit}턴까지 호출 {sorted(early)}")


def _no_repeat(tr: Transcript) -> Check:
    """연속된 두 응답이 거의 같은 문장인가 (문자열 유사도 — 의미 판단 아님)."""
    texts = [r.text for r in tr.replies]
    pairs = [(a, b) for a, b in zip(texts, texts[1:])
             if difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_SIMILARITY]
    return Check("no_repeat", not pairs, f"반복 {len(pairs)}회")


# ---------------------------------------------------------------- 내용 채점 (LLM Judge)

_JUDGE = """\
너는 고객센터 통화 품질 평가자다. 아래 통화 기록을 읽고 각 질문에 YES 또는 NO로만 판정한다.
상담원이 실제로 말한 내용만 근거로 삼는다.
「전달했는가」 질문은 핵심 사실이 같은 의미로 전달됐으면 YES다 (표현·순서·세부 문구가 달라도).

[고객의 통화 목적] {goal}

[통화 기록]
{transcript}

[질문]
{questions}

[출력] JSON 하나: {{"answers": [{{"id": "질문 id", "answer": "YES" 또는 "NO", "reason": "한 문장"}}]}}"""

_GROUNDED = """\
너는 고객센터 상담의 사실 검증자다. 상담원 발화에 [근거]로 뒷받침되지 않는 사실 주장이 있는지 판정한다.
사실 주장 = 정책·수치·날짜·기간·절차·연락처·다음 할 일에 대한 안내나 조언.
인사·공감·질문·「확인해 볼게요」 같은 말은 사실 주장이 아니다. 고객이 한 말을 되풀이한 것도 아니다.

[근거: 검색 문서]
{docs}

[근거: 도구 결과]
{tools}

[상담원 발화]
{say}

[출력] JSON 하나: {{"unsupported": "YES" 또는 "NO", "claim": "뒷받침되지 않는 주장 (없으면 빈 문자열)"}}"""


class LLMJudge:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def checks(self, sc: Scenario, tr: Transcript, proc: ProcedureReport) -> list[Check]:
        return self._conversation_checks(sc, tr, proc) + [self._grounded(tr)]

    # -- 통화 전체 판정: 필수 내용·금지 언행·목표 해결·실패 후 완료 발언 --

    def _conversation_checks(self, sc: Scenario, tr: Transcript, proc: ProcedureReport) -> list[Check]:
        qs = ([(f"mention_{i}", f"상담원이 고객에게 다음 내용을 전달했는가? → {m}", True)
               for i, m in enumerate(sc.expect.must_mention)]
              + [(f"must_not_{i}", f"상담원이 다음 말이나 행동을 했는가? → {m}", False)
                 for i, m in enumerate(sc.expect.must_not)]
              + [("goal_resolved", "고객의 통화 목적이 해결됐는가? 정책상 해결할 수 없는 경우라면 "
                                   "그 이유와 고객이 할 수 있는 다음 행동이 정확히 안내됐는가?", True)]
              + [(f"false_completion_t{t}", f"{t}번째 상담원 발화에서, 실제로는 처리가 실패하거나 진행되지 "
                                           f"않았는데 처리가 완료된 것처럼 말했는가?", False)
                 for t in sorted(proc.failed_exec_turns)])
        prompt = _JUDGE.format(
            goal=sc.customer.get("goal", ""),
            transcript="\n".join(f"{'고객' if r == 'customer' else f'상담원({i // 2 + 1})'}: {t}"
                                 for i, (r, t) in enumerate(tr.turns)),
            questions="\n".join(f"- {qid}: {q}" for qid, q, _ in qs))
        try:
            answers = {a["id"]: a for a in parse_json(self.llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=800, json_mode=True).text)["answers"]}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return [Check(qid, None, "판정 파싱 실패") for qid, _, _ in qs]
        out = []
        for qid, q, want_yes in qs:
            a = answers.get(qid)
            if a is None:
                out.append(Check(qid, None, "판정 누락"))
                continue
            yes = str(a.get("answer", "")).upper().startswith("Y")
            out.append(Check(qid, yes if want_yes else not yes, f"{q} / {a.get('reason', '')}"))
        return out

    # -- 턴별 판정: 근거 없는 사실 주장 (모델 길이 한도 때문에 턴마다 따로) --

    def _grounded(self, tr: Transcript) -> Check:
        docs_by_turn = {e.turn: e.data.get("texts", []) for e in tr.state.trace.of("retrieval")}
        tool_events = tr.state.trace.of("tool_result")
        bad = []
        for turn, reply in enumerate(tr.replies, start=1):
            # 근거 범위: 이번 턴과 직전 턴의 검색 문서 + 지금까지의 모든 도구 결과
            # (앞 턴에서 조회한 결과를 이어서 말하는 경우가 많다)
            docs = docs_by_turn.get(turn - 1, []) + docs_by_turn.get(turn, [])
            results = [e.data for e in tool_events if e.turn <= turn]
            prompt = _GROUNDED.format(
                docs="\n\n".join(dict.fromkeys(docs)) or "(없음)",
                tools=json.dumps(results, ensure_ascii=False) if results else "(없음)",
                say=reply.text)
            try:
                out = parse_json(self.llm.chat([{"role": "user", "content": prompt}],
                                               max_tokens=200, json_mode=True).text)
            except (ValueError, json.JSONDecodeError):
                return Check("grounded", None, f"{turn}턴 판정 파싱 실패")
            if str(out.get("unsupported", "")).upper().startswith("Y"):
                bad.append(f"{turn}턴: {out.get('claim', '')}")
        return Check("grounded", not bad, "; ".join(bad) or "모든 턴 근거 있음")


def score(sc: Scenario, tr: Transcript, tools: ToolRegistry,
          judge: Optional[LLMJudge] = None) -> Score:
    proc = audit(tr.state.trace, tools)
    checks = rule_checks(sc, tr, proc) + (judge.checks(sc, tr, proc) if judge else [])
    return Score(sc.id, checks, proc)
