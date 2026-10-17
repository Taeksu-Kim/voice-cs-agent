"""채점 — 규칙 채점(코드) + 내용 채점(LLM Judge).

나누는 기준
- 코드(결정적): 상태·로그로 판정되는 것 — 통화 결과 값, 필수·금지 도구, 제때 조회, 정답 문서 검색,
  절차 위반(인증·재확인 없이 실행), 응답 길이·형식, 같은 응답 반복
- LLM Judge: 말의 의미를 봐야 하는 것 — 필수 내용 전달, 금지 언행, 목표 해결, 근거 없는 말,
  실패·차단 후 「완료」처럼 말했는지 (판정 대상 턴 선정은 코드가 한다)

진단(diagnostics): 에이전트 통과 여부에는 넣지 않고, 실패를 해석할 때 보는 항목 — 시뮬레이터가 설정에 없는
사실을 지어냈는지 등. 실패가 에이전트 탓인지 측정 잡음인지 나누는 데 쓴다.

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
from csagent.evaluation.path import check_path, critical_executions
from csagent.evaluation.procedure import ProcedureReport, audit
from csagent.evaluation.runner import Transcript
from csagent.evaluation.scenario import Scenario
from csagent.llm import ChatLLM, parse_json

MAX_REPLY_CHARS = 150   # 판정: 고객이 대답하기 전까지 한 턴에 말한 전체 (프롬프트 목표 120 + 여유)
MAX_UTTERANCE_CHARS = 120   # 진단: 스텝 하나가 스트리밍한 발화 (도구 대기로 끊기는 단위)
REPEAT_SIMILARITY = 0.9
_NUMBER = re.compile(r"\d[\d,.]*")
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
    diagnostics: list[Check] = field(default_factory=list)   # 통과 판정에 안 들어감

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
    checks += path_checks(sc.expect.milestones, tr.state.trace.to_dicts())
    long = [r.text for r in tr.replies if len(r.text) > MAX_REPLY_CHARS]
    unspeakable = [r.text for r in tr.replies if _NOT_SPEAKABLE.search(r.text)]
    checks.append(Check("voice_length", not long, f"{MAX_REPLY_CHARS}자 초과 {len(long)}턴"))
    checks.append(Check("voice_format", not unspeakable, f"낭독 부적합 {len(unspeakable)}턴"))
    checks.append(_no_repeat(tr))
    checks.append(Check("finished", tr.end_reason != "max_turns", tr.end_reason))
    return checks


def path_checks(milestones: list, trace: list[dict]) -> list[Check]:
    """지점 순서(milestones)와 치명적 실행(critical_exec). trace만 보므로 저장된 결과 재채점에도 쓴다."""
    out = []
    if milestones:
        r = check_path(milestones, trace)
        out.append(Check("milestones", r.passed,
                         f"{r.reached}/{r.total}" + (f" 끊긴 지점 {r.broken_at}" if r.broken_at else "")))
    else:
        out.append(Check("milestones", None))
    bad = critical_executions(milestones, trace)
    out.append(Check("critical_exec", not bad, "; ".join(bad) or "없음"))
    return out


def speech_diagnostics(replies: list[str], trace: list[dict]) -> list[Check]:
    """말하기 진단 (통과 판정에 안 들어감). 응답 텍스트와 trace만 보므로 저장된 결과 재채점에도 쓴다.

    글자 수(voice_length)는 대리 지표다. 실제로 보고 싶은 것 — 대답 전에 들어야 할 질문·숫자의 양,
    도구 대기로 끊기는 발화 하나의 길이, 버려졌지만 이미 스트리밍으로 나간 말 — 을 따로 센다.
    spoken 기록이 없는 이전 실행이면 발화 단위 항목은 None.
    """
    def turns(pred):
        return [f"A{i}" for i, t in enumerate(replies, start=1) if pred(t)]

    multi_q = turns(lambda t: t.count("?") >= 2)
    nums = [len(_NUMBER.findall(t)) for t in replies]
    out = [Check("questions_per_turn", not multi_q, f"질문 2개 이상 {len(multi_q)}턴 {multi_q}" if multi_q else ""),
           Check("numbers_per_turn", None,
                 f"최대 {max(nums)}개 (A{nums.index(max(nums)) + 1})" if nums and max(nums) else "")]

    spoken = [e for e in trace if e.get("kind") == "spoken"]
    if not spoken:
        return out + [Check("utterance_len", None, "spoken 기록 없음"), Check("spoken_unsaid", None, "spoken 기록 없음")]
    long = [f"{e['turn']}턴 {len(e['data']['text'])}자" for e in spoken
            if len(e["data"]["text"]) > MAX_UTTERANCE_CHARS]
    # 고객에게 나갔지만 채점 대상 응답(reply.text)에는 없는 말 — 절차 강제·도구 차단으로 버려진 스텝
    unsaid = [f"{e['turn']}턴: {e['data']['text'][:40]}" for e in spoken
              if 0 < e["turn"] <= len(replies) and e["data"]["text"].strip() not in replies[e["turn"] - 1]]
    longest = max(len(e["data"]["text"]) for e in spoken)
    out.append(Check("utterance_len", not long, f"최대 {longest}자" + (f", {MAX_UTTERANCE_CHARS}자 초과 {long}" if long else "")))
    out.append(Check("spoken_unsaid", not unsaid, "; ".join(unsaid)))
    return out


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
    """연속된 두 응답이 거의 같은 문장인가 (문자열 유사도 — 의미 판단 아님).

    고객도 직전과 거의 같은 말을 했으면 세지 않는다 — 같은 질문에 같은 답은 정상이다
    (전화에서는 못 들었다는 뜻일 때가 많다. 2026-10-14 s1_b0_tracking: 시뮬레이터가 같은 질문을 글자 그대로 반복).
    """
    exchanges, asked = [], None     # (직전 고객 말, 상담원 말)
    for role, text in tr.turns:
        if role == "customer":
            asked = text
        else:
            exchanges.append((asked or "", text))
    similar = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio() >= REPEAT_SIMILARITY
    pairs = [(a, b) for (qa, a), (qb, b) in zip(exchanges, exchanges[1:])
             if similar(a, b) and not similar(qa, qb)]
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

[근거: 상담원이 받은 업무 지시 — 회사가 정한 절차·안내이므로 근거로 인정한다]
{instructions}

[상담원 발화]
{say}

[출력] JSON 하나: {{"unsupported": "YES" 또는 "NO", "claim": "뒷받침되지 않는 주장 (없으면 빈 문자열)"}}"""


_SIM_FAITHFUL = """\
너는 평가용 고객 시뮬레이터의 감시자다. 시뮬레이터는 아래 [설정]대로만 고객을 연기해야 한다.
고객 발화 중에 [설정]에 없는 구체적 사실을 지어낸 것이 있는지 판정한다.
구체적 사실 = 이름·번호·날짜·금액·물건, 이미 한 행동(신고했다, 연락했다 등), 조회 결과.
설정에서 바로 따라 나오는 말(예: 중고거래 판매자는 모르는 사람이다), 감정 표현, 질문, 동의·거절은 지어낸 것이 아니다.

[설정]
목표: {goal}
알고 있는 정보: {facts}
행동 지침: {behavior_note}

[고객 발화]
{utterances}

[출력] JSON 하나: {{"invented": "YES" 또는 "NO", "what": "지어낸 사실과 몇 번째 발화인지 (없으면 빈 문자열)"}}"""

# 상담원 연결로 끝난 통화의 목표 판정 — 연결 자체가 올바른 처리인 경우 (사기 의심, 전화로 처리 불가 등)
_GOAL_TRANSFERRED = ("이 통화는 사람 상담원에게 연결됐다. 상담원이 직접 처리할 수 없거나 연결이 필요한 이유를 "
                     "고객에게 알리고 연결했는가? (연결된 상담원이 이어서 처리하므로, 그 뒤 절차까지 안내할 필요는 없다)")
_GOAL = ("고객의 통화 목적이 해결됐는가? 정책상 해결할 수 없는 경우라면 "
         "그 이유와 고객이 할 수 있는 다음 행동이 정확히 안내됐는가?")


class LLMJudge:
    def __init__(self, llm: ChatLLM, instructions: Optional[dict[str, str]] = None) -> None:
        """instructions: 워커 이름 → 업무 지시문. 근거 판정에서 그 턴을 맡은 워커의 지시문도 근거로 준다."""
        self.llm = llm
        self.instructions = instructions or {}

    def checks(self, sc: Scenario, tr: Transcript, proc: ProcedureReport) -> list[Check]:
        return self._conversation_checks(sc, tr, proc) + [self._grounded(tr)]

    def diagnostics(self, sc: Scenario, tr: Transcript) -> list[Check]:
        return [self._simulator_faithful(sc, tr)]

    # -- 통화 전체 판정: 필수 내용·금지 언행·목표 해결·실패 후 완료 발언 --

    def _conversation_checks(self, sc: Scenario, tr: Transcript, proc: ProcedureReport) -> list[Check]:
        qs = ([(f"mention_{i}", f"상담원이 고객에게 다음 내용을 전달했는가? → {m}", True)
               for i, m in enumerate(sc.expect.must_mention)]
              + [(f"must_not_{i}", f"상담원이 다음 말이나 행동을 했는가? → {m}", False)
                 for i, m in enumerate(sc.expect.must_not)]
              + [("goal_resolved", _GOAL_TRANSFERRED if _transfer_expected(sc, tr) else _GOAL, True)]
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
        workers_by_turn: dict[int, list[str]] = {}
        for e in tr.state.trace.of("llm"):
            workers_by_turn.setdefault(e.turn, [])
            if e.data.get("role") not in workers_by_turn[e.turn]:
                workers_by_turn[e.turn].append(e.data.get("role"))
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
                instructions="\n\n".join(self.instructions[w] for w in workers_by_turn.get(turn, [])
                                         if w in self.instructions) or "(없음)",
                say=reply.text)
            try:
                out = parse_json(self.llm.chat([{"role": "user", "content": prompt}],
                                               max_tokens=200, json_mode=True).text)
            except (ValueError, json.JSONDecodeError):
                return Check("grounded", None, f"{turn}턴 판정 파싱 실패")
            if str(out.get("unsupported", "")).upper().startswith("Y"):
                bad.append(f"{turn}턴: {out.get('claim', '')}")
        return Check("grounded", not bad, "; ".join(bad) or "모든 턴 근거 있음")


    # -- 진단: 시뮬레이터가 설정에 없는 사실을 지어냈나 --

    def _simulator_faithful(self, sc: Scenario, tr: Transcript) -> Check:
        c = sc.customer
        utterances = [t for r, t in tr.turns if r == "customer"]
        prompt = _SIM_FAITHFUL.format(
            goal=c.get("goal", ""), facts=c.get("facts", {}), behavior_note=c.get("behavior_note", "(없음)"),
            utterances="\n".join(f"{i}. {u}" for i, u in enumerate(utterances, start=1)))
        try:
            out = parse_json(self.llm.chat([{"role": "user", "content": prompt}],
                                           max_tokens=200, json_mode=True).text)
        except (ValueError, json.JSONDecodeError):
            return Check("sim_faithful", None, "판정 파싱 실패")
        invented = str(out.get("invented", "")).upper().startswith("Y")
        return Check("sim_faithful", not invented, out.get("what", "") if invented else "")


def _transfer_expected(sc: Scenario, tr: Transcript) -> bool:
    """상담원 연결로 끝났고, 이 시나리오에서 연결이 허용된 결과인가 (기대 결과를 따로 정하지 않았으면 허용)."""
    allowed = sc.expect.dispositions
    return tr.disposition == "transferred" and (not allowed or "transferred" in allowed)


def score(sc: Scenario, tr: Transcript, tools: ToolRegistry,
          judge: Optional[LLMJudge] = None) -> Score:
    proc = audit(tr.state.trace, tools)
    checks = rule_checks(sc, tr, proc) + (judge.checks(sc, tr, proc) if judge else [])
    diagnostics = speech_diagnostics([r.text for r in tr.replies], tr.state.trace.to_dicts())
    return Score(sc.id, checks, proc, diagnostics=diagnostics + (judge.diagnostics(sc, tr) if judge else []))
