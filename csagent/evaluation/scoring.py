"""채점 — 규칙 채점(코드) + 내용 채점(LLM Judge).

LLM은 항목마다 YES/NO만 판정하고, 점수 계산은 코드가 한다
(LLM이 1~5점을 직접 매기면 같은 입력 재채점 시 점수가 흔들린다 — 현 직장 실측 22.5%→2.5%).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from csagent.evaluation.runner import Transcript
from csagent.evaluation.scenario import Scenario
from csagent.llm import ChatLLM, parse_json

MAX_REPLY_CHARS = 150
_NOT_SPEAKABLE = re.compile(r"https?://|\n\s*[-*•]|\n\s*\d+\.|[()\[\]]")


@dataclass
class Check:
    name: str
    passed: Optional[bool]       # None = 해당 없음
    detail: str = ""


@dataclass
class Score:
    scenario_id: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]


# ---------------------------------------------------------------- 규칙 채점


def rule_checks(sc: Scenario, tr: Transcript) -> list[Check]:
    e, tools = sc.expect, tr.tool_names()
    checks = [
        Check("disposition", tr.disposition in e.dispositions if e.dispositions else None,
              f"{tr.disposition} (기대 {e.dispositions})"),
        Check("tools_required", all(t in tools for t in e.tools_required) if e.tools_required else None,
              f"호출 {tools}"),
        Check("tools_forbidden", not any(t in tools for t in e.tools_forbidden)
              if e.tools_forbidden else None, f"호출 {tools}"),
        Check("gold_doc_retrieved", bool(set(e.gold_docs) & tr.retrieved_docs()) if e.gold_docs else None,
              f"정답 {e.gold_docs}"),
    ]
    long = [r.text for r in tr.replies if len(r.text) > MAX_REPLY_CHARS]
    unspeakable = [r.text for r in tr.replies if _NOT_SPEAKABLE.search(r.text)]
    checks.append(Check("voice_length", not long, f"{MAX_REPLY_CHARS}자 초과 {len(long)}턴"))
    checks.append(Check("voice_format", not unspeakable, f"낭독 부적합 {len(unspeakable)}턴"))
    checks.append(Check("finished", tr.end_reason != "max_turns", tr.end_reason))
    return checks


# ---------------------------------------------------------------- 내용 채점 (LLM Judge)

_JUDGE = """\
너는 고객센터 통화 품질 평가자다. 아래 통화 기록을 읽고 각 질문에 YES 또는 NO로만 판정한다.
상담원이 실제로 말한 내용만 근거로 삼는다. 애매하면 NO.

[통화 기록]
{transcript}

[질문]
{questions}

[출력] JSON 하나: {{"answers": [{{"id": "질문 id", "answer": "YES" 또는 "NO", "reason": "한 문장"}}]}}"""


class LLMJudge:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def checks(self, sc: Scenario, tr: Transcript) -> list[Check]:
        qs = ([(f"mention_{i}", f"상담원이 고객에게 다음 내용을 전달했는가? → {m}", True)
               for i, m in enumerate(sc.expect.must_mention)]
              + [(f"must_not_{i}", f"상담원이 다음 말이나 행동을 했는가? → {m}", False)
                 for i, m in enumerate(sc.expect.must_not)])
        if not qs:
            return []
        prompt = _JUDGE.format(
            transcript="\n".join(f"{'고객' if r == 'customer' else '상담원'}: {t}" for r, t in tr.turns),
            questions="\n".join(f"- {qid}: {q}" for qid, q, _ in qs))
        try:
            answers = {a["id"]: a for a in parse_json(self.llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=600, json_mode=True).text)["answers"]}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return [Check(qid, None, "판정 파싱 실패") for qid, _, _ in qs]
        out = []
        for qid, q, want_yes in qs:
            a = answers.get(qid, {})
            yes = str(a.get("answer", "")).upper().startswith("Y")
            out.append(Check(qid, yes if want_yes else not yes, f"{q} / {a.get('reason', '')}"))
        return out


def score(sc: Scenario, tr: Transcript, judge: Optional[LLMJudge] = None) -> Score:
    checks = rule_checks(sc, tr) + (judge.checks(sc, tr) if judge else [])
    return Score(sc.id, checks)
