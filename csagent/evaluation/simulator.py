"""고객 시뮬레이터 — 시나리오의 고객 역할로 다음 발화를 만든다.

- ScriptedCustomer: 정해진 발화를 순서대로 (테스트·재현용)
- LLMCustomer: LLM이 페르소나·목표·행동 지침대로 말한다. 전화 말투(STT 결과처럼 구두점 적게).
STT 오류 주입은 noise 훅으로 붙인다 (발화 텍스트 → 오류가 섞인 텍스트).
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from csagent.evaluation.scenario import Scenario
from csagent.llm import ChatLLM, parse_json

Noise = Callable[[str], str]


class CustomerSimulator(Protocol):
    satisfied: Optional[bool]  # 고객이 끊을 때 목표가 해결됐다고 느꼈는지 (모르면 None)

    def next(self, scenario: Scenario, transcript: Sequence[tuple[str, str]]) -> Optional[str]:
        """다음 고객 발화. None이면 고객이 전화를 끊는다."""
        ...


class ScriptedCustomer:
    def __init__(self, lines: Sequence[str], satisfied: Optional[bool] = None) -> None:
        self.lines = list(lines)
        self.satisfied = satisfied

    def next(self, scenario, transcript) -> Optional[str]:
        return self.lines.pop(0) if self.lines else None


_SYSTEM = """\
너는 토스 고객센터에 전화한 고객을 연기한다. 상담원(AI)과 통화 중이다.

[너의 상황]
- 목표: {goal}
- 성격·말투: {persona}
- 알고 있는 정보(상담원이 물어보면 말한다): {facts}
- 행동 지침: {behavior_note}

[말하기 규칙]
- 한국어로만 말한다. 실제 전화처럼 짧게 1~2문장. 구어체, 구두점은 적게. 상담원 말을 반복하지 않는다.
- 묻지 않은 정보를 한꺼번에 다 말하지 않는다.
- [너의 상황]에 없는 사실(배송 방식, 날짜, 판매자와의 연락, 조회 결과 등)을 지어내지 않는다. 모르면 모른다고 한다.
- 상담원이 같은 말을 되풀이하며 진전이 없으면, 한 번 더 요청해 보고 그래도 같으면 끊는다.
- 목표가 해결됐거나, 더 이상 진전이 없거나, 상담원이 통화를 마무리하면 짧게 인사하고 끝낸다.

[출력] JSON 하나: {{"utterance": "할 말", "hang_up": true/false, "satisfied": true/false}}
hang_up=true면 이번 말이 마지막이다. satisfied는 끊을 때 목표가 해결됐다고 느끼는지."""


class LLMCustomer:
    def __init__(self, llm: ChatLLM, noise: Optional[Noise] = None, *, temperature: float = 0.7) -> None:
        self.llm = llm
        self.noise = noise
        self.temperature = temperature
        self.satisfied: Optional[bool] = None
        self._done = False

    def next(self, scenario: Scenario, transcript: Sequence[tuple[str, str]]) -> Optional[str]:
        if self._done:
            return None
        c = scenario.customer
        if not transcript and c.get("opening"):
            return self._noisy(c["opening"])
        msgs = [{"role": "system", "content": _SYSTEM.format(
            goal=c.get("goal", ""), persona=c.get("persona", "평범한 고객"),
            facts=c.get("facts", {}), behavior_note=c.get("behavior_note", "협조적으로 답한다"))}]
        # 시뮬레이터 입장에서는 상담원 말이 user, 자기 말이 assistant
        for role, text in transcript:
            msgs.append({"role": "assistant" if role == "customer" else "user", "content": text})
        out = parse_json(self.llm.chat(msgs, max_tokens=150, temperature=self.temperature,
                                       json_mode=True).text)
        self._done = bool(out.get("hang_up"))
        if self._done:
            self.satisfied = bool(out.get("satisfied"))
        return self._noisy((out.get("utterance") or "").strip()) or None

    def _noisy(self, text: str) -> str:
        return self.noise(text) if self.noise else text
