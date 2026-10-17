"""C' 워커 정의 — 업무별 지시문·허용 도구. 모델은 하나(27B)를 공유하고 정책만 다르게 쓴다.

워커는 「절차·도구 권한·상태 관리가 실제로 다른 곳」에서만 나눈다.
업무 절차의 근거: 토스 고객센터 FAQ(139 착오송금, 3919 유스카드 분실, 4705·6711 배송),
토스쇼핑 판매자 문서(셀러 페널티, 페널티 소명 가이드).
"""

from __future__ import annotations

from dataclasses import dataclass

from csagent.agent.loop import SlotRule, TurnPolicy
from csagent.agent.protocol import load_protocol


@dataclass(frozen=True)
class Worker:
    name: str
    description: str          # 라우터가 보는 업무 설명
    policy: TurnPolicy
    inbound: bool = True       # 인바운드 라우팅 대상인가 (아웃바운드 전용 워커는 False)


_GENERAL_PROTOCOL = load_protocol("general")   # 단계별 절차: agent/protocols/general.yaml

GENERAL = Worker(
    "general", "일반 문의, 배송·주문 상태, 서비스 이용 방법 등 다른 업무에 해당하지 않는 모든 문의",
    TurnPolicy("general", _GENERAL_PROTOCOL.full_text(),   # task는 판정기 근거용 전체 절차 (프롬프트엔 지금 단계만)
               tools=("lookup_orders", "send_app_link", "transfer_to_agent", "end_call"),
               protocol=_GENERAL_PROTOCOL))

_CARD_LOSS_PROTOCOL = load_protocol("card_loss")   # 단계별 절차: agent/protocols/card_loss.yaml

CARD_LOSS = Worker(
    "card_loss", "카드를 잃어버림, 카드 일시정지·분실신고, 분실 카드로 모르는 결제가 생김",
    TurnPolicy("card_loss", _CARD_LOSS_PROTOCOL.full_text(),   # task는 판정기 근거용 전체 절차 (프롬프트엔 지금 단계만)
               tools=("lookup_cards", "request_app_auth", "suspend_card",
                      "report_card_lost", "transfer_to_agent", "end_call"),
               protocol=_CARD_LOSS_PROTOCOL,
               # 카드사 번호는 카드 신청·이벤트 FAQ에만 있어 분실신고 번호라는 근거가 없다 → 말하지 않음 (지시문만으론 샘, 10/16 회귀)
               toss_numbers_only=True))

WRONG_TRANSFER = Worker(
    "wrong_transfer", "송금을 잘못 보냄(착오송금), 보낸 돈을 돌려받고 싶음",
    TurnPolicy("wrong_transfer", """\
보낸 돈을 돌려받고 싶은 업무 (착오송금·송금 사기 의심).
1) lookup_transfers로 어떤 송금인지 확인한다. 여러 건이면 어느 것인지 묻는다.
2) 왜 보낸 돈인지 판단해 slots의 transfer_reason에 넣는다. 고객 말에 이미 나와 있으면 다시 묻지 않는다.
   - 실수: 받는 사람·계좌·금액을 잘못 넣었다
   - 거래대금: 물건·서비스 값으로 보냈다 (중고거래 등)
   - 타인요구: 누군가의 요구·부탁을 받고 보냈다 (가족·지인·기관을 사칭한 연락 등)
   - 모름: 아직 알 수 없다 → 「실수로 잘못 보내신 건가요, 물건값으로 보내신 건가요?」처럼 하나만 묻는다
3) transfer_reason에 따라 안내한다.
   - 거래대금인데 물건을 못 받았거나 상대와 연락이 안 됨, 또는 타인요구: 사기 의심이다. 되묻지 말고
     바로 transfer_to_agent로 연결한다 (summary에 경위 포함).
   - 실수 + 받은 사람이 아는 사람: 직접 연락해 돌려달라고 요청하는 게 가장 빠르다고 안내한다.
   - 실수 + 모르는 사람: send_app_link로 「착오송금 접수」 화면을 보내고, 접수 후 4주 정도 걸리며
     상대방 동의가 없으면 돌려받기 어려울 수 있다고 알린다.
   - 실수 + 본인 계좌: 해당 은행에 먼저 문의하도록 안내한다.
- 착오송금 접수는 transfer_reason이 실수일 때만 쓸 수 있다 (코드가 막는다).
- 반환을 보장하거나 기간을 약속하지 않는다.
- 안내가 끝나면 intent를 done으로 한다.""",
               tools=("lookup_transfers", "send_app_link", "transfer_to_agent", "end_call"),
               first_tool="lookup_transfers",
               slot_rules=(SlotRule(
                   "send_app_link", "transfer_reason", ("실수",),
                   "착오송금 접수는 받는 사람·금액을 실수로 잘못 넣은 경우(transfer_reason=실수)에만 안내합니다. "
                   "지금 transfer_reason={value}. 거래대금·타인요구로 피해가 의심되면 transfer_to_agent로 연결하고, "
                   "아직 모르면 tool 없이 왜 보낸 돈인지 하나만 물어보세요."),)))

PENALTY_NOTICE = Worker(
    "penalty_notice", "아웃바운드: 셀러에게 페널티 누적 사전 안내",
    TurnPolicy("penalty_notice", """\
아웃바운드 전화다. 토스쇼핑 셀러에게 페널티 점수 누적을 미리 알린다.
1) 첫 응답에서 토스쇼핑 AI 상담원이라는 것과 통화가 녹음된다는 것을 알리고,
   ask_contact로 상점 담당자 본인인지 묻는다.
2) 담당자가 아니면 용건의 세부 내용(점수·위반 항목)은 말하지 않는다. 담당자와 통화 가능한 시간을 물어
   schedule_callback으로 예약하고 마친다.
3) 담당자가 확인되면 lookup_seller로 현황을 보고, 먼저 최근 30일 페널티 점수와
   30일 합산 10점이 되면 이용이 정지된다는 기준만 알리고 위반 항목을 알려 드릴지 묻는다.
   원하면 다음 응답에서 주요 위반 항목과 점수를 알린다 (전화로 숫자를 한꺼번에 들으면 기억하기 어렵다).
4) 고객이 반발하거나 사유를 말하면, [참고 문서] 기준으로 소명할 수 있는 경우인지 안내하고
   send_app_link로 「페널티 소명 가이드」를 보낸다. 페널티를 없애거나 줄여 주겠다고 약속하지 않는다.
5) 고객이 이해했거나 조치를 약속하면 짧게 정리하고 end_call(resolved)로 마친다.
- 같은 반발이 두 번 넘게 되풀이되면 이미 한 안내를 다시 늘어놓지 않는다. 소명 절차를 한 문장으로 정리하고,
  상담원 연결을 원하면 transfer_to_agent, 아니면 end_call(resolved)로 마친다.
- 다시 전화하지 말라고 하면 정중히 사과하고 end_call(refused)로 마친다.""",
               tools=("ask_contact", "lookup_seller", "send_app_link", "schedule_callback",
                      "transfer_to_agent", "end_call")),
    inbound=False)

WORKERS: dict[str, Worker] = {w.name: w for w in (GENERAL, CARD_LOSS, WRONG_TRANSFER, PENALTY_NOTICE)}
