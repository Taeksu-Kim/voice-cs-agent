"""테스트 공용 도구 스택 — 세 층(에이전트 ↔ MCP 도구 서버 ↔ 도메인 API)을 한 프로세스에서 실제 경로로 돈다.

통화는 call(ctx)로 연다: 시나리오 업무 데이터(cards·orders 등)를 도메인 API에 심고, 에이전트 컨텍스트에는 이름만 남긴다.
"""

from csagent.evaluation.runner import run_scenario
from csagent.stack import local_stack

STACK = local_stack()
TOOLS = STACK.tools


def call(ctx):
    return STACK.open(ctx)


def run_call(agent, customer, scenario):
    return run_scenario(agent, customer, scenario, STACK)


def looked_up(state, tool="lookup_cards", args=None):
    """이번 통화에서 조회를 이미 한 상태 — 근거 검증(grounded_by)을 통과하는 앞 단계 (실제 흐름도 조회가 먼저다)."""
    r = TOOLS.for_call(state).run(state, tool, args or {})
    state.trace.add(state.turn, "tool_result", {"name": tool, "ok": r.ok, **r.result})
    return state


def fixture(state):
    """도메인 API에 있는 이 통화의 현재 업무 데이터."""
    return STACK.fixtures.get(state.ctx.customer_id)
