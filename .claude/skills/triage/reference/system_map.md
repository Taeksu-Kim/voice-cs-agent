# 시스템 지도 — triage가 원인을 찾을 때 보는 곳

## 에이전트가 말을 만드는 경로

```
고객 말 → 라우터(csagent/agent/router.py, 업무 판별) → 업무 워커(csagent/agent/workers/specs.py)
  → 검색(csagent/retrieval/, 근거 문서) → LLM 스텝 반복(csagent/agent/loop.py: say·tool·intent JSON)
  → 실행 게이트(csagent/agent/guardrails/gate.py) → 도구(MCP 도구 서버 → 도메인 API)
  → 말 걸러내기(guardrails/contacts.py 번호, links.py 링크) → 고객이 들은 말(trace spoken)
```

| 무엇 | 어디 | 원인 층 |
|---|---|---|
| 공통 규칙 (모든 업무) | `csagent/agent/prompts.py` `SYSTEM` | 지시문 |
| 업무별 지시 (공통 + 단계별 say) | `csagent/agent/protocols/<업무>.yaml` — `common`, `steps[].say`, 머리 주석에 과거 실패와 결정 | 지시문 |
| 업무 정의·허용 도구·슬롯 규칙 | `csagent/agent/workers/specs.py` | 코드 |
| 스텝 루프·말 이어 붙이기·대체 문장 | `csagent/agent/loop.py` (`FALLBACK`, `TOOL_STUCK`, 완료 한마디) | 코드 |
| 실행 허락 (인증·고객 확인·결과 확인) | `csagent/agent/guardrails/gate.py` | 코드 |
| 도구 정의 (이름·인자·위험도) | `catalog/*.yaml` | 코드 |
| 도구 동작·모의 데이터 | `csagent/domain/` (`store.py`, 안내 항목 `guides.yaml`) | 코드 / 가이드 |
| 근거 문서 (FAQ·판매자 문서) | 검색 색인 (원문은 `data/`, 저장소 밖) — 그 턴에 들어온 문서는 trace `retrieval.texts` | 문서 |
| 시나리오 기대값 | `scenarios/`, `scenarios_heldout/` (yaml `expect`) | 판정기(채점 기준) |
| 판정기 | `csagent/evaluation/scoring.py` (규칙 채점·LLM Judge), `grounding.py` (근거 판정 v4) | 판정기 |
| 고객 시뮬레이터 | `csagent/evaluation/simulator.py` | 환경 |

설계 결정과 과거 실패의 기록: `docs/09_오동작_가드레일.md`(가드레일 후보·선별 이유 — 「실제로 일어났고 논리적으로 필요한 것만」),
`docs/07_실험_Judge_신뢰도.md`(판정기 신뢰도·오판 사례), `docs/10_관측과_재현.md`(좌표·재현기·이상 탐지).

## 기록 형식

- 케이스 묶음 (`scripts/detect.py`가 만든다): `case.json` · `trace.md` · `llm_calls.jsonl` · `repro.yaml`
- trace 이벤트: `{turn, kind, data, at, where}`. `at` = `t<턴>.s<스텝>.<종류>` (s0 = LLM 전), `where` = `업무/단계`
  - 주요 종류: `turn_start`(고객 말) · `state`(턴 시작 상태) · `route` · `retrieval`(문서) · `llm`(call_id) · `tool_request` · `gate`(allowed·reason) · `tool_result`(ok) · `spoken`(들은 말) · `unheard_drop`(막혀서 안 나간 말) · `execution_unknown` · `turn_error`
- `llm_calls.jsonl` 한 줄 = LLM 호출 하나: `call_id`, `messages`(입력 전체), 출력 텍스트. trace `llm` 이벤트의 `call_id`로 찾는다
- 대화 이력(`turns` / `history`)은 **고객이 실제로 들은 말** 기준 (10/19부터)

## 쓸 수 있는 명령 (프로젝트 루트에서, python = `/home/ubuntu/miniconda3/envs/customer-service-agent/bin/python`)

```
python scripts/repro.py <repro.yaml>                     # 명세대로 N번 → 실패 수 (runs/repro/ 에 결과)
python scripts/repro.py <case> --at t3 --mode turn --repeat 5 --fail repeat=true
python scripts/trace_graph.py <출처 폴더> <키>           # mermaid 흐름도 다시 그리기
python scripts/triage_save.py <케이스 폴더> < 보고서     # 보고서 저장 (형식 검사)
```

재현 모드: `step`(그 스텝 LLM 입력 그대로 — 모델 흔들림만) · `turn`(지금 코드로 그 턴) · `continue`(그 턴부터 끝까지, 고객 `recorded`|`simulator`).
술어: `len_over` · `repeat` · `regex` · `tool` · `no_tool` · `event` · `gate` · `disposition` · `check`(continue만). step은 `len_over`·`regex`·`tool`·`no_tool`만.
10/21 전 기록(턴 시작 상태 없음)은 step 모드만 된다.

모델 서버가 떠 있어야 재현이 된다: `bash scripts/serve.sh status` (꺼져 있으면 재현 없이 확신도 = 가설로 보고).
