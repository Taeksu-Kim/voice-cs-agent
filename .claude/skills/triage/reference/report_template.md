# 보고서 형식

맨 위 `---` 사이 블록(YAML)은 저장 스크립트가 검사하고 누적 목록에 문제마다 한 줄로 옮긴다. 필드 이름·값 종류를 바꾸지 않는다.
아래 본문은 사람이 읽는다 — 결론이 위, 근거가 아래.

**문제 하나 = issues 항목 하나.** 한 케이스에 원인이 다른 문제가 여럿이면(예: 반복과 150자 초과) 항목을 나눈다.
`secondary`는 **같은 문제**에 원인이 하나 더 걸려 있을 때만 쓴다 — 다른 문제를 넣는 칸이 아니다.

```markdown
---
case: runs/20261021_010852_C/s3_b0_defect:0          # case.json의 case
issues:
  - signal: repeat @t3                                # case.json 신호 (종류 @좌표)
    summary: 고객이 새 질문을 붙여도 앞 턴 거절 문장을 그대로 되풀이한다   # 한 줄 요약
    layer: prompt                                     # prompt | guide | doc | code | judge | env
    secondary: ""                                     # 같은 문제의 보조 원인 층 (없으면 "")
    confidence: reproduced                            # reproduced (재현됨) | hypothesis (가설)
    baseline: "5/5"                                   # 수정 전 재현 실패 수 (가설이면 "")
    spec: runs/repro/20261025_120151_s3_b0_defect_r0_t3_turn/spec.json   # 기준 재현 명세 = 재현 결과 폴더의 spec.json
    proposal: 반복 금지 지시를 상담원 자기 말 기준으로 바꾼다          # 개선 방향 한두 문장
    next: apo                                         # apo | human | judge_set | none
  - signal: over_len @t1
    summary: …
    layer: code
    …
---

## 문제 1 — repeat @t3
### 한 줄 요약
### 개선 제안
- (층에 맞는 제안. guide면 사람이 답할 질문 형태)
### 근거
- 그 지점의 LLM 입력에 있던 것 / 없던 것 (llm_calls.jsonl 원문 인용, call_id)
- 들은 말 (trace spoken 인용)
### 재현
- 명세 (at · mode · fail_if · customer) · 결과 (N회 중 실패 수, runs/repro/...)
### 판단 이유
- 왜 이 층인가, 다른 층이 아닌 이유

## 문제 2 — over_len @t1
(같은 구성)

## 같은 원인의 다른 케이스
- (runs/triage/index.jsonl에서 찾은 것, 없으면 "없음")

## 더 볼 것
- (가설이면 무엇을 보면 가려지는지, 확인 못 한 것)
```
